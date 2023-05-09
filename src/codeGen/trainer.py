import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.cuda.amp import autocast, GradScaler
import os
from collections import OrderedDict
from tqdm import tqdm
from transformers import pipeline, set_seed
from torch.cuda import OutOfMemoryError
from torchmetrics.functional.text.rouge import rouge_score
import torchmetrics
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.codeGen.config import MAX_SEQUENCE_SIZE, BATCH_SIZE, MODEL_NAME
from src.codeGen.datasets.collate_functor import CollateFunctor
from src.codeGen.datasets.local_dataset.local_dataset import LocalDataset

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gpu_id: int,
        save_every: int,
        eval_every: int,
        total_epochs: int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.eval_every = eval_every
        self.total_epochs = total_epochs
        self.model = DDP(model, device_ids=[gpu_id]) if gpu_id is not None else model

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        # with autocast():
        output = self.model(**source, labels=targets)

        output.loss.backward()
        self.optimizer.step()
        return output.loss.item()

    def _run_epoch(self, epoch):
        self.model.train()
        pbar_prefix = f"[{epoch}/{self.total_epochs}]"
        pbar = tqdm(self.train_data, leave=False)
        epoch_loss = 0
        for (x, _), (y, _) in pbar:
            x = x.to(f"cuda:{self.gpu_id}")
            y = y.to(f"cuda:{self.gpu_id}")
            loss = self._run_batch(x, y)
            pbar.set_description_str(f"{pbar_prefix} - loss = {loss:.4f}")
            epoch_loss += loss
        self.scheduler.step()
        
        return epoch_loss / self.train_data.__len__()

    def _save_current_checkpoint(self, **kwargs):
        ckp = {**kwargs}
        PATH = "/tmp/xsaman02/CodeGen/"
        if not os.path.isdir(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save(ckp, PATH+"codegen.current.pt")
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def _save_evaluated_checkpoint(self, **kwargs):
        ckp = {
                    **kwargs
               }
        PATH = "~/Project/models/codeGen/"
        if not os.path.isdir(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save(ckp, PATH+"codegen.evaluated.pt")

    def train(self, model_d : dict):
        if model_d is None:
            model_d = {"epoch" : 0, "loss_list" : []}
        elif model_d.get("loss_list") is None:
            model_d["loss_list"] = []
            
        for epoch in range(model_d.get("epoch", 0) + 1, self.total_epochs+1):
            model_d["epoch"] = epoch
            model_d["model_dict"] = self.model.state_dict()
            model_d["optimizer_dict"] = self.optimizer.state_dict()
            model_d["scheduler_dict"] = self.scheduler.state_dict()
            
            if epoch % self.eval_every == 0:
                eval_data = self.evaluate()
                if self.gpu_id == 0 or self.gpu_id is None:
                    self._save_evaluated_checkpoint(**eval_data, **model_d)
            
            model_d.get("loss_list").append(self._run_epoch(epoch))            
            if (self.gpu_id == 0 or self.gpu_id is None) and epoch % self.save_every == 0:
                self._save_current_checkpoint(**model_d)
                
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        sources_list = []
        sentences_target = []
        sentences_pred = []
        tokenizer = self.test_data.dataset.datasampler.tokenizer

        test_dataloader = tqdm(self.test_data, leave=True)

        bleu_score = torchmetrics.BLEUScore(tokenizer=tokenizer)
        cur_bleu_score = 0
        skipped = 0
        set_seed(1)
        # generator = pipeline('text-generation', model=self.model.module, tokenizer=tokenizer, device=f"cuda:{self.gpu_id}")
        generator = pipeline('text-generation', model=self.model, tokenizer=tokenizer, device=self.model.device)
        # rouge_score = torchmetrics.text.rouge.ROUGEScore(tokenizer=tokenizer, rouge_keys="rougeL")
        for (x, x_str), (_, y_str) in test_dataloader:
            y_pred = None
            x = x.to(f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cuda:0")
            try:
                y_pred = [sample[0]["generated_text"][len(prompt):].replace("{\t", "{\n\t").replace("}\t", "}\n\t").replace(";\t", ";\n\t").replace("{ ", "{\n ").replace("} ", "}\n ").replace("; ", ";\n ") for prompt, sample in zip(x_str, generator(x_str, max_length=MAX_SEQUENCE_SIZE, num_return_sequences=1))]
                # generated_ids = torch.argmax(self.model(**x).logits, dim=-1)
                # generated_text = generator(x_str, max_length=MAX_SEQUENCE_SIZE, num_return_sequences=1)
            except:
                skipped += 1
                test_dataloader.set_postfix_str(f"Skipped: {skipped}")
                torch.cuda.empty_cache()
            
            if y_pred is None:
                continue
            
            # y_pred = [prompt[len(output):] for prompt, output in zip(x_str, tokenizer.batch_decode(generated_ids, skip_special_tokens=True))]
            # y_pred = [sample[0]["generated_text"] for sample in generated_text]
            
            sources_list.extend(x_str)
            sentences_target.extend(y_str)
            sentences_pred.extend(y_pred)

            y_str = [[y_sentence] for y_sentence in y_str]
            bleu_score.update(y_pred, y_str)
            cur_bleu_score = bleu_score.compute()
            
            test_dataloader.set_description_str("BLEU: {:.3f}".format(cur_bleu_score))
            
            # break
        
        print("BLEU: {:.3f}".format(cur_bleu_score))
        
        return {
            "bleu" : float(cur_bleu_score),
            "source_sentences" : sources_list,
            "target_sentences" : sentences_target,
            "predic_sentences" : sentences_pred
            }



def prepare_dataloader(dataset: Dataset, batch_size: int, collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset),
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    model_path = "/tmp/xsaman02/CodeGen/codegen.current.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    configuration = AutoConfig.from_pretrained(MODEL_NAME)
    configuration.max_length = MAX_SEQUENCE_SIZE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.model_max_length=MAX_SEQUENCE_SIZE
    tokenizer.add_special_tokens({
        "pad_token" : "</s>"
    })
    
    model_dict = torch.load(model_path, map_location="cpu")
    model = AutoModelForCausalLM.from_config(configuration).to(device)
    model_state_dict = OrderedDict()
    for key, val in model_dict["model_dict"].items():
        model_state_dict[".".join(key.split(".")[1:])] = val
    model.load_state_dict(model_state_dict)
    optimizer = transformers.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005, no_deprecation_warning=True)
    optimizer.load_state_dict(model_dict["optimizer_dict"])
    
    scheduler = transformers.get_linear_schedule_with_warmup(                
            optimizer = optimizer,
            num_warmup_steps = 3,
            num_training_steps = 21,
            last_epoch=model_dict.get("epoch", 0) if model_dict.get("epoch", 0) > 0 else -1
    )
    
    if model_dict.get("scheduler_dict") is not None:
        scheduler.load_state_dict(model_dict.get("scheduler_dict"))
        
    train_dataset = LocalDataset(tokenizer, "train")
    valid_dataset = LocalDataset(tokenizer, "valid")
        
    collate_fn = CollateFunctor(tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer, scheduler, None, 1, 1, 21)
    trainer.train(model_dict)
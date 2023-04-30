import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm

from torch.cuda import OutOfMemoryError
from torchmetrics.functional.text.rouge import rouge_score
import torchmetrics

from src.codeGen.config import MAX_SEQUENCE_SIZE


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
        self.model = DDP(model, device_ids=[gpu_id])
        
        self.scaler = GradScaler()

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        with autocast():
            output = self.model(**source, labels=targets)

        loss = output.loss
        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        # loss.backward()
        # self.optimizer.step()
        return loss.item()

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
        
        
        return epoch_loss / self.train_data.__len__()

    def _save_current_checkpoint(self, epoch, **kwargs):
        ckp = {
                    "model_dict" : self.model.module.state_dict(),
                    "optimizer_dict" : self.optimizer.state_dict(),
                    "scheduler_dict" : self.scheduler.state_dict(),
                    "epoch" : epoch,
                    **kwargs
               }
        PATH = "/tmp/xsaman02/CodeGen/"
        if not os.path.isdir(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save(ckp, PATH+"codegen.current.pt")
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def _save_best_checkpoint(self, epoch, eval_data, **kwargs):
        ckp = {
                    "model_dict" : self.model.module.state_dict(),
                    "optimizer_dict" : self.optimizer.state_dict(),
                    "scheduler_dict" : self.scheduler.state_dict(),
                    "epoch" : epoch,
                    **eval_data
                    **kwargs
               }
        PATH = "/tmp/xsaman02/CodeGen/"
        if not os.path.isdir(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save(ckp, PATH+"codegen.best.pt")

    def train(self, model_d : dict):
        epoch_loss_list = []
        last_epoch = 1
        if model_d is not None:
            epoch_loss_list = model_d.get("loss_list", [])
            last_epoch = model_d.get("epoch", 0) + 1
            
        for epoch in range(last_epoch, self.total_epochs+1):
            if epoch % self.eval_every == 0 and epoch > 1:
                eval_data = self.evaluate()
                if self.gpu_id == 0:
                    self._save_best_checkpoint(epoch, eval_data, loss_list=epoch_loss_list)
            
            epoch_loss_list.append(self._run_epoch(epoch))
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_current_checkpoint(epoch, loss_list=epoch_loss_list)
                
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        sources_list = []
        sentences_target = []
        sentences_pred = []
        tokenizer = self.test_data.dataset.datasampler.tokenizer

        test_dataloader = tqdm(self.test_data, leave=False)

        bleu_score = torchmetrics.BLEUScore(tokenizer=tokenizer)
        cur_bleu_score = 0
        cur_rouge_score = 0
        
        # rouge_score = torchmetrics.text.rouge.ROUGEScore(tokenizer=tokenizer, rouge_keys="rougeL")
        for (x, x_str), (_, y_str) in test_dataloader:
            generated_ids = None
            x = x.to(f"cuda:{self.gpu_id}")
            while True:
                try:
                    generated_ids = self.model.module.generate(**x, num_beams=1, min_length=0, do_sample=False, max_new_tokens=MAX_SEQUENCE_SIZE)
                    # generated_text = generator(x_str, max_length=MAX_SEQUENCE_SIZE, num_return_sequences=1)
                except Exception as e:
                    if type(e) != OutOfMemoryError:
                        raise e
                    torch.cuda.empty_cache()
                    continue
                break
            y_pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # y_pred = [sample[0]["generated_text"] for sample in generated_text]
            
            sources_list.extend(x_str)
            sentences_target.extend(y_str)
            sentences_pred.extend(y_pred)
            cur_rouge_score = rouge_score(sentences_pred, sentences_target, tokenizer=tokenizer, rouge_keys="rougeL")["rougeL_recall"]

            y_str = [[y_sentence] for y_sentence in y_str]
            bleu_score.update(y_pred, y_str)
            cur_bleu_score = bleu_score.compute()
            
            test_dataloader.set_description("BLEU: {:.3f}, ROUGE: {:.3f}".format(cur_bleu_score, cur_rouge_score))
            
            # break
        
        print("BLEU: {:.3f}, ROUGE: {:.3f}".format(cur_bleu_score, cur_rouge_score))
        
        return {
            "bleu" : float(cur_bleu_score), 
            "rouge" : float(cur_rouge_score), 
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


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='simple distributed training job')
#     parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
#     parser.add_argument('save_every', type=int, help='How often to save a snapshot')
#     parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
#     args = parser.parse_args()
    
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
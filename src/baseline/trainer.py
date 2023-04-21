import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm

from torch.cuda import OutOfMemoryError
from torchmetrics.functional.text.rouge import rouge_score
import torchmetrics

from src.codeGen.config import MAX_SEQUENCE_SIZE

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
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
        self.save_every = save_every
        self.eval_every = eval_every
        self.total_epochs = total_epochs
        self.model = model

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        while True:
            try:
                output = self.model(**source, labels=targets)
            except Exception as e:
                if type(e) != OutOfMemoryError:
                    raise e
                torch.cuda.empty_cache()
                continue
            break
        
        loss = output.loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        self.model.train()
        pbar_prefix = f"[{epoch}/{self.total_epochs}]"
        pbar = tqdm(iter(self.train_data), leave=False)
        epoch_loss = 0
        for (x, _), (y, _) in pbar:
            loss = self._run_batch(x, y)
            pbar.set_description_str(f"{pbar_prefix} - loss = {loss:.4f}")
            epoch_loss += loss
        
        return epoch_loss / self.train_data.__len__()

    def _save_current_checkpoint(self, epoch, **kwargs):
        ckp = {
                    "model_dict" : self.model.module.state_dict(),
                    "optimizer_dict" : self.optimizer.state_dict(),
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
            last_epoch = model_d.get("epoch", 1)
            
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
        sampler=dataset,
        collate_fn=collate_fn
    )

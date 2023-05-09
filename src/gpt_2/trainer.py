import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm

from torch.cuda import OutOfMemoryError
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.functional.text.rouge import rouge_score
import torchmetrics
from transformers import pipeline, set_seed

from src.gpt_2.config import MAX_SEQUENCE_SIZE
from src.gpt_2.datasets.config import DEVICE


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        eval_every: int,
        total_epochs: int
    ) -> None:
        self.model = model.to(DEVICE)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_every = eval_every
        self.total_epochs = total_epochs
        self.model = model
        
        self.loss_count = 0
        self.scaler = GradScaler()

    def _run_batch(self, x, y, cumulative_loss_step=3):
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        # x_ids, x_mask = ({**x}.values())
        # y_ids, y_mask = ({**y}.values())
        preds = None
        while True:
            try:
                with torch.autocast(DEVICE if DEVICE == "cpu" else "cuda"):
                    preds = self.model(**x, labels=y)

                # with torch.autocast(DEVICE if DEVICE == "cpu" else "cuda"):
                #     losses.append(preds.loss)
            except Exception as e:
                if type(e) != OutOfMemoryError:
                    raise e
                torch.cuda.empty_cache()
                continue
            break
        
        # preds.loss.backward()        
        # self.optimizer.step()
        # Backward pass
        # loss = preds.loss / cumulative_loss_step
        preds.loss.backward()
        self.optimizer.step()
        return preds.loss.item()

    def _run_epoch(self, epoch):
        self.model.train()
        pbar_prefix = f"[{epoch}/{self.total_epochs}]"
        pbar = tqdm(self.train_data, leave=True)
        epoch_loss = 0
        for (x, _), (y, _) in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            loss = self._run_batch(x, y)
            pbar.set_description_str(f"{pbar_prefix} - loss = {loss:.4f}")
            epoch_loss += loss
        
        return epoch_loss / self.train_data.__len__()

    def _save_current_checkpoint(self, **kwargs):
        ckp = {
                    **kwargs
               }
        PATH = "/tmp/xsaman02/gpt2/"
        if not os.path.isdir(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save(ckp, PATH+"gpt2.current.pt")
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def _save_evaluated_checkpoint(self, **kwargs):
        ckp = {
                **kwargs    
            }
        
        PATH = "/home/xsaman02/Project/models/gpt2/"
        if not os.path.isdir(PATH):
            os.makedirs(PATH, exist_ok=True)
        torch.save(ckp, PATH+"gpt2.evaluated.pt")

    def train(self, model_d : dict):
        
        if model_d is None:
            model_d = {
            }
        
        if model_d.get("epoch") is None:
            model_d["epoch"] = 0
            
        if model_d.get("loss_list") is None:
            model_d["loss_list"] = []
            
        for epoch in range(model_d.get("epoch")+1, self.total_epochs+1):
            model_d["epoch"] = epoch
            model_d["model_dict"] = self.model.state_dict()
            model_d["optimizer_dict"] = self.optimizer.state_dict()
            
            if epoch % self.eval_every == 0 and epoch > 1:
                eval_data = self.evaluate()
                self._save_evaluated_checkpoint(**eval_data, **model_d)
            
            model_d.get("loss_list").append(self._run_epoch(epoch))
            self.scheduler.step()
            self._save_current_checkpoint(**model_d)
                
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        sources_list = []
        sentences_target = []
        sentences_pred = []
        tokenizer = self.test_data.dataset.datasampler.tokenizer

        test_dataloader = tqdm(self.test_data, leave=True)
        
        set_seed(1)
        generator = pipeline('text-generation', model=self.model, tokenizer=tokenizer, device=DEVICE, max_length=MAX_SEQUENCE_SIZE)


        bleu_score = torchmetrics.BLEUScore(tokenizer=tokenizer)
        cur_bleu_score = 0
        skipped = 0
    
        for (_, x_str), (_, y_str) in test_dataloader:
            # x = x.to(DEVICE)
            y_pred = None
            generated_ids = None
            try:
                y_pred = [sample[0]["generated_text"][len(prompt):] for prompt, sample in zip(x_str, generator(x_str, max_length=MAX_SEQUENCE_SIZE, num_return_sequences=1))]
                
                # y_pred = generator(x_str, max_new_tokens=MAX_SEQUENCE_SIZE, num_return_sequences=1, do_sample=False)
                # generated_ids = self.model.generate(**x, num_beams=1, min_length=0, max_new_tokens=MAX_SEQUENCE_SIZE)
            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                skipped += 1
                test_dataloader.set_postfix_str(f"Skipped: {skipped}")
            
            # break
            
            if generated_ids is None and y_pred is None:
                continue

            # y_pred = [y_pred[len(xs_str):] for xs_str, y_pred in zip(x_str, tokenizer.batch_decode(generated_ids, skip_special_tokens=True))]            
            # y_pred = [ys_pred[0]["generated_text"][len(xs_str):] for xs_str, ys_pred in zip(x_str, y_pred)]
            
            # y_pred = [sample[0]["generated_text"] for sample in generated_text]
            
            sources_list.extend(x_str)
            sentences_target.extend(y_str)
            sentences_pred.extend(y_pred)
            # cur_rouge_score = rouge_score(sentences_pred, sentences_target, tokenizer=tokenizer, rouge_keys="rougeL")["rougeL_recall"]

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
        collate_fn=collate_fn
    )

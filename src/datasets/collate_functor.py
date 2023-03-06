
import torch
import torch.nn.functional as F
import numpy as np

from src.datasets.config import DEVICE, EOS_TOKEN, PAD_TOKEN

class CollateFunctor:
    def __init__(self, tokenizer, max_x : int, max_y : int):
        self.tokenizer = tokenizer
        self.max_x = max_x
        self.max_y = max_y

    def __call__(self, samples: list):
        x_str, y_str = zip(*samples)
        
        x = self.tokenizer(x_str, max_length=self.max_x, padding=True, truncation=True, return_tensors="pt")
        y = self.tokenizer(y_str, max_length=self.max_y, padding=True, truncation=True, return_tensors="pt")
        
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        

        # labels_size = len(batch["labels"][0])
        # ids_size = len(batch["input_ids"][0])
        # max_size = max(labels_size, ids_size)
        
        # batch["labels"] = torch.nn.ConstantPad1d((0, max_size-labels_size), pad_id)(batch["labels"])
        # batch["input_ids"] = torch.nn.ConstantPad1d((0, max_size-ids_size), pad_id)(batch["input_ids"])
        # batch["attention_mask"] = torch.nn.ConstantPad1d((0, max_size-ids_size), 0)(batch["attention_mask"])
        
        
        # batch["labels"] = F.pad(batch["labels"], tuple([max_size-labels_size]), "constant", pad_id)
        # batch["input_ids"] = F.pad(batch["input_ids"], tuple([max_size-ids_size]), "constant", pad_id)
        # batch["attention_mask"] = F.pad(batch["attention_mask"], tuple([max_size-ids_size]), "constant", 0)
        
        # x = self.tokenizer(x_str, padding_side="left", pad_token=self.tokenizer.pad_token, truncation=True, padding=True, max_length=self.max_x, return_tensors="pt")
        # y = self.tokenizer(y_str, padding_side="right", pad_token=self.tokenizer.pad_token, truncation=True, padding=True, max_length=self.max_y, return_tensors="pt")


        adjust_end_ids = torch.where(y["input_ids"][:,-1] != pad_id, eos_id, pad_id)
        y["input_ids"][:,-1] = adjust_end_ids

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        

        return (x, x_str), (y, y_str)

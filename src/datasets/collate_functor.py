
import torch
import torch.nn.functional as F
import numpy as np
from src.model.t5_small.config import MAX_SEQUENCE_SIZE

from src.datasets.config import DEVICE

class CollateFunctor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: list):
        x_str, y_str = zip(*samples)
        
        x = self.tokenizer(x_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt")
        y = self.tokenizer(y_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt")
        
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        adjust_end_ids = torch.where(y["input_ids"][:,-1] != pad_id, eos_id, pad_id)
        y["input_ids"][:,-1] = adjust_end_ids

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        

        return (x, x_str), (y, y_str)

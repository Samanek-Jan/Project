
import torch
import torch.nn.functional as F
import numpy as np
from baseline.config import BATCH_SIZE

from src.baseline.datasets.config import DEVICE
from src.t5_small.config import MAX_SEQUENCE_SIZE

class CollateFunctor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: list):
        
        x_str, y_str = zip(*samples)
        
        x = self.tokenizer(x_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt")
        x["input_ids"] = x["input_ids"].type(torch.long)
        x["attention_mask"] = ~x["attention_mask"].type(torch.bool)
        
        y = self.tokenizer(y_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt")
        y["input_ids"] = y["input_ids"].type(torch.long)
        y["attention_mask"] = ~y["attention_mask"].type(torch.bool)
        
        return (x, x_str), (y, y_str)
        # return {**x, "labels" : y["input_ids"]}

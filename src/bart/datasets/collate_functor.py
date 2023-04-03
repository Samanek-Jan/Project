
import torch
import torch.nn.functional as F
import numpy as np

from src.bart.datasets.config import DEVICE

class CollateFunctor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: list):
        from src.bart.config import MAX_SEQUENCE_SIZE
        
        x_str, y_str = zip(*samples)
        
        x_str = [x_s + self.tokenizer.mask_token for x_s in x_str]
        
        x = self.tokenizer(x_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt")
        y = self.tokenizer(y_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt")

        # x = x
        # y = y
        
        return (x, x_str), (y, y_str)
        # return {**x, "labels" : y["input_ids"]}

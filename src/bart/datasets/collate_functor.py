
import torch
import torch.nn.functional as F
import numpy as np

from src.bart.datasets.config import DEVICE
from src.bart.config import MAX_SEQUENCE_SIZE


class CollateFunctor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: list):
        
        x_str, y_str = zip(*samples)
        
        x = self.tokenizer(x_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        y = self.tokenizer(y_str, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        # x = x
        # y = y
        
        return (x, x_str), (y, y_str)
        # return {**x, "labels" : y["input_ids"]}

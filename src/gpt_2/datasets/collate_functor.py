
from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np

from src.gpt_2.datasets.config import DEVICE
from src.gpt_2.config import MAX_SEQUENCE_SIZE


class CollateFunctor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: list):
        
        x = self.tokenizer(samples, max_length=MAX_SEQUENCE_SIZE, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        y = deepcopy(x["input_ids"][:,1:])
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        y = torch.hstack([y, torch.full((len(y), 1), self.tokenizer.pad_token_id).to(DEVICE)])
        y[y == self.tokenizer.pad_token_id] = -100
        y = y.to(DEVICE)
            
            
        return (x, samples), (y)
        # return {**x, "labels" : y["input_ids"]}

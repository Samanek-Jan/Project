from copy import deepcopy
import torch
import numpy as np
import datasets
import tokenizers
import random

from src.gpt_2.datasets.config import DEVICE
from src.gpt_2.config import MAX_SEQUENCE_SIZE


class LocalDataSampler():
    
    def __init__(self, tokenizer, part : str):
        self.part = part
        self.tokenizer = tokenizer
        
    def __call__(self, kernel):
        return self.sample(kernel)

    def wrap_sample(self, sample : str) -> str:
        return self.tokenizer.bos_token + sample + self.tokenizer.eos_token

    def validate_and_tokenize_kernel(self, kernel : dict) -> str:
        
        x = kernel.get("comment", "") + "\n" + kernel.get("header", "")
        y = kernel.get("body", "")
        
        if self.part == "valid":
            tokenized = self.tokenizer.encode(x+y)
            if len(tokenized) >= MAX_SEQUENCE_SIZE:
                x_tokenized = self.tokenizer.encode(x)
                y_tokenized = self.tokenizer.encode(y)
                s = MAX_SEQUENCE_SIZE
                x_tokenized = x_tokenized[:min(len(x_tokenized, s))]
                s -= len(x_tokenized)
                if s <= 0:
                    return self.tokenizer.decode(x_tokenized), ""
                y_tokenized = y_tokenized[:min(len(y_tokenized, s))]
                return self.tokenizer.decode(x_tokenized), self.tokenizer.decode(y_tokenized)

            return x, y
        else:
            return (x + "\n" + y) , (y + self.tokenizer.eos_token)

    def sample(self, kernel):
        return self.validate_and_tokenize_kernel(kernel)
        
        
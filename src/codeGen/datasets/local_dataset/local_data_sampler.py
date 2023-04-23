from copy import deepcopy
import torch
import numpy as np
import datasets
import tokenizers
from src.codeGen.datasets.config import DEVICE
import random

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
        
        return x, self.wrap_sample(y)
        
        # if self.part == "valid":
        #     return self.tokenizer.bos_token + x, self.wrap_sample(x + "\n" + y)
        # else:
        #     return self.wrap_sample(x + "\n" + y) , y + self.tokenizer.eos_token

    def sample(self, kernel):
        return self.validate_and_tokenize_kernel(kernel)
        
        
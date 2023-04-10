from copy import deepcopy
import torch
import numpy as np
import datasets
import tokenizers
from src.gpt_2.datasets.config import DEVICE
import random

class LocalDataSampler():
    
    def __init__(self, tokenizer, part : str):
        self.kernel_prefix = "Generate CUDA function defined as:\n"
        self.part = part
        self.tokenizer = tokenizer
        
    def __call__(self, kernel):
        return self.sample(kernel)

    def validate_and_tokenize_kernel(self, kernel : dict) -> str:
        if self.part == "valid":
            return self.kernel_prefix + kernel.get("comment", "") + "\n" + kernel.get("header", "")
        else:
            return self.kernel_prefix + kernel.get("comment", "") + "\n" + kernel.get("header", "") + "\n" + kernel.get("body", "")

    def sample(self, kernel):
        return self.validate_and_tokenize_kernel(kernel)
        
        
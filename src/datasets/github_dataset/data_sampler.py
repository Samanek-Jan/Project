import torch
import numpy as np
import datasets
import tokenizers
from src.datasets.tokenizer import CupydTokenizer
from src.datasets.github_dataset.config import *
from src.datasets.interface.config import MASK_TOKEN
import random

class RemoteDataSampler():
    
    def __init__(self, tokenizer : CupydTokenizer, max_x : int, max_y : int, sampling_type : str = "NSP"):
        self.sampling_type = sampling_type
        self.tokenizer = tokenizer
        self.max_x = max_x
        self.max_y = max_y
        
    def __call__(self, kernel):
        if self.sampling_type == "NSP":
            return self.sample_NSP(kernel)
        elif self.sampling_type == "MLM":
            return self.sample_MLM(kernel)
        else:
            raise ValueError("Unknown sampling type: %s" % self.sampling_type)
    
    def validate_and_tokenize_kernel(self, kernel) -> str:
        if type(kernel) != str:
            # kernel = kernel.get("comment", "") + kernel.get("header", "") + kernel.get("body", "")
            kernel = kernel.get("code", None)
            
        kernel_tokens = self.tokenizer.encode(kernel)
        
        if len(kernel_tokens.ids) > self.max_x + self.max_y:
            max_size = self.max_x + self.max_y
            sub_kernel_size = random.randint(max_size//3, max_size)
            start_idx = random.randrange(0, len(kernel_tokens.ids) - sub_kernel_size + 1)
            kernel_tokens.ids = kernel_tokens.ids[start_idx:start_idx + sub_kernel_size]
            kernel_tokens.tokens = kernel_tokens.tokens[start_idx:start_idx + sub_kernel_size]
        
        return kernel_tokens
    
    def sample_MLM(self, kernel):
        kernel_tokens = self.validate_and_tokenize_kernel(kernel)
        number_of_masked_tokens = random.randint(1, len(kernel_tokens.ids)//10)
        random.shuffle(ids := range(kernel_tokens.ids))
        masked_ids = ids[:number_of_masked_tokens]
        np_ids = np.array(kernel_tokens.ids)
        np_tokens = np.array(kernel_tokens.tokens)
        
        mask_token_id = self.tokenizer.token_to_id(MASK_TOKEN)
        
        x_ids = np_ids
        x_ids[masked_ids] = mask_token_id
        x_tokens = np_tokens
        x_tokens[masked_ids] = "[MASKED]"
        
        y_ids = np_ids[masked_ids]
        y_tokens = np_tokens[masked_ids]
        
        return (x_ids, x_tokens), (y_ids, y_tokens)
        
    
    def sample_NSP(self, kernel):
        kernel_tokens = self.validate_and_tokenize_kernel(kernel)
        pivot = random.randint(len(kernel_tokens.ids)//10, len(kernel_tokens.ids)//10*9)
        
        return (kernel_tokens.ids[:pivot], kernel_tokens.tokens[:pivot]), (kernel_tokens.ids[pivot:], kernel_tokens.tokens[pivot:])
            


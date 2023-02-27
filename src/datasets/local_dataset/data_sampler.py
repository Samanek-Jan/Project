import torch
import numpy as np
import datasets
import tokenizers
from src.datasets.tokenizer import CupydTokenizer
from src.datasets.github_dataset.config import *
from src.datasets.interface.config import MASK_TOKEN
import random

class LocalDataSampler():
    
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
            raise ValueError("DataSampler.__call__: Unknown sampling type: %s" % self.sampling_type)

    def validate_and_tokenize_kernel(self, kernel : dict) -> str:

        kernel_x = kernel.get("comment", "") + kernel.get("header", "")
        kernel_y = kernel.get("body", "")
        
        if self.sampling_type == "MLM":
            kernel_tokens = self.tokenizer.encode(kernel_x + kernel_y)
            if len(kernel_tokens.ids) > self.max_x:
                sub_kernel_size = random.randint(self.max_x//3, self.max_x)
                start_idx = random.randrange(0, len(kernel_tokens.ids) - sub_kernel_size + 1)
                kernel_tokens.ids = kernel_tokens.ids[start_idx:start_idx + sub_kernel_size]
                kernel_tokens.tokens = kernel_tokens.tokens[start_idx:start_idx + sub_kernel_size]
            return kernel_tokens.ids, kernel_tokens.tokens
        
        elif self.sampling_type == "NSP":
            kernel_x_tokens = self.tokenizer.encode(kernel_x)
            kernel_y_tokens = self.tokenizer.encode(kernel_y)
            
            if len(kernel_x_tokens.ids) > self.max_x:
                kernel_x_tokens.ids = kernel_x_tokens.ids[len(kernel_x_tokens.ids) - self.max_x:]
                kernel_x_tokens.tokens = kernel_x_tokens.tokens[len(kernel_x_tokens.tokens) - self.max_x:]
            
            if len(kernel_y_tokens.ids) > self.max_y:
                kernel_y_tokens.ids = kernel_y_tokens.ids[:self.max_y]
                kernel_y_tokens.tokens = kernel_y_tokens.tokens[:self.max_y]
            
            return (kernel_x_tokens.ids, kernel_x_tokens.tokens), (kernel_y_tokens.ids, kernel_y_tokens.tokens)
        else:
            raise Exception("DataSampler.validate_and_tokenize_kernel: Unknown sampling type")
    
    def sample_MLM(self, kernel):
        token_ids, tokens = self.validate_and_tokenize_kernel(kernel)
        
        number_of_masked_tokens = random.randint(1, len(token_ids)//10)
        random.shuffle(ids := range(token_ids))
        masked_ids = ids[:number_of_masked_tokens]
        np_ids = np.array(token_ids)
        np_tokens = np.array(tokens)
        
        mask_token_id = self.tokenizer.token_to_id(MASK_TOKEN)
        
        x_ids = np_ids
        x_ids[masked_ids] = mask_token_id
        x_tokens = np_tokens
        x_tokens[masked_ids] = "[MASKED]"
        
        y_ids = np_ids[masked_ids]
        y_tokens = np_tokens[masked_ids]
        
        return (x_ids, x_tokens), (y_ids, y_tokens)
        
    
    def sample_NSP(self, kernel):
        return self.validate_and_tokenize_kernel(kernel)


from copy import deepcopy
import torch
import numpy as np
import datasets
import tokenizers
from src.t5_small.datasets.config import DEVICE
import random

class LocalDataSampler():
    
    def __init__(self, tokenizer, sampling_type : str = "NSP"):
        self.sampling_type = sampling_type
        self.tokenizer = tokenizer
        
    def __call__(self, kernel):
        if self.sampling_type == "NSP":
            return self.sample_NSP(kernel)
        # elif self.sampling_type == "MLM":
        #     return self.sample_MLM(kernel)
        else:
            raise ValueError("DataSampler.__call__: Unknown sampling type: %s" % self.sampling_type)

    def validate_and_tokenize_kernel(self, kernel : dict) -> str:

        kernel_x = kernel.get("comment", "") + "\n" + kernel.get("header", "")
        kernel_y = kernel.get("body", "")
            
        return kernel_x, kernel_y

    def sample_NSP(self, kernel):
        return self.validate_and_tokenize_kernel(kernel)
        
        
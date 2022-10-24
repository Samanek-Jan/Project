import os, sys, json
import logging
import torch
import random
from transformers import tokenizers
from dataset.data_sampler import DataSampler

from dataset.dataset_errors import WrongParameterError

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, in_folder : str, epoch_len : int, shuffle : bool, max_file_buffer : int = 1000, **data_sampler_kwargs):
        super().__init__()
        
        self.in_folder = in_folder
        self.file_idx = 0
        self.epoch_len = epoch_len
        self.max_file_buffer = max_file_buffer
        self.shuffle = shuffle
        self.data_sampler = DataSampler(**data_sampler_kwargs)
        
        self.data = []
        
        if not os.path.isdir(in_folder):
            raise WrongParameterError("Parameter 'in_folder' must be a directory")
        
        self.__load_data()
        
    def __len__(self):
        return self.epoch_len

    def __load_data(self):
        files = os.listdir(self.in_folder)
        
        for i, file in enumerate(files, self.file_idx):
            
            full_path = os.path.join(self.in_folder, file)
            with open(full_path, "r") as fd:
                self.data.append(json.load(fd))
                
            self.file_idx = i
            if self.file_idx % self.max_file_buffer:
                break
    
        if self.shuffle:
            random.shuffle(self.data)
    
    def __getitem__(self, _):
        ...
        

        
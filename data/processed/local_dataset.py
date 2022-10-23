import os, sys, json
from typing import Dict, List
import torch
import logging
import random
from copy import deepcopy

from data.parser.parser import DATA_FILE_SUFFIX

class LocalDataSampler(torch.utils.data.Dataset):
    
    def __init__(self,
                 max_file_buffer : int, 
                 max_x_size      : int,
                 max_y_size      : int,
                 shuffle         : bool = True, 
                 folder          : str = "./"):
        super(LocalDataSampler, self).__init__()
        self.logger = logging.getLogger('LocalDataSampler')
        self.files = [file for file in os.listdir(folder) if file.endswith(DATA_FILE_SUFFIX)]
        self.parsing_files = deepcopy(self.files)

        self.max_x_size = max_x_size
        self.max_y_size = max_y_size
        self.max_file_buffer = max_file_buffer

        self.folder = folder
        self.shuffle = shuffle
        
        self.samples : List[Dict] = []
        
        self.__get_more_samples()
        self.dataset_len = self.__estimate_dataset_length()
            

    def __getitem__(self, _ : int):
        sample = self.samples.pop(-1)
        if sample.get("type") == "function":
            ...
        
    def __len__(self) -> int:
        self.dataset_len
        
    def __estimate_dataset_length(self) -> int:
        if len(self.files) < self.max_file_buffer:
            return len(self.samples)
        else:
            # Just an estimate
            return round(len(self.samples) * len(self.files) / self.max_file_buffer)
        
    def __get_more_samples(self) -> None:
        if self.shuffle:
            random.shuffle(self.parsing_files)
        
        for _ in range(self.max_file_buffer):
            if len(self.parsing_files) == 0:
                break
            
            file = self.parsing_files.pop(-1)
            
            with open(os.path.join(self.folder, file), 'r') as fd:
                self.samples.extend(json.load(fd))
        
        if self.shuffle:
            random.shuffle(self.samples)
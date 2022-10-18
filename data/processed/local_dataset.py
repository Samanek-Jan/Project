import os, sys, json
from typing import Dict, List
import torch
import logging
import random
from copy import deepcopy

from data.parser.parser import DATA_FILE_SUFFIX

class LocalDataSampler(torch.utils.data.Dataset):
    
    def __init__(self, partial_samples_ratio : float, max_file_buffer : int, shuffle : bool = True, folder : str = "./"):
        super(LocalDataSampler, self).__init__()
        self.logger = logging.getLogger('LocalDataSampler')
        self.files = [file for file in os.listdir(folder) if file.endswith(DATA_FILE_SUFFIX)]
        self.parsing_files = deepcopy(self.files)

        self.partial_samples_ratio = partial_samples_ratio
        self.max_file_buffer = max_file_buffer

        self.folder = folder
        self.shuffle = shuffle
        
        self.samples : List[Dict] = []
        
        self.__get_more_samples()
        self.dataset_len = self.__estimate_dataset_length()
        random.shuffle(self.samples)
            

    def __getitem__(self, _ : int):
        sample = self.samples.pop(-1)
        if sample.get("type") == "struct":
            ...
        elif sample.get("type") == "class":
            ...
        elif sample.get("type") == "function":
            ...
        
    def __len__(self) -> int:
        self.dataset_len
        
    def __estimate_dataset_length(self) -> int:
        if len(self.files) < self.max_file_buffer:
            return len(self.samples)
        else:
            # Just an estimate
            return len(self.samples) * len(self.files) / self.max_file_buffer
        
    def __get_more_samples(self) -> None:
        if self.shuffle:
            random.shuffle(self.parsing_files)
        
        for _ in range(self.max_file_buffer):
            if len(self.parsing_files) == 0:
                return
            
            file = self.parsing_files.pop(-1)
            
            with open(os.path.join(self.folder, file), 'r') as fd:
                self.samples.extend(json.load(fd))
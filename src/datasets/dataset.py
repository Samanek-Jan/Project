import os, sys, json
import logging
from typing import List, Union
import torch
import random
from tqdm import tqdm

from datasets.data_sampler import DataSampler
from datasets.dataset_errors import EmptyDatasetError, WrongParameterError

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 in_folder : str, 
                 epoch_len : int, 
                 shuffle : bool, 
                 samples_per_obj : int,
                 max_file_buffer : int = 1000, 
                 **data_sampler_kwargs) -> None:
        super().__init__()
        
        self.in_folder = in_folder
        self.file_idx = 0
        self.epoch_len = epoch_len
        self.max_file_buffer = max_file_buffer
        self.shuffle = shuffle
        self.sample_buffer = []
        self.samples_per_obj = samples_per_obj
        self.data_sampler = DataSampler(**data_sampler_kwargs)
        
        self.data = []
        
        if not os.path.isdir(in_folder):
            raise WrongParameterError("Parameter 'in_folder' must be a directory")
        
        self.__load_data()
        
    def __len__(self) -> int:
        return self.epoch_len

    def __load_data(self) -> None:
        files = os.listdir(self.in_folder)
        
        if len(files) == 0:
            raise EmptyDatasetError()
        
        next_file_batch_size = min(len(files) - self.file_idx, self.max_file_buffer - self.file_idx)
        pb = tqdm(range(self.file_idx, self.file_idx + next_file_batch_size), leave=False)
        for i in pb:
            
            if i >= len(files):
                break
        
            if i >= len(files):
                self.file_idx = 0
                self.data += self.__load_data()
                break
            
            file = files[i]
            full_path = os.path.join(self.in_folder, file)
            with open(full_path, "r") as fd:
                self.data.append(json.load(fd))
                
            self.file_idx = i
    
        if self.shuffle:
            random.shuffle(self.data)
    
    def __getitem__(self, _) -> List:
        
        if len(self.data) == 0:
            self.__load_data()
                    
        if len(self.sample_buffer) == 0:
            parsed_objs = self.data.pop(0)
            self.sample_buffer = self.data_sampler.sample(parsed_objs, self.samples_per_obj)
        
        sample = self.sample_buffer.pop(0)
        x = sample["x"]
        y = sample["y"]
        is_gpu = sample["is_gpu"]
        return x, y, is_gpu


if __name__ == "__main__":
    data_sampler_kwargs = {"min_x" : 50, "max_x" : 150, "min_y" : 10, "max_y" : 50}
    in_folder_path = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/processed"
    dataset = Dataset(in_folder_path, 10, True, 5,**data_sampler_kwargs)
    
    while True:
        x, y, is_gpu = dataset.__getitem__(0)
        print(f"""x = {x}\ny = {y}\nis_gpu = {is_gpu}\n------------------------------------------\n""")
        
        inp = input()
        if inp.lower() == "exit":
            break 
        
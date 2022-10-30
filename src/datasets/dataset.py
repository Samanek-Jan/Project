import os, sys, json
import logging
from typing import List
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from datasets.config import DEVICE

from datasets.data_sampler import DataSampler
from datasets.dataset_errors import EmptyDatasetError, WrongParameterError


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 in_folder : str,
                 tokenizer_path : str,
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
        self.data_sampler = DataSampler(tokenizer_path, **data_sampler_kwargs)
        
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
        x_str = sample["x_str"] 
        y = sample["y"]
        y_str = sample["y_str"] 
        is_gpu = sample["is_gpu"]
        return (x, x_str), (y, y_str), is_gpu
    
    
    def get_token_id(self, token : str) -> int:
        return self.data_sampler.get_token_id(token)
    
    def get_vocab_size(self) -> int:
        return self.data_sampler.get_vocab_size()
        

class CollateFunctor:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, samples: list):
        (x_ids, x_str), (y_ids, y_str), cuda_map = zip(*samples)
        x_ids, x_mask = self.collate_sentences(x_ids)
        y_ids, y_mask = self.collate_sentences(y_ids)
        return (x_ids, x_mask), (y_ids, y_mask), (x_str, y_str), cuda_map

    def collate_sentences(self, samples: list):
        lengths = [sentence.size(0) for sentence in samples]
        max_length = max(lengths)

        subword_ids = torch.stack([
            F.pad(sentence, (0, max_length - length), value=self.pad_id)
            for length, sentence in zip(lengths, samples)
        ])
        attention_mask = subword_ids == self.pad_id

        return subword_ids.to(DEVICE), attention_mask.to(DEVICE)
    


if __name__ == "__main__":
    from tokenizers import Tokenizer
    data_sampler_kwargs = {"min_x" : 10, "max_x" : 50, "min_y" : 5, "max_y" : 20}
    in_folder_path = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/processed"
    tokenizer_path = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/tokenizer/vocab_10000.json"
    dataset = Dataset(in_folder_path, tokenizer_path, 10, True, 5,**data_sampler_kwargs)
    tokenizer : Tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # for i in tqdm(range(10000)):
    #     x, y, is_gpu = dataset.__getitem__(0)
    
    while True:
        x, y, is_gpu = dataset.__getitem__(0)
        print([tokenizer.id_to_token(id) for id in x])
        print()
        print(tokenizer.decode(x))
        print()
        print([tokenizer.id_to_token(id) for id in y])
        print()
        print(tokenizer.decode(y))
        print()
        print(f"is_gpu = {is_gpu}\n------------------------------------------\n""")

        inp = input()
        if inp.lower() == "exit":
            break 
        
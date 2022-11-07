import os, sys, json
import logging
from typing import List
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from datasets.config import CPP_BOS_TOKEN, CUDA_BOS_TOKEN, DEVICE

from datasets.data_sampler import DataSampler
from datasets.dataset_errors import EmptyDatasetError, WrongParameterError


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 in_folder : str,
                 epoch_len : int, 
                 samples_per_obj : int,
                 shuffle : bool = True, 
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

    def __load_data(self, next_file_batch_size : int = None) -> None:
        files = os.listdir(self.in_folder)
        
        if len(files) == 0:
            raise EmptyDatasetError()
        if next_file_batch_size is None:
            next_file_batch_size = min(self.max_file_buffer, len(files))
            
        pb = tqdm(range(self.file_idx, self.file_idx + next_file_batch_size), leave=False)
        pb.set_description("Caching data")
        for i in pb:
        
            if i >= len(files):
                self.file_idx = 0
                next_file_batch_size = next_file_batch_size - self.file_idx
                self.data += self.__load_data(next_file_batch_size)
                break
            
            file = files[i]
            full_path = os.path.join(self.in_folder, file)
            with open(full_path, "r") as fd:
                self.data.append(json.load(fd))
                
            self.file_idx = i
    
        if self.shuffle:
            random.shuffle(self.data)
    
    def __getitem__(self, _) -> List:
                    
        while len(self.sample_buffer) == 0:
            if len(self.data) == 0:
                self.__load_data()
                
            parsed_objs = self.data.pop(0)
            self.sample_buffer = self.data_sampler.sample(parsed_objs, self.samples_per_obj)
        
        sample = self.sample_buffer.pop(0)
        x = torch.tensor(sample["x"])
        x_str = sample["x_str"] 
        y = torch.tensor(sample["y"])
        y_str = sample["y_str"]
        
        return x, x_str, y, y_str
    
    
    def get_token_id(self, token : str) -> int:
        return self.data_sampler.get_token_id(token)
    
    def get_vocab_size(self) -> int:
        return self.data_sampler.get_vocab_size()
    
    def decode_batch(self, batch, *args, **kwargs):
        return self.data_sampler.decode_batch(batch, *args, **kwargs)
    
    def get_tokenizer(self):
        return self.data_sampler.get_tokenizer()
        

class CollateFunctor:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, samples: list):
        x_ids, x_str, y_ids, y_str = zip(*samples)
        x_ids, x_mask = self.collate_sentences(x_ids)
        y_ids, y_mask = self.collate_sentences(y_ids)
        return (x_ids, x_mask), (y_ids, y_mask), (x_str, y_str)

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
    in_folder_path = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/processed/train"
    tokenizer_path = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/tokenizer/vocab_20000.json"
    data_sampler_kwargs = {"min_x" : 10, "max_x" : 50, "min_y" : 5, "max_y" : 20, "tokenizer_path" : tokenizer_path}
    dataset = Dataset(in_folder_path, 10, True, 5,**data_sampler_kwargs)
    tokenizer : Tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # BOS_CUDA_ID = tokenizer.token_to_id(CUDA_BOS_TOKEN)
    # BOS_CPP_ID = tokenizer.token_to_id(CPP_BOS_TOKEN)
    
    # cpp_sentence_counter = 0
    # cuda_sentence_counter = 0
    # for i in tqdm(range(10000)):
    #     x, x_str, y, y_str = dataset.__getitem__(0)
    #     if y[0] == BOS_CUDA_ID:
    #         cuda_sentence_counter += 1
    #     elif y[0] == BOS_CPP_ID:
    #         cpp_sentence_counter += 1
    #     else:
    #         raise ValueError("Weird start id : {}, token: {}".format(y[0], tokenizer.id_to_token(y[0])))
    
    # print("Number of sentences:")
    # print(" - cuda sentence counter: {}".format(cuda_sentence_counter))
    # print(" - cpp sentence counter: {}".format(cpp_sentence_counter))
    # print(" - cuda ratio: {:.2%}".format(cuda_sentence_counter / (cuda_sentence_counter + cpp_sentence_counter)))
    
    while True:
        x , x_str, y, y_str = dataset.__getitem__(0)
        print([tokenizer.id_to_token(id) for id in x])
        print()
        print(tokenizer.decode(x.tolist()))
        print()
        print([tokenizer.id_to_token(id) for id in y])
        print()
        print(tokenizer.decode(y.tolist()))
        print()
        print("\n------------------------------------------\n")

        inp = input()
        if inp.lower() == "exit":
            break 
        
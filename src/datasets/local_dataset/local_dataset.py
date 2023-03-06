import random
import tokenizers
import torch
from src.datasets.local_dataset.local_data_sampler import LocalDataSampler
from src.datasets.config import SAMPLING_TYPES, mongoDB

from src.datasets.tokenizer import CupydTokenizer

class LocalDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer : CupydTokenizer, max_x : int, max_y : int, part : str, sampling_type = SAMPLING_TYPES["NSP"], buffer_size : int = 1000):
        
        self.datasampler = LocalDataSampler(tokenizer, max_x, max_y, sampling_type)
        self.buffer_size = buffer_size
        if part == "train":
            self.db = mongoDB["cuda_snippets"]["train"]
        else:
            self.db = mongoDB["cuda_snippets"]["validation"]
            
        self.len = self.db.count_documents({})
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        kernel = self.db.find_one({"index" : i})
        if kernel is None:
            raise KeyError("LocalDataset.__iter__: Invalid index")

        return self.datasampler(kernel)
    
    def __next__(self):
        i = random.randint(0, self.__len__())
        return self.__iter__(i)
    

if __name__ == "__main__":
    tokenizer = tokenizers.Tokenizer.from_file("/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/tokenizer/vocab_20000.json")
    dataset = LocalDataset(tokenizer, 10, 10, "train")
    
    while True:
        (x_ids, x_tokens), (y_ids, y_tokens) = next(dataset)
        print(len(x_ids), len(y_ids))
        
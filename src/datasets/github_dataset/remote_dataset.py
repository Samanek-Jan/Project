from random import shuffle
import tokenizers
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from src.model.bart.config import MAX_SEQUENCE_SIZE

from src.datasets.collate_functor import CollateFunctor
from src.datasets.config import SAMPLING_TYPES
from src.datasets.github_dataset.remote_data_sampler import RemoteDataSampler


class RemoteDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer, epoch_len : int, sampling_type = SAMPLING_TYPES["NSP"], buffer_size : int = 5000):
        
        self.datasampler = RemoteDataSampler(tokenizer, sampling_type)
        self.epoch_len = epoch_len
        self.buffer_size = buffer_size
        self.buffer = []
        self.sample_buffer = []
        self.init_ds()
    
    def init_ds(self):
        self.ds = iter(load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["C", "C++"]))
       
    def cache(self):
        size = self.buffer_size - len(self.buffer)
        pbar = tqdm(range(size), leave=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        pbar.set_description_str(f"Caching data")
        try:
            for _ in pbar:
                self.buffer.append(next(self.ds)["code"])
        except Exception as e:
            self.init_ds()
            size = self.buffer_size - len(self.buffer)
            pbar = tqdm(range(size), leave=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            pbar.set_description_str(f"Retrying caching data")
            for _ in pbar:
                self.buffer.append(next(self.ds)["code"])
                
        shuffle(self.buffer)
        
    def __len__(self):
        return self.epoch_len
    
    def __getitem__(self, i):            
        while len(self.sample_buffer) == 0:
            while len(self.buffer) == 0:
                self.cache()
            self.sample_buffer.extend(self.datasampler(self.buffer.pop(0)))
            
        return self.sample_buffer.pop(i % len(self.sample_buffer))
    

if __name__ == "__main__":
    # tokenizer = tokenizers.Tokenizer.from_file("/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/tokenizer/vocab_20000.json")
    configuration = AutoConfig.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    dataset = RemoteDataset(tokenizer, 10, 10, 10)
    
    collate_fn = CollateFunctor(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=2, collate_fn=collate_fn)
    
    for (x, x_str), (y, y_str) in dataloader:
        ...
    # while True:
    #     (x_ids, x_tokens), (y_ids, y_tokens) = next(dataset)
        # print(len(x), len(y))
        
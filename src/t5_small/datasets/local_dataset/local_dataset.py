import random
import transformers
import torch
from src.t5_small.config import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from src.t5_small.datasets.collate_functor import CollateFunctor
from src.t5_small.datasets.local_dataset.local_data_sampler import LocalDataSampler
from src.t5_small.datasets.config import SAMPLING_TYPES, mongoDB

class LocalDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer, part : str, max_epoch_size : int = None, sampling_type = SAMPLING_TYPES["NSP"], buffer_size : int = 5000, shuffle : bool = True):
        
        self.datasampler = LocalDataSampler(tokenizer, sampling_type)
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        if part == "train":
            self.db = mongoDB["cuda_snippets"]["train"]
        else:
            self.db = mongoDB["cuda_snippets"]["validation"]
        
        self.db.create_index("index")
        self.db.create_index("validation.compiled")
            
        self.len = self.db.count_documents({})
        self.max_epoch_size = max_epoch_size if max_epoch_size is not None else self.len
        self.indecies = list(range(self.__len__()))
        self.buffer = []
        if shuffle:
            random.shuffle(self.indecies)
        self.cache()
    
    def cache(self):
        append_size = self.buffer_size - len(self.buffer)
        append_indices = None
        if len(self.indecies) < append_size:
            append_indices = self.indecies
            self.indecies = list(range(self.__len__()))
            if self.shuffle:
                random.shuffle(self.indecies)
            append_indices.extend(self.indecies[:min(self.buffer_size-len(append_indices), len(self.indecies))])
        else:
            append_indices = self.indecies[:min(append_size, len(self.indecies))]
        
        self.buffer.extend(self.db.find({"index" : {"$in" : append_indices}}))
    
    def __len__(self):
        return min(self.len, self.max_epoch_size)
    
    def __getitem__(self, i):
        if len(self.buffer) == 0:
            self.cache()
        kernel = self.buffer[i%len(self.buffer)]
        return self.datasampler(kernel)
    
    def __next__(self):
        i = random.randint(0, self.__len__())
        return self.__getitem__(i)
    

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", use_fast=False, model_max_length=512, add_bos_token=True)
    tokenizer.add_special_tokens({
        # "bos_token" : BOS_TOKEN,
        # "eos_token" : EOS_TOKEN,
        "unk_token" : UNK_TOKEN,
        "pad_token" : PAD_TOKEN,
    })
    dataset = LocalDataset(tokenizer, "train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, collate_fn=CollateFunctor(tokenizer))
    
    for (x_ids, x_tokens), (y_ids, y_tokens) in dataloader:        
        print(len(x_ids), len(y_ids))
        
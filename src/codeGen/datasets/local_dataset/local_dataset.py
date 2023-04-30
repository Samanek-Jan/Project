import random
import transformers
import torch
from codeGen.config import MODEL_NAME
from src.codeGen.datasets.collate_functor import CollateFunctor
from src.codeGen.datasets.local_dataset.local_data_sampler import LocalDataSampler
from src.codeGen.datasets.config import SAMPLING_TYPES, mongoDB

class LocalDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer, part : str):
        
        self.datasampler = LocalDataSampler(tokenizer, part)
        self.cursor = []
        if part == "train":
            self.db = mongoDB["cuda_snippets"]["train"]
        else:
            self.db = mongoDB["cuda_snippets"]["validation"]
        
        self.db.create_index("index")
        self.db.create_index("validation.compiled")
            
        self.match_query = {"metadata.header_cuda_prefixes" : "__global__", "metadata.correct_syntax" : True}
        # self.match_query = {"metadata.correct_syntax" : True}
        # self.match_query = {"metadata.uses_local_mem" : True}
                
        self.len = self.db.count_documents(self.match_query)
        print(f"{part} dataset found {self.len} matching docs")
        # if self.len == 0:
        #     raise Exception("No data found.")
        self._cache()

    def _cache(self):
        self.cursor = self.db.aggregate([
            {"$match" : self.match_query},
            {"$sample" : {"size" : self.len}}
        ], allowDiskUse=True)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        try:
            for doc in self.cursor:
                return self.datasampler(doc)
        except:
            self._cache()
        return self.__getitem__(i)
    
    def __next__(self):
        return self.__getitem__(0)
    

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, model_max_length=1024, add_bos_token=True)
    tokenizer.add_special_tokens({
        "pad_token" : tokenizer.eos_token
    })
    
    dataset = LocalDataset(tokenizer, "train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 2, collate_fn=CollateFunctor(tokenizer))
    
    for (x_ids, x_str), (y_ids, y_str) in dataloader:        
        # print()
        ...
        
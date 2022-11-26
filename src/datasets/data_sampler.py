import os
from typing import Dict, List
import random
import numpy as np
from tokenizers import Tokenizer
import tokenizers

from src.datasets.config import CPP_BOS_TOKEN, CUDA_BOS_TOKEN, EOS_TOKEN, NEW_LINE_TOKEN, RANDOM_SEED
from src.datasets.dataset_errors import WrongParameterError
from src.datasets.tokenizer import SUBWORD_PREFIX

# Init of random module
random.seed(RANDOM_SEED)

def flatten_object_list(parsed_objects : List):
    flatten_list = []
    
    for parsed_object in parsed_objects:
        flatten_list += flatten_object_list(parsed_object["inner_objects"])
        flatten_list.append(parsed_object)
    
    return flatten_list

class DataSampler:
    
    def __init__(self, 
                 tokenizer_path : str, 
                 min_x : int, 
                 max_x : int, 
                 min_y : int, 
                 max_y : int, 
                 **kwargs):
        
        if not os.path.isfile(tokenizer_path):
            raise WrongParameterError(f"tokenizer_path \"{tokenizer_path}\" is not a file")
        elif min_x < 1 or max_x < min_x or min_y < 1 or max_y < min_y:
            raise WrongParameterError("""Size parameters are invalid.
                                      min_x = {}
                                      max_x = {}
                                      min_y = {}
                                      max_y = {}""".format(min_x, max_x, min_y, max_y))
        
        self.tokenizer : Tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.model.dropout = 0 if "bpe_dropout" not in kwargs else kwargs["bpe_dropout"]
        self.new_line_id = self.tokenizer.token_to_id("\n") 
        
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
        # Check if first data from given object 
        self.basic_first_samples = kwargs.get("init_first_sample", True)
        
    def __shuffle_objects(self, parsed_objects : List[Dict[str, str]]) -> Dict[str, str]:
        random.shuffle(parsed_objects)
        return parsed_objects

    def __clear_line(self, line : str, seq_check_len : int = 3) -> str:
        if line.endswith("\n"):
            line = line[:-1]
        
        line = line.strip(" -")
        
        if len(line) < seq_check_len:
            return line

        if line[-seq_check_len:] == line[-1] * seq_check_len:
            line = line.strip(line[-1])
        
        return line       
                    
    def __clear_comment(self, comment : str) -> str:
        clear_comment = []
        comment = comment.replace("//", "")
        comment = comment.replace(" * ", "")
        comment_lines = comment.split("\n")
        for line in comment_lines:
            if line.lstrip().startswith("/") or line.rstrip().endswith("/"):
                line = line.strip("/")
            
            if line.lstrip().startswith("*") or line.rstrip().endswith("*"):
                line = line.strip("*")
            
            if line.lstrip().startswith("\\") or line.rstrip().endswith("\\"):
                line = line.strip("\\")
                
            if line != "":
                clear_comment.append(self.__clear_line(line))

        clear_comment = " ".join(clear_comment)
        return clear_comment
    
    def __get_basic_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        
        x = self.tokenizer.encode(self.__clear_comment(obj.get("comment", "")))
        y = self.tokenizer.encode(obj.get("header", "") + obj.get("body", ""))
        
        assert len(x.ids) == len(x.tokens)
        assert len(y.ids) == len(y.tokens)  
        
        if len(x.ids) + len(y.ids) < self.min_x + self.min_y:
            return None
        
        elif len(x.ids) < self.min_x or len(y.ids) < self.min_y:
            # return self.__get_random_sample(obj)   
            return None
        
        # 50 % chance of using code section in X input if snippet is bigger than max size
        elif len(x.ids) + len(y.ids) > self.max_x + self.max_y and np.random.random() < 0.5:
            if len(x.ids) > self.max_x:
                x_size = self.max_x
                y_size = min(self.max_y - (len(x.ids) - x_size) - 2, len(y.ids))
                if y_size < self.min_y:
                    return None
                
                x_ids = x.ids[:x_size]
                x_str = self.tokenizer.decode(x_ids, skip_special_tokens=True)
                y_ids = x.ids[x_size:] + y.ids[:y_size]
                y_str = self.tokenizer.decode(y_ids, skip_special_tokens=False)
                
            else: # len(y.ids) > self.max_y
                x_size = len(x.ids)
                y_size = min(len(y.ids) - (self.max_x - x_size), self.max_y-2)
                
                x_ids = x.ids + y.ids[:self.max_x-x_size]
                y_ids = y.ids[self.max_x-x_size:self.max_x-x_size+y_size]

        else:
            x_size = min(self.max_x, len(x.ids))
            y_size = min(self.max_y-2, len(y.ids))
            
            x_ids = x.ids[len(x.ids) - x_size:]
            y_ids = y.ids[:y_size]
        
        x_str = self.tokenizer.decode(x_ids, skip_special_tokens=False)
        y_str = self.tokenizer.decode(y_ids, skip_special_tokens=False)
        BOS_ID = self.tokenizer.token_to_id(CUDA_BOS_TOKEN if obj.get("is_gpu", False) else CPP_BOS_TOKEN)
        EOS_ID = self.tokenizer.token_to_id(EOS_TOKEN)
        y_ids.insert(0, BOS_ID)
        y_ids.append(EOS_ID)
        
        return {
                "x" : x_ids, 
                "x_str" : x_str,
                "y" : y_ids, 
                "y_str" : y_str
               }
    
    def __get_random_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        encoding = self.tokenizer.encode(obj.get("comment", "") + obj.get("header", "") + obj.get("body", ""))
        ids_len = len(encoding.ids)
        if ids_len < self.min_x + self.min_y:
            return None
        
        assert len(encoding.ids) == len(encoding.tokens)

        x_size = random.randint(self.min_x, min(self.max_x, ids_len - self.min_y))
        y_size = random.randint(self.min_y, min(self.max_y-2, ids_len - x_size))
        pivot = random.randint(x_size, ids_len - y_size)
        
        x, x_str = self.__align_x(encoding, x_size, pivot)
        y, y_str = self.__align_y(encoding, y_size, obj.get("is_gpu", False), pivot)
        
        if x is None or y is None:
            return None
        
        return {
                "x" : x, 
                "x_str" : x_str,
                "y" : y, 
                "y_str" : y_str,
                "is_gpu" : obj.get("is_gpu", False)
               }
            
    def sample(self, 
               parsed_objects : List[Dict[str, str]], 
               samples_per_obj : int = 1, 
               max_tries : int = None) -> List[Dict]:

        samples = []
        parsed_objects = flatten_object_list(parsed_objects)
        
        for obj in parsed_objects:
            sample = self.__get_basic_sample(obj)
            if sample is not None:
                samples.append(sample)
                
        return samples
        
        # if samples_per_obj < 1:
        #     return samples
        
        # sample_n = samples_per_obj * len(parsed_objects)
        # tries_left = sample_n * 10 if max_tries is None else max_tries
        
        # if self.basic_first_samples:
            
        #     parsed_objects = self.__shuffle_objects(parsed_objects)
            
        #     for parsed_obj in parsed_objects:
        #         sample = self.__get_basic_sample(parsed_obj)
        #         if sample is not None:
        #             samples.append(sample)          
        #             sample_n -= 1
                
        #         tries_left -= 1
        #         if tries_left < 1:
        #             return samples
        
        
        # parsed_objects = self.__shuffle_objects(parsed_objects)
            
        # while sample_n > 0:        
        #     parsed_obj = self.__get_random_object(parsed_objects)
        #     sample = self.__get_random_sample(parsed_obj)
        #     if sample is not None:
        #         samples.append(sample)
        #         sample_n -= 1
            
        #     tries_left -= 1
        #     if tries_left < 1:
        #         break

        
        # return samples
    
    def get_token_id(self, token : str) -> int:
        return self.tokenizer.token_to_id(token)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def decode_batch(self, batch, *args, **kwargs):
        return self.tokenizer.decode_batch(batch, *args, **kwargs)
    
    def get_tokenizer(self):
        return self.tokenizer
import os
from typing import Dict, List
import random
import numpy as np
from tokenizers import Tokenizer
import tokenizers

from src.datasets.config import BOS_TOKEN, EOS_TOKEN, RANDOM_SEED
from src.datasets.tokenizer import SUBWORD_PREFIX

# Init of random module


def flatten_object_list(parsed_objects : List):
    flatten_list = []
    
    for parsed_object in parsed_objects:
        flatten_list += flatten_object_list(parsed_object["inner_objects"])
        flatten_list.append(parsed_object)
    
    return flatten_list

class DataSamplerInterface:
    
    def __init__(self, 
                 tokenizer_path : str, 
                 max_x : int, 
                 max_y : int, 
                 **kwargs):
        
        if not os.path.isfile(tokenizer_path):
            raise Exception(f"tokenizer_path \"{tokenizer_path}\" is not a file")
        elif max_x < 1 or max_y < 1:
            raise Exception("""Size parameters are invalid.
                                      max_x = {}
                                      max_y = {}""".format(max_x, max_y))
        
        self.tokenizer : Tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.model.dropout = 0 if "bpe_dropout" not in kwargs else kwargs["bpe_dropout"]
        
        self.max_x = max_x
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
        while comment:
            if comment[0].isalpha():
                break
            comment = comment[1:]
            
        while comment:
            if comment[-1].isalpha():
                break
            comment = comment[:-1]
        
        comment = comment.replace("/", "").replace("**", "").replace("--", "").replace("\\", "")
        return comment
    
    def __get_basic_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        
        clear_comment = self.__clear_comment(obj.get("comment", ""))
        x = self.tokenizer.encode(clear_comment)
        y = self.tokenizer.encode(obj.get("header", "") + obj.get("body", ""))
        
        assert len(x.ids) == len(x.tokens)
        assert len(y.ids) == len(y.tokens)  
        
        # Too short snippet
        if len(x.ids) + len(y.ids) < self.min_x + self.min_y:
            return None
        
        # Snippet is long enough but insufficient comment or body
        elif len(x.ids) < self.min_x or len(y.ids) < self.min_y:
            return None
            # tokens = self.tokenizer.encode(obj.get("comment", "") + obj.get("header", "") + obj.get("body", ""))
            # if len(tokens.ids) <= self.max_x + self.max_y - 2:
            #     x_size = random.randint(self.min_x, min(len(tokens.ids) - self.min_y, self.max_x))
            #     x_ids = tokens.ids[:x_size]
            #     y_ids = tokens.ids[x_size:]
            # else:
            #     x_size = self.max_x
            #     y_size = self.max_y-2
            #     x_ids = tokens.ids[:x_size]
            #     y_ids = tokens.ids[x_size:x_size + y_size]

        
        # 50 % chance of using code section in X input if snippet is bigger than max size
        elif len(x.ids) + len(y.ids) > self.max_x + self.max_y and np.random.random() < 0.5:
            if len(x.ids) > self.max_x:
                x_size = self.max_x
                y_size = min(self.max_y - (len(x.ids) - x_size) - 2, len(y.ids))
                if y_size < self.min_y:
                    return None
                
                x_ids = x.ids[:x_size]
                y_ids = x.ids[x_size:] + y.ids[:y_size]
                
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
        BOS_ID = self.tokenizer.token_to_id(BOS_TOKEN if obj.get("is_gpu", False) else CPP_BOS_TOKEN)
        EOS_ID = self.tokenizer.token_to_id(EOS_TOKEN)
        y_ids.insert(0, BOS_ID)
        y_ids.append(EOS_ID)
        
        return {
                "x" : x_ids, 
                "x_str" : x_str,
                "y" : y_ids, 
                "y_str" : y_str
               }
            
    def sample(self, 
               parsed_objects : List[Dict[str, str]], 
               *args, **kwargs) -> List[Dict]:

        samples = []
        parsed_objects = flatten_object_list(parsed_objects)
        
        for obj in parsed_objects:
            sample = self.__get_basic_sample(obj)
            if sample is not None:
                samples.append(sample)
                
        return samples
    
    def get_token_id(self, token : str) -> int:
        return self.tokenizer.token_to_id(token)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def decode_batch(self, batch, *args, **kwargs):
        return self.tokenizer.decode_batch(batch, *args, **kwargs)
    
    def get_tokenizer(self):
        return self.tokenizer
import os
from typing import Dict, List
import random
from tokenizers import Tokenizer
from datasets.config import CPP_BOS_TOKEN, CUDA_BOS_TOKEN, EOS_TOKEN, RANDOM_SEED

from datasets.dataset_errors import TokenizerError, WrongParameterError
from datasets.tokenizer import SUBWORD_PREFIX

# Init of random module
random.seed(RANDOM_SEED)

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
    
    def __get_random_object(self, parsed_objects : List[Dict[str, str]]) -> Dict[str, str]:
        return random.choice(parsed_objects)
    
    def __align_x(self, tokens : List[str], x_size : int, pivot : int = None):
        content_size = len(tokens)
        
        if pivot is None:
            end_token = tokens[content_size-x_size]
            while x_size < content_size and end_token.startswith(SUBWORD_PREFIX):
                x_size += 1
                end_token = tokens[content_size-x_size]
            
            tokens = tokens[content_size-x_size:]    

        else:
            end_token = tokens[pivot - x_size]
            while x_size < content_size - pivot and x_size < pivot and end_token.startswith(SUBWORD_PREFIX):
                x_size += 1
                end_token = tokens[content_size - x_size]
                
            tokens = tokens[pivot - x_size : pivot]
            
        x_ids = self.tokenizer.encode(tokens, is_pretokenized=True).ids
        x_str = self.tokenizer.decode(x_ids)
        return x_ids, x_str

    def __align_y(self, tokens : List[str], y_size : int, is_cuda_snippet : bool, pivot : int = 0):
        content_size = len(tokens)
        
        end_token = tokens[pivot + y_size - 1]
        while y_size < content_size - pivot and y_size < self.max_y and end_token.startswith(SUBWORD_PREFIX):
            y_size += 1
            end_token = tokens[pivot + y_size - 1]
            
        tokens = tokens[pivot : pivot + y_size]
        y_ids = self.tokenizer.encode(tokens, is_pretokenized=True).ids
        y_ids.insert(0, self.tokenizer.token_to_id(CUDA_BOS_TOKEN if is_cuda_snippet else CPP_BOS_TOKEN))
        y_ids.append(self.tokenizer.token_to_id(EOS_TOKEN))
        y_str = self.tokenizer.decode(y_ids, skip_special_tokens=True)
        
        return y_ids, y_str
        
    
    def __get_basic_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        
        x = self.tokenizer.encode(obj.get("comment", "") + obj.get("header", "")).tokens
        y = self.tokenizer.encode(obj.get("body", "")).tokens
        
        if len(x) + len(y) < self.min_x + self.min_y:
            return None
        
        elif len(x) < self.min_x or len(y) < self.min_y:
            return self.__get_random_sample(obj)          
            
        else:
            x_size = min(self.max_x, len(x))
            y_size = min(self.max_y, len(y))
                    
            x, x_str = self.__align_x(x, x_size)
            y, y_str = self.__align_y(y, y_size, obj.get("is_gpu", False))
        
        return {
                "x" : x, 
                "x_str" : x_str,
                "y" : y, 
                "y_str" : y_str
               }
    
    def __get_random_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        tokens = self.tokenizer.encode(obj.get("comment", "") + obj.get("header", "") + obj.get("body", "")).tokens
        if len(tokens) < self.min_x + self.min_y:
            return None

        x_size = random.randint(self.min_x, min(self.max_x, len(tokens) - self.min_y))
        y_size = random.randint(self.min_y, min(self.max_y, len(tokens) - x_size))
        pivot = random.randint(x_size, len(tokens) - y_size)
        
        x, x_str = self.__align_x(tokens, x_size, pivot)
        y, y_str = self.__align_y(tokens, y_size, obj.get("is_gpu", False), pivot)
        
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
               max_tries : int = None)              -> List[Dict]:

        samples = []
        
        if samples_per_obj < 1:
            return samples
        
        sample_n = samples_per_obj * len(parsed_objects)
        tries_left = sample_n * 10 if max_tries is None else max_tries
        
        if self.basic_first_samples:
            
            parsed_objects = self.__shuffle_objects(parsed_objects)
            
            for i in range(len(parsed_objects)):
                sample = self.__get_basic_sample(parsed_objects[i])
                if sample is not None:
                    samples.append(sample)          
                    sample_n -= 1
                
                tries_left -= 1
                if tries_left < 1:
                    return samples
        
        
        parsed_objects = self.__shuffle_objects(parsed_objects)
            
        while sample_n > 0:        
            parsed_obj = self.__get_random_object(parsed_objects)
            sample = self.__get_random_sample(parsed_obj)
            if sample is not None:
                samples.append(sample)
                sample_n -= 1
            
            tries_left -= 1
            if tries_left < 1:
                break

        
        return samples
    
    def get_token_id(self, token : str) -> int:
        return self.tokenizer.token_to_id(token)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def decode_batch(self, batch, *args, **kwargs):
        return self.tokenizer.decode_batch(batch, *args, **kwargs)
    
    def get_tokenizer(self):
        return self.tokenizer
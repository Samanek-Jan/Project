from typing import Dict, List
import random

from dataset.dataset_errors import WrongParameterError


class DataSampler:
    
    def __init__(self, min_x : int, max_x : int, min_y : int, max_y : int, **kwargs):
        
        if min_x < 1 or max_x < min_x or min_y < 1 or max_y < min_y:
            raise WrongParameterError("""Size parameters are invalid.
                                      min_x = {}
                                      max_x = {}
                                      min_y = {}
                                      max_y = {}""".format(min_x, max_x, min_y, max_y))
            
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
        # Check if first data from given object 
        self.basic_first_samples = kwargs["init_first_sample"] if "init_first_sample" in kwargs else True
    
    def __shuffle_objects(self, parsed_objects : List[Dict[str, str]]) -> Dict[str, str]:
        random.shuffle(parsed_objects)
        return parsed_objects
    
    def __get_random_object(self, parsed_objects : List[Dict[str, str]]) -> Dict[str, str]:
        return random.choice(parsed_objects)
        
    def __get_basic_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        
        content = obj.get("comment", "") + obj.get("header", "") + obj.get("body", "")
        if content < self.min_x + self.min_y:
            return None
        
        x = obj.get("comment", "") + obj.get("header", "")
        y = obj.get("body", "")

        if len(x) < self.min_x or len(y) < self.min_y:
            return self.__get_random_sample(obj)            
            
        else:
            
            x_size = random.randint(self.min_x, min(self.max_x, len(x)))
            y_size = random.randint(self.min_y, min(self.max_y, len(y)))
                    
            x = x[len(x) - x_size:]
            y = y[:y_size]
        
        return {"x" : x, "y" : y, "is_gpu" : obj.get("is_gpu", False)}
    
    def __get_random_sample(self, obj : Dict[str, str]) -> Dict[str, str]:
        content = obj.get("comment", "") + obj.get("header", "") + obj.get("body", "")
        if content < self.min_x + self.min_y:
            return None

        x_size = random.randint(self.min_x, min(self.max_x, len(content) - self.min_y))
        y_size = random.randint(self.min_y, min(self.max_y, len(content) - x_size))
        pivot = random.randint(x_size, len(content - y_size))
        
        x = content[pivot-x_size : pivot]
        y = content[pivot        : pivot+y_size]
        
        return {"x" : x, "y" : y, "is_gpu" : obj.get("is_gpu", False)}
                
    def sample(self, parsed_objects : List[Dict[str, str]], sample_n : int = 1, max_tries : int = None) -> List[List[str, str, str]]:

        samples = []
        
        if sample_n < 1:
            return samples

        if max_tries is None:
            tries_left = sample_n * 10
        else:
            tries_left = max_tries
        
        if self.basic_first_samples:
            
            parsed_objects = self.__shuffle_objects(parsed_objects)
            if len(parsed_objects) > sample_n:
                parsed_objects = parsed_objects[:sample_n]
            
            for i in range(min(len(parsed_objects), sample_n)):
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
                return samples

        
        return samples
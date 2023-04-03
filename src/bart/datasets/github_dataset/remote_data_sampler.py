import re
from typing import List
import torch
import numpy as np
import datasets
import tokenizers
from bart.config import BOS_TOKEN, EOS_TOKEN
from src.bart.datasets.config import DEVICE
import random
from copy import deepcopy

class RemoteDataSampler():
    
    def __init__(self, tokenizer, sampling_type : str = "NSP"):
        self.sampling_type = sampling_type
        self.tokenizer = tokenizer
        
    def __call__(self, kernel):
        if self.sampling_type == "NSP":
            return self.sample_NSP(kernel)
        # elif self.sampling_type == "MLM":
        #     return self.sample_MLM(kernel)
        else:
            raise ValueError("Unknown sampling type: %s" % self.sampling_type)
            
    def sample_NSP(self, code):
        if type(code) != str:
            code = code.get("code", None)
        return self.find_commented_blocks(code)
        
        # pivot = random.randint(len(code)//10, len(code)//10*9)
        
        # x = code[:pivot]
        # y = code[pivot:]
        # y = BOS_TOKEN + y + EOS_TOKEN
        
        # return x, y
    
    def find_commented_blocks(self, code : str):
        
        lines = code.splitlines(keepends=True)
        found_block_start = False
        found_comment = False
        
        block_comment_start_id = 0
        content_id = 0

        comment_lines = []
        code_lines = []
        commented_blocks = []
        
        block_bracket_count = 0
        
        for line in lines:
            if not found_comment and not found_block_start and (res := line.find("/*")) != -1:
                block_comment_start_id = content_id + res
                found_block_start = True
            
            if not found_comment and found_block_start and (res := line.find("*/")) != -1:
                block_comment_end_id = content_id + res
                comment_lines.extend(self.__transform_comment(code[block_comment_start_id:block_comment_end_id].splitlines()))
                found_block_start = False
                found_comment = True
            
            elif not found_block_start and (res := line.find("//")) != -1:
                comment_lines.append(line.rstrip())
                found_comment = True
            
            elif found_comment:
                if line.strip() != "":
                    delta = self.__count_brackets(line)
                    block_bracket_count += delta
                    if block_bracket_count > 0 or \
                       (block_bracket_count == 0 and delta == 0):
                        code_lines.append(line)
                        
                    elif block_bracket_count <= 0 and delta != 0:
                        trimmed_line = line
                        for _ in range(abs(block_bracket_count)):
                            last_idx = line.rfind("}")
                            trimmed_line = trimmed_line[:last_idx]

                        last_idx = trimmed_line.rfind("}")
                        code_lines.append(trimmed_line[:last_idx+1])    
                        commented_blocks.append(
                            ["".join(comment_lines).replace("\r\n", "\n").rstrip() + "\n", BOS_TOKEN + "".join(code_lines).replace("\r\n", "\n") + EOS_TOKEN]
                        )
                    
                        comment_lines.clear()
                        code_lines.clear()
                        found_comment = False
                        block_bracket_count = 0
                        
                elif block_bracket_count == 0:
                    if len(code_lines) > 0:
                        commented_blocks.append(
                            ["".join(comment_lines).replace("\r\n", "\n"), BOS_TOKEN + "".join(code_lines).replace("\r\n", "\n") + EOS_TOKEN]
                        )
                    
                    comment_lines.clear()
                    code_lines.clear()
                    found_comment = False
                    block_bracket_count = 0

            content_id += len(line)
            
        return commented_blocks
            
    def __transform_comment(self, comment : List[str]) -> List[str]:

        transformer_comment = []
        for line in comment:
            for i, c in enumerate(line):
                if c.isalnum():
                    transformer_comment.append("// " + line[i:].strip() + "\n")
                    break
        return transformer_comment
    
    def __count_brackets(self, line : str) -> int:
        m = {"{" : 1, "}" : -1}
        return sum(m.get(c, 0) for c in line)
                
            


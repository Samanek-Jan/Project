import os, sys
import zipfile
import json
import re
import libarchive
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pymongo import MongoClient

MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
db = MongoClient(MONGODB_CONNECTION_STRING)["cuda_snippets"]["stack_overflow_posts"]
db.drop()

buffer_size = 5000
buffer = []

cuda_reg = re.compile("(\W+|\s+|^)cuda(\W+|\s+|$)", re.IGNORECASE)
cleaner_reg = re.compile('<.*?>') 

class PbWrapper:
    
    def __init__(self, pb, len=57721551):
        self.pb = pb
        self.len = len
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return iter(self.pb)
    
    def __next__(self):
        return self.__iter__()

def clean_doc(doc : dict) -> dict:
    new_doc = {}
    for key, value in doc.items():
        key = key[0].lower() + key[1:]
        new_doc[key] = value
        
    new_doc["title"] = re.sub(cleaner_reg, '', new_doc["title"])
    new_doc["body"] = re.sub(cleaner_reg, '', new_doc["body"])
    return new_doc

def process_line(line : str) -> list:
    return ET.fromstring(line)
    
if __name__ == "__main__":
    tagset = {}
    i = -1
    try:
        with open("/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/stack-overflow/Posts.xml", "r") as fd:
            fd = PbWrapper(fd)
            for (line) in tqdm(fd):
                i += 1
                if i < 2:
                    continue
                
                if len(buffer) > buffer_size:
                    db.insert_many(buffer)
                    buffer.clear()
                
                elem = process_line(line)
                tag = elem.tag
                doc = elem.attrib
                tags = doc.get("Tags", None)
                if tags is not None:
                    res = cuda_reg.match(tags)
                    if res is not None:
                        buffer.append(clean_doc(doc))
                        continue
                title = doc.get("Title", None)
                if title is not None:
                    res = cuda_reg.match(title)
                    if res is not None:
                        buffer.append(clean_doc(doc))
                        continue
                
    except Exception as e:
        print(f"Ended on {i+1}. line with error:")   
        print(e)
                
    finally:
        if len(buffer) > 0:
            db.insert_many(buffer)
        

from pymongo import MongoClient
import random
import time
from random import shuffle
from tqdm import tqdm

random.seed(time.time())

MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
db = MongoClient(MONGODB_CONNECTION_STRING)["cuda_snippets"]

train = db["train"]
valid = db["validation"]

train_n = train.count_documents({})
train_faulty_n = train.count_documents({"validation.compiled" : False})
train_faulty_syntax_n = 0
train_faulty_inludes_n = 0

for doc in tqdm(iter(train.find({"validation.compiled" : False})), leave=False):
    last_iter = doc["validation"]["iterations"][-1]
    if len(last_iter["error_analyses"]["include_errors"]) > 0:
        train_faulty_inludes_n += 1
    
    if len(last_iter["error_analyses"]["syntax_errors"]) > 0:
        train_faulty_syntax_n += 1
        
print(f"total trainining docs: {train_n}")
print(f"total not-compiled docs: {train_faulty_n}")
print(f"total syntax errors: {train_faulty_syntax_n}")
print(f"total include errors: {train_faulty_inludes_n}\n")

# validation part

valid_n = valid.count_documents({})
valid_faulty_n = valid.count_documents({"validation.compiled" : False})
valid_faulty_syntax_n = 0
valid_faulty_inludes_n = 0

for doc in tqdm(iter(valid.find({"validation.compiled" : False})), leave=False):
    last_iter = doc["validation"]["iterations"][-1]
    if len(last_iter["error_analyses"]["include_errors"]) > 0:
        valid_faulty_inludes_n += 1
    
    if len(last_iter["error_analyses"]["syntax_errors"]) > 0:
        valid_faulty_syntax_n += 1
        
print(f"total validation docs: {valid_n}")
print(f"total not-compiled docs: {valid_faulty_n}")
print(f"total syntax errors: {valid_faulty_syntax_n}")
print(f"total include errors: {valid_faulty_inludes_n}")


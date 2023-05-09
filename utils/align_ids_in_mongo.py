from pymongo import MongoClient
import random
import time
from random import shuffle
from tqdm import tqdm

random.seed(time.time())

MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
db = MongoClient(MONGODB_CONNECTION_STRING)["cuda_snippets"]


print("training part")
train = db["train"]
docs = sorted(list(train.find({})), key=lambda x: x["index"])
for i, doc in tqdm(enumerate(docs)):
    if doc["index"] == i:
        continue
    db.update_one({"_id": doc["_id"]}, {"$set" : {"index" : i}})

print("validation part")
validation = db["validation"]
docs = sorted(list(validation.find({})), key=lambda x: x["index"])
for i, doc in tqdm(enumerate(docs)):
    if doc["index"] == i:
        continue
    db.update_one({"_id": doc["_id"]}, {"$set" : {"index" : i}})
    
print("\nCheck")
print("training part")
docs = sorted(list(train.find({})), key=lambda x: x["index"])
for i, doc in enumerate(docs):
    if doc["index"] != i:
        raise ValueError("Wrong alignment. doc index: {}, wanted: {}".format(doc["index"], i))
    
print("validation part")
docs = sorted(list(validation.find({})), key=lambda x: x["index"])
for i, doc in enumerate(docs):
    if doc["index"] != i:
        raise ValueError("Wrong alignment. doc index: {}, wanted: {}".format(doc["index"], i))
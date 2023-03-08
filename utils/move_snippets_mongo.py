from pymongo import MongoClient
import random
import time
from random import shuffle
from tqdm import tqdm

random.seed(time.time())

MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
db = MongoClient(MONGODB_CONNECTION_STRING)["cuda_snippets"]

db_from = db["validation"]
db_to = db["train"]
sample_random = True
move_ratio = 0.1

# ----------------------------------------------------------------

db_from_count = db_from.count_documents({})
db_to_count = db_to.count_documents({})

print("db_from_count", db_from_count)
print("db_to_count", db_to_count)

indices = list(range(db_from_count))
if sample_random:
    shuffle(indices)

indices = indices[:round(db_from_count*move_ratio)]
moving_part = list(db_from.find({"index" : {"$in" : indices}}))

print(f"Moving {len(moving_part)} docs")
pbar = tqdm(moving_part)
moved_ids = []
moved_indices = []
for i, (doc) in enumerate(pbar):
    pbar.set_postfix_str(str(doc["_id"]))
    moved_ids.append(doc["_id"])
    moved_indices.append(doc["index"])
    del doc["_id"]
    doc["index"] = db_to_count
    moving_part[i] = doc
    db_to_count += 1

db_to.insert_many(moving_part)
db_from.delete_many({"_id" : {"$in" : moved_ids}})

print("Aligning indices in from collection")
docs = sorted(list(db_from.find({})), key=lambda x: x["index"])
for i, doc in tqdm(enumerate(docs)):
    if doc["index"] == i:
        continue
    db_from.update_one({"_id": doc["_id"]}, {"$set" : {"index" : i}})
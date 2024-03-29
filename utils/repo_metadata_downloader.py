import sys
from pymongo import MongoClient
from tqdm import tqdm
from http import client
import requests
import time

db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
repo_metadata_db = db["repo_metadata"]
file_metadata_db = db["file_metadata"]

repo_metadata_db.create_index("full_name")
repo_metadata_db.create_index("name")
repo_metadata_db.create_index("id")

REPO_DATA_FILE = "addon.csv"

STATUS_ENUM = {
    "PENDING" : "PENDING",
    "READY" : "READY",
}

TIME_SLEEP = 60

def get_repos():
    with open(REPO_DATA_FILE, "r") as fd:
        lines = fd.read().splitlines()
    
    return [line.split(",") for line in lines]
    
if __name__ == "__main__":
    lines = get_repos()
    pb = tqdm(lines, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, (full_name, url) in enumerate(pb):
        pb.set_postfix_str(full_name)
        doc = repo_metadata_db.find_one({"full_name" : full_name, "status" : STATUS_ENUM["READY"]})
        if doc is not None:
            continue
        
        doc = repo_metadata_db.insert_one({
            "full_name" : full_name,
            "url" : url,
            "status" : STATUS_ENUM["PENDING"]
        })

        try:
            res : requests.Response = requests.get(url, timeout=10)
        
            while res.status_code == 403:
                sleeping_time = TIME_SLEEP
                while sleeping_time > 0:
                    pb.set_postfix_str(f"{full_name} - SLEEPING... ({sleeping_time})")
                    time.sleep(1)
                    sleeping_time -= 1
                res : requests.Response = requests.get(url, timeout=5)        
                continue
            
            if (res.status_code != 200):
                continue
        
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            continue
        
        body = res.json()
        
        body["status"] = STATUS_ENUM["READY"]
        repo_metadata_db.update_one({"full_name" : full_name}, {"$set" : {**body}})
    
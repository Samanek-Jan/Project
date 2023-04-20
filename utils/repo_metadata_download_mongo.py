import sys
from pymongo import MongoClient
from tqdm import tqdm
from http import client
import requests
import time

TIME_SLEEP = 600
STATUS_ENUM = {
    "PENDING" : "PENDING",
    "READY" : "READY",
}

db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
repo_metadata_db = db["repo_metadata"]

not_downloaded_repos = list(repo_metadata_db.aggregate([
     {"$match" : {"status" : "PENDING"}},
     {"$project" : {"full_name" : "$full_name", "url" : "$url"}}
]))

pb = tqdm(not_downloaded_repos, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
for repo in pb:
    url = repo.get("url")
    full_name = repo.get("full_name")
    pb.set_postfix_str(f"{full_name}")
    
    res : requests.Response = requests.get(url, timeout=10)
    
    try:
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
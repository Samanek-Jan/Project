import os, sys
from tqdm import tqdm
import json
import pandas as pd
import requests
import zipfile
import io
import time
from typing import List

input_file = "../data/raw/bigQueryRepoLog_filtered.csv"

df = pd.read_csv(input_file)

ARCHIVE_SUFFIX = "/archive/refs/heads/master.zip"
output_folder = "../data/raw/"

with open("skipped_repos.json", "w") as fd:
    pb = tqdm(df.iterrows())
    for i, (_, _, name, url) in pb:
        pb.set_description(f"[{i+1}/{len(df)}]")
        try:
            while ((response := { **requests.get(url).json() }).get("message", None)) and response.get("documentation_url", None):
                time.sleep(1)
            
            topics : List[str] = response.get("topics", None)
            if topics is None:
                fd.write(f"{name},\t\tNo topics found\n")
                continue
            
            elif all([topic.find("cuda") == -1 for topic in topics]):

                fd.write(f"{name},\t\tNo cuda in topics\n")                
                continue
            
            elif ((repo_url := response.get("html_url", None)) == None):
                fd.write(f"{name},\t\tNo repo url found\n")
                continue
            else:
                archive_url = repo_url + ARCHIVE_SUFFIX
                r = requests.get(archive_url, stream=True)
                if not r.ok:
                    fd.write(f"{name},\t\tWas not able to download archive\n")
                    continue
                
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(output_folder)
                archive_name = z.filename
                z.close()
                os.remove(os.path.join(output_folder, archive_name))

        except Exception as e:
            fd.write(f"{name},\t\t{str(e)}\n")
    

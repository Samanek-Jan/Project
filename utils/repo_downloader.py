import os, sys
from tqdm import tqdm
import json
import pandas as pd
import requests
import zipfile
import io
import time
from typing import List

input_file = "bigQueryRepoLog_filtered.csv"
meta_data_file = "repo_metadata.json"


df = pd.read_csv(input_file)

ARCHIVE_PREFIX = "https://github.com/"
ARCHIVE_SUFFIX = "/archive/refs/heads/master.zip"
output_folder = "../data/raw"
meta_data = {}

with open("skipped_repos.json", "w") as fd:
    with open(meta_data_file, "w") as md:
        pb = tqdm(df.iterrows(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        skipped_i = 0
        for i, (name, url) in pb:
            pb.set_description(f"[{i+1}/{len(df)}]")
            pb.set_postfix_str(name)
            try:
                repo_archive_url = f"{ARCHIVE_PREFIX}{name}{ARCHIVE_SUFFIX}"
                r = requests.get(repo_archive_url, stream=True)
                if not r.ok:
                    fd.write(f"{name},\t\tWas not able to download archive: {str(r.content)}\n")
                    skipped_i += 1
                    meta_data[name] = {
                        "index" : i,
                        "status" : f"Failed: {str(r.content)}"
                    }
                    continue
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(output_folder)
                archive_name = z.filename
                fileList = z.filelist
                z.close()
                if archive_name is not None:
                    os.remove(os.path.join(output_folder, archive_name))
                    
                meta_data[name] = {
                    "index" : i,
                    "download_index" : i - skipped_i,
                    "metadata" : url,
                    "archive_name" : archive_name,
                    "files" : [file.filename for file in fileList]
                }              

            except Exception as e:
                fd.write(f"{name},\t\t{str(e)}\n")
        json.dump(meta_data, md)
        

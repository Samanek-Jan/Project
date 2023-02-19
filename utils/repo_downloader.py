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

df = pd.read_csv(input_file)

ARCHIVE_PREFIX = "https://github.com/"
ARCHIVE_SUFFIX = "/archive/refs/heads/master.zip"
output_folder = "../data/raw/"

with open("skipped_repos.json", "w") as fd:
    pb = tqdm(df.iterrows()[1000:], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, (_, _, name, url) in pb:
        pb.set_description(f"[{i+1}/{len(df)}]")
        pb.set_postfix_str(name)
        try:
            repo_archive_url = f"{ARCHIVE_PREFIX}{name}{ARCHIVE_SUFFIX}"
            r = requests.get(repo_archive_url, stream=True)
            if not r.ok:
                fd.write(f"{name},\t\tWas not able to download archive\n")
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(output_folder)
            archive_name = z.filename
            z.close()
            if archive_name is not None:
                os.remove(os.path.join(output_folder, archive_name))                

        except Exception as e:
            fd.write(f"{name},\t\t{str(e)}\n")
    

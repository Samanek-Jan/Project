import sys, os
import pandas as pd

if __name__ == '__main__':
    file_path = "../data/raw/bigQueryRepoLog.csv"
    out_file = "../data/raw/bigQueryRepoLog_filtered.csv"
    
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=['name'])
    print(df[:10])
    df.to_csv(out_file, sep=",")
    
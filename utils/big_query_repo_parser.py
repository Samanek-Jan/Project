import sys, os
import pandas as pd

if __name__ == '__main__':
    files = [
        "bq-results-repos-2019.csv",
        "bq-results-repos-2020.csv",
        "bq-results-repos-2021.csv",
        "bq-results-repos-2022.csv",
    ]
    out_file = "bq-results-joint.csv"
    out_df = pd.concat([pd.read_csv(file, index_col=False) for file in files])
    out_df = out_df.drop_duplicates(subset=["name"])
    print(f"{len(out_df)} rows")
    out_df.to_csv(out_file, sep=",", index=False)
    
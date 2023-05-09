import sys, os
import re
from tqdm import tqdm
import json
import numpy as np

losses = []
d_val_bleu = []

if __name__ == "__main__":
    file = "../screenlog.0"
    loss_reg = re.compile("\[(\d+)/\d+\].+loss = (\d+\.\d+)")
    bleu_reg = re.compile("\[(\d+)/\d+\].+score: (\d+\.\d+)")
    val_bleu_reg = re.compile("(\d+)\. best ver. BLEU. = (\d+\.\d+), currect ver. BLEU. = (\d+\.\d+)")
    last_epoch = 0
    
    with open(file, "r") as fd:
        content_line = fd.readlines()
        
    for line in tqdm(content_line):
        if (loss_res := loss_reg.search(line)) is not None:
            epoch = int(loss_res.group(1))
            loss_val = float(loss_res.group(2))
            if len(losses) == 0:
                losses.append([loss_val])
                last_epoch = epoch
            elif last_epoch != epoch:
                losses[-1] = float(np.mean(losses[-1]))
                losses.append([loss_val])
                last_epoch = epoch
            else:
                losses[-1].append(loss_val)
                
        elif (val_bleu_res := val_bleu_reg.search(line)) is not None:
            epoch = int(val_bleu_res.group(1))
            val_best_bleu = float(val_bleu_res.group(2))
            val_cur_bleu = float(val_bleu_res.group(3))
            d_val_bleu.append({"current" : val_cur_bleu, "best" : val_best_bleu})

    losses[-1] = float(np.mean(losses[-1]))

    with open("loss_vals.json", "w") as fd:
        json.dump(losses, fd, indent=2)
    
    with open("valid_bleu.json", "w") as fd:
        json.dump(d_val_bleu, fd, indent=2)
    
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import json

font = {
        'weight' : 'bold',
        'size'   : 12}


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
matplotlib.rc("font", **font)

if __name__ == '__main__':
    loss_file = "loss_vals.json"
    val_file = "valid_bleu.json"
    
    with open(loss_file, 'r') as fd:
        losses = json.load(fd)
    
    with open(val_file, 'r') as fd:
        vals = json.load(fd)
    
    best_bleus = []
    current_bleus = []
    for val in vals:
        best_bleus.append(val["best"])
        current_bleus.append(val["current"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].set_title("Loss over epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].plot(losses)
    
    ax[1].set_title("BLEU over epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("BLEU")
    ax[1].plot(current_bleus, label="epoch BLEU")
    ax[1].plot(best_bleus, label="best BLEU")
    ax[1].legend()
    
    fig.tight_layout()
    fig.savefig("plots.png")
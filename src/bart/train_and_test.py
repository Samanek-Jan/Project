from collections import OrderedDict
from copy import deepcopy
import os, sys
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError
import torchmetrics
from tqdm import tqdm
import argparse
import transformers
import json
from torchmetrics.functional.text.rouge import rouge_score
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed

from src.bart.datasets.config import DEVICE
from src.bart.datasets.collate_functor import CollateFunctor
from src.bart.config import BATCH_SIZE, LR, MAX_SEQUENCE_SIZE, MODELS_OUT_FOLDER, WARMUP_DURATION, MODEL_NAME
from src.bart.datasets.github_dataset.remote_dataset import RemoteDataset
from src.bart.datasets.local_dataset.local_dataset import LocalDataset
from src.bart.trainer import Trainer, prepare_dataloader

def format_memory_int(number : int) -> str:
    if number > 1e9:
        return "{:.2f}GB".format(number/1e9)
    elif number > 1e6:
        return "{:.2f}MB".format(number/1e6)
    elif number > 1e3:
        return "{:.2f}KB".format(number/1e3)
    return "{}B".format(number)
    
def main():
    print(f"Using {DEVICE}")
    global pretraining
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=1)
    argument_parser.add_argument("--model", "-d", type=str, default=None)
    args = argument_parser.parse_args()
    
    configuration = AutoConfig.from_pretrained(MODEL_NAME)
    configuration.classif_dropout = 0.1
    configuration.classifier_dropout = 0.1
    configuration.encoder_layerdrop = 0.2
    configuration.max_length = MAX_SEQUENCE_SIZE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, model_max_length=MAX_SEQUENCE_SIZE, add_bos_token=True)
    
    # Initializing model
    model = None
    optimizer = None
    model_dict = {}
    if args.model is not None:
        model = AutoModelForSeq2SeqLM.from_config(configuration).to(DEVICE)
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict["model_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        optimizer.load_state_dict(model_dict["optimizer_dict"])
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    collate_f = CollateFunctor(tokenizer)
    
    train_dataset = LocalDataset(tokenizer, "train")
    valid_dataset = LocalDataset(tokenizer, "valid")
        
    train_dataloader = prepare_dataloader(train_dataset, BATCH_SIZE, collate_f)
    valid_dataloader = prepare_dataloader(valid_dataset, BATCH_SIZE, collate_f)
    
    scheduler = transformers.get_linear_schedule_with_warmup(                
            optimizer = optimizer,
            num_warmup_steps = WARMUP_DURATION,
            num_training_steps = args.epoch_n,
            last_epoch=model_dict.get("epoch", -1)
    )
    
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer, scheduler, 10, args.epoch_n)
    trainer.train(model_dict)
    print("Done")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == "__main__":
    main()
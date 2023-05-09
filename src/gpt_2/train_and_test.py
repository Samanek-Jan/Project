from copy import deepcopy
import os, sys
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError
import torchmetrics
from tqdm import tqdm
import argparse
import json
import transformers
from transformers import AutoTokenizer
from torchmetrics.functional.text.rouge import rouge_score
from src.gpt_2.datasets.local_dataset.local_dataset import LocalDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, pipeline

from src.gpt_2.config import LR, MAX_SEQUENCE_SIZE, MODEL_NAME, TOKENIZER_NAME, WARMUP_DURATION, BATCH_SIZE
from src.gpt_2.datasets.config import DEVICE
from src.gpt_2.datasets.collate_functor import CollateFunctor
from src.gpt_2.trainer import Trainer, prepare_dataloader


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
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=1)
    argument_parser.add_argument("--model", "-d", type=str, default=None)
    args = argument_parser.parse_args()
    
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False, model_max_length=MAX_SEQUENCE_SIZE, padding_side = "left", padding=True, truncation=True,)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({
        "pad_token" : "<pad>"
    })

    # Initializing model
    model = None
    optimizer = None
    model_dict = {}
    configuration = GPT2Config.from_pretrained(MODEL_NAME)
    if args.model is not None:
        model_dict = torch.load(args.model, map_location="cpu")
        model = AutoModelForCausalLM.from_config(configuration).to(DEVICE)
        model.load_state_dict(model_dict["model_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)
        optimizer.load_state_dict(model_dict["optimizer_dict"])
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)

    collate_f = CollateFunctor(tokenizer)
    
    train_dataset = LocalDataset(tokenizer, "train")
    valid_dataset = LocalDataset(tokenizer, "valid")
        
    train_dataloader = prepare_dataloader(train_dataset, BATCH_SIZE, collate_f)
    valid_dataloader = prepare_dataloader(valid_dataset, 1, collate_f)
    
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    
    scheduler = transformers.get_constant_schedule_with_warmup(                
        optimizer = optimizer,
        num_warmup_steps = WARMUP_DURATION,
        last_epoch=model_dict.get("epoch", -1)
    )
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer, scheduler, 1, args.epoch_n)
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
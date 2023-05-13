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
from baseline.datasets.local_dataset.local_dataset import LocalDataset

from src.baseline.config import LR, MAX_SEQUENCE_SIZE, OUTPUT_FOLDER, TOKENIZER_NAME, WARMUP_DURATION, BATCH_SIZE
from src.baseline.datasets.config import DEVICE
from src.baseline.datasets.collate_functor import CollateFunctor
from src.baseline.model import Model
from src.baseline.search import GreedySearch
from src.baseline.trainer import Trainer, prepare_dataloader


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
    
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False, model_max_length=MAX_SEQUENCE_SIZE, add_bos_token=True)
    # tokenizer.add_tokens(["{", "}", "<", ">", ";", "[", "]", "&", "*"])
    tokenizer.add_special_tokens({
        "pad_token" : "<pad>"
    })
    configuration = {
        "num_encoder_layers" : 4,
        "num_decoder_layers" : 3,
        "d_model" : 400,
        "nhead" : 4,
        "dropout" : 0.1
    }
    # Initializing model
    model = None
    optimizer = None
    model_dict = {"configuration" : configuration}
    loss_fce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    if args.model is not None:
        model_dict = torch.load(args.model)
        model = Model(len(tokenizer), configuration.get("d_model"), loss_fce, tokenizer.pad_token_id, tokenizer.bos_token_id, configuration).to(DEVICE)
        model.load_state_dict(model_dict["model_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)
        optimizer.load_state_dict(model_dict["optimized_dict"])
    else:
        model = Model(len(tokenizer), configuration.get("d_model"), loss_fce, tokenizer.pad_token_id, tokenizer.bos_token_id, configuration).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)

    collate_f = CollateFunctor(tokenizer)
    
    train_dataset = LocalDataset(tokenizer, "train")
    valid_dataset = LocalDataset(tokenizer, "valid")
        
    train_dataloader = prepare_dataloader(train_dataset, BATCH_SIZE, collate_f)
    valid_dataloader = prepare_dataloader(valid_dataset, BATCH_SIZE, collate_f)
    
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    
    scheduler = transformers.get_linear_schedule_with_warmup(                
            optimizer = optimizer,
            num_warmup_steps = WARMUP_DURATION,
            num_training_steps = args.epoch_n,
            last_epoch=model_dict.get("epoch", -1)
    )
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer, scheduler, 41, args.epoch_n)
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
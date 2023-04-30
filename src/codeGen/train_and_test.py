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
from transformers import AutoConfig, CodeGenForCausalLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, pipeline

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


from src.codeGen.datasets.config import DEVICE
from src.codeGen.datasets.collate_functor import CollateFunctor
from src.codeGen.config import BATCH_SIZE, LR, MAX_SEQUENCE_SIZE, MODEL_NAME, WARMUP_DURATION
from src.codeGen.datasets.github_dataset.remote_dataset import RemoteDataset
from src.codeGen.datasets.local_dataset.local_dataset import LocalDataset
from src.codeGen.trainer import *

pretraining = False

def format_memory_int(number : int) -> str:
    if number > 1e9:
        return "{:.2f}GB".format(number/1e9)
    elif number > 1e6:
        return "{:.2f}MB".format(number/1e6)
    elif number > 1e3:
        return "{:.2f}KB".format(number/1e3)
    return "{}B".format(number)


def main(rank: int, world_size: int, save_every: int, total_epochs: int, model_d = None):
    
    ddp_setup(rank, world_size)
    
    configuration = AutoConfig.from_pretrained(MODEL_NAME)
    configuration.max_length = MAX_SEQUENCE_SIZE
    configuration.summary_first_dropout = 0.4
    configuration.resid_pdrop = 0.4
    configuration.embd_pdrop = 0.1
    configuration.attn_pdrop = 0.4
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, model_max_length=MAX_SEQUENCE_SIZE, padding_side='left')
    tokenizer.add_special_tokens({
        "pad_token" : "</pad>"
    })
    
    # Initializing model
    model = None
    optimizer = None
    model_dict = {}
    if model_d is not None:
        model = AutoModelForCausalLM.from_config(configuration).to(DEVICE+f":{rank}")
        model_dict = torch.load(model_d,map_location="cpu")
        model.load_state_dict(model_dict["model_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.005)
        optimizer.load_state_dict(model_dict["optimizer_dict"])
    else:
        model = CodeGenForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE+f":{rank}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.005)

    
    if pretraining:
        train_dataset = RemoteDataset(tokenizer, total_epochs)
        valid_dataset = LocalDataset(tokenizer, "valid")
    else:
        train_dataset = LocalDataset(tokenizer, "train")
        valid_dataset = LocalDataset(tokenizer, "valid")
    
    collate_fn = CollateFunctor(tokenizer)
    
    train_dataloader = prepare_dataloader(train_dataset, BATCH_SIZE, collate_fn)
    valid_dataloader = prepare_dataloader(valid_dataset, BATCH_SIZE, collate_fn)
    
    scheduler = transformers.get_constant_schedule_with_warmup(                
        optimizer = optimizer,
        num_warmup_steps = WARMUP_DURATION,
        last_epoch=model_dict.get("epoch", -1)
    )
    
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    try:
        trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer, scheduler, rank, save_every, 100, total_epochs)
        trainer.train(model_dict)
    except Exception as e:
        raise e
    finally:
        destroy_process_group()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == "__main__":
    print(f"Using {DEVICE}")
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=1)
    argument_parser.add_argument("--model", "-d", type=str, default=None)
    args = argument_parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, 1, args.epoch_n, args.model), nprocs=world_size)
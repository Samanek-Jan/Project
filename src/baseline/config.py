import torch
import random
import os, sys
import time

SEED = time.time()
random.seed(SEED)

TOKENIZER_NAME = "gpt2"

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# LR = 5e-4
LR = 1e-3
WARMUP_DURATION = 3
BATCH_SIZE = 23

UNK_TOKEN = "$unk$"
PAD_TOKEN = "$pad$"
EOS_TOKEN = "$eos$"
BOS_TOKEN = "$bos$"

MAX_SEQUENCE_SIZE = 700


OUTPUT_FOLDER = "/tmp/xsaman02/baseline/"
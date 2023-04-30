import torch
import random
import os, sys
import time

SEED = time.time()
random.seed(SEED)

TOKENIZER_NAME = "distilgpt2"

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# LR = 5e-4
LR = 5e-5
WARMUP_DURATION = 1000
BATCH_SIZE = 35

MAX_SEQUENCE_SIZE = 700


OUTPUT_FOLDER = "/tmp/xsaman02/baseline/"
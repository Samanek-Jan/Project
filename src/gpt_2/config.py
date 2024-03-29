import torch
import random
import os, sys
import time

SEED = time.time()
random.seed(SEED)
MODEL_NAME = "distilgpt2"
TOKENIZER_NAME = "distilgpt2"

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LR = 1e-4
# LR = 1e-4
WARMUP_DURATION = 3
BATCH_SIZE = 1

MODELS_OUT_FOLDER = "./"

MAX_SEQUENCE_SIZE = BATCH_SIZE*700
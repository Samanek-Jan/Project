import torch
import random
import os, sys

SEED = 123456
random.seed(SEED)

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# LR = 5e-4
LR = 1e-4
WARMUP_DURATION = 5
BATCH_SIZE = 10

MODELS_OUT_FOLDER = "./"

MAX_SEQUENCE_SIZE = 900
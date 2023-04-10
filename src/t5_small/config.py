import torch
import random
import os, sys

SEED = 123456
random.seed(SEED)

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# LR = 5e-4
LR = 5e-4
WARMUP_DURATION = 5
BATCH_SIZE = 20

MODELS_OUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp/xsaman02/models/t5_small")

UNK_TOKEN = "$unk$"
PAD_TOKEN = "$pad$"
EOS_TOKEN = "$eos$"
BOS_TOKEN = "$bos$"

MAX_SEQUENCE_SIZE = 512

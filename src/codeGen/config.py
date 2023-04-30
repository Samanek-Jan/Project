import torch
import random
import os, sys
import time

MODEL_NAME = "Salesforce/codegen-350M-multi"


SEED = 123123
random.seed(SEED)

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LR = 1e-5
# LR = 1e-4
WARMUP_DURATION = 200
BATCH_SIZE = 4

MODELS_OUT_FOLDER = "./"

MAX_SEQUENCE_SIZE = min(BATCH_SIZE*600, 2048)
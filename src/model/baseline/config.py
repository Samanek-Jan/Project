import torch
import random
import os, sys

SEED = 123456
random.seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LR = 1e-4
WARMUP_DURATION = 50
BATCH_SIZE = 64

MIN_X = 10
MAX_X = 128
MIN_Y = 10
MAX_Y = 128

MODELS_OUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../models/baseline")
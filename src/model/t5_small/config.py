import torch
import random
import os, sys

SEED = 123456
random.seed(SEED)

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LR = 5e-5
WARMUP_DURATION = 50
BATCH_SIZE = 20

MODELS_OUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../models/bart")

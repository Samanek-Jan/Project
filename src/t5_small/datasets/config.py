import torch
import random
from pymongo import MongoClient


RANDOM_SEED = 123456
random.seed(RANDOM_SEED)

NEWLINE_TOKEN = "\n"
SPACE_TOKEN = " "
NEWLINE_TOKEN_TRANSLATION = "\u0394" # Delta
SPACE_TOKEN_TRANSLATION = "\u03C3" # Sigma
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
LOWERCASE = False
SUBWORD_PREFIX = '$$'

SAMPLING_TYPES = {
    "NSP" : "NSP",
    "MLM" : "MLM",
}

MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
mongoDB = MongoClient(MONGODB_CONNECTION_STRING)
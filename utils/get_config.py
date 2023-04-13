import transformers
from transformers import AutoConfig

NAME = "distilgpt2"

config = AutoConfig.from_pretrained(NAME)
pass
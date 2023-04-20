import transformers
from transformers import AutoConfig

NAME = "t5-small"

config = AutoConfig.from_pretrained(NAME)
pass
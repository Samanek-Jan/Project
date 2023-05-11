import transformers
from transformers import AutoConfig

NAME = "Salesforce/codegen-350M-multi"

config = AutoConfig.from_pretrained(NAME)
pass
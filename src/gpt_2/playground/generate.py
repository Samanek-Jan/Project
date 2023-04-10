from collections import OrderedDict
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from bart.datasets.config import DEVICE

name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, model_max_length=1024, add_bos_token=True)
# model = AutoModelForSeq2SeqLM.from_pretrained(name)

configuration = AutoConfig.from_pretrained(name)
# model = AutoModelForSeq2SeqLM.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_config(configuration)
model_dict = torch.load("/tmp/xsaman02/bart/facebook-bart-large_pretrained.current.pt")
model_state_dict = OrderedDict()
for key, val in model_dict["model_dict"].items():
    model_state_dict[".".join(key.split(".")[1:])] = val
model.load_state_dict(model_state_dict)

# print(DEVICE)

text_input = """
// function for matrix multiplication
// param1: float** A
// param2: float** B
// param3: float** out
// param4: int row_size
// param5: int col_size
__device__ matrixMul(float** A, float** B, float** out, int row_size, int col_size) <mask>
"""

batch = tokenizer(text_input, return_tensors="pt", max_length=1024, truncation=True)
generated_ids = model.generate(batch["input_ids"], max_new_tokens=1024)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

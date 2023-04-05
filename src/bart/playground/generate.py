import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from bart.datasets.config import DEVICE

name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, model_max_length=1024, add_bos_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)

print(DEVICE)

text_input = """
// function for matrix multiplication
__device__ matrixMul(float** A, float** B, float** out, int row_size, int col_size) <mask>
"""

batch = tokenizer(text_input, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
generated_ids = model.generate(batch["input_ids"], max_new_tokens=1024)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

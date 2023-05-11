from collections import OrderedDict
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, pipeline, set_seed

from src.gpt_2.datasets.config import DEVICE

MAX_SIZE = 500
name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(name, use_fast=False, model_max_length=MAX_SIZE, add_bos_token=True)
tokenizer.add_special_tokens({
    "pad_token" : "<pad>"
})

# tokenizer.add_tokens(["<bot>"])

configuration = AutoConfig.from_pretrained("distilgpt2")
# model = AutoModelForCausalLM.from_pretrained(name)
# model.resize_token_embeddings(len(tokenizer))
model = AutoModelForCausalLM.from_config(configuration)
model_dict = torch.load(f"/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/models/gpt2/{name}.pcknot8.current.pt")
model.load_state_dict(model_dict["model_dict"])

# print(DEVICE)

text_input = """
// kernels
template <typename T>
__global__ void clMatLinspace_kernel(IN OUT T* pA, IN T a, IN T d, int dim)
""".strip()

text_input = """
// Kernel for vector addition
template <typename T>
__global__ void vectorAddition(T* v1, T* v2, T* out, int size)
""".strip()

# text_input = """
# Hi, how are you? <bot>
# """.strip()

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(1)
print(generator(text_input, max_length=MAX_SIZE, num_return_sequences=1)[0]["generated_text"])
# batch = tokenizer(text_input, return_tensors="pt", max_length=MAX_SIZE, truncation=True)
# generated_ids = model.generate(**batch, max_new_tokens=MAX_SIZE, do_sample=False)

# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

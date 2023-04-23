from collections import OrderedDict
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, pipeline, set_seed

from src.gpt_2.datasets.config import DEVICE

MAX_SIZE = 500
name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, model_max_length=MAX_SIZE, add_bos_token=True)
tokenizer.add_tokens(["{", "}", "<", ">", ";", "[", "]", "&", "*"])


# tokenizer.add_tokens(["<bot>"])

configuration = AutoConfig.from_pretrained(name)
# model = AutoModelForSeq2SeqLM.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_config(configuration)
model.resize_token_embeddings(len(tokenizer))
model_dict = torch.load(f"/tmp/xsaman02/models/t5-small/{name}_pretrained.current.pt", map_location="cpu")
model.load_state_dict(model_dict["model_dict"])

# print(DEVICE)

text_input = """
"supplement code:// function for matrix multiplication with __shared__ memory
// param1: float** A
// param2: float** B
// param3: float** out
// param4: int row_size
// param5: int col_size
__global__ void matrixMultiplication(float* A, float* B, float* out, int row_size, int col_size)
""".strip()

#text_input = """
#answer question:
#Hi, how are you?
#""".strip()

# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# set_seed(1)
# print(generator(text_input, max_length=256, num_return_sequences=1)[0]["generated_text"])
batch = tokenizer(text_input, return_tensors="pt", max_length=MAX_SIZE, truncation=True)
generated_ids = model.generate(**batch, max_new_tokens=MAX_SIZE, do_sample=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].replace(";", ";\n").replace("{", "{\n").replace("}", "}\n"))

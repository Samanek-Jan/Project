from collections import OrderedDict
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed

MAX_SIZE = 256

name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, model_max_length=MAX_SIZE, add_bos_token=True)
# model = AutoModelForSeq2SeqLM.from_pretrained(name)

configuration = AutoConfig.from_pretrained(name)
configuration.max_length = MAX_SIZE
# model = AutoModelForSeq2SeqLM.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_config(configuration)
model_dict = torch.load("/tmp/xsaman02/bart/facebook-bart-large_pretrained.current.pt", map_location="cpu")
# model_state_dict = OrderedDict()
# for key, val in model_dict["model_dict"].items():
#     model_state_dict[".".join(key.split(".")[1:])] = val
model.load_state_dict(model_dict["model_dict"])

# print(DEVICE)

text_input = """
supplement code:// function for matrix multiplication
// param1: float** A
// param2: float** B
// param3: float** out
// param4: int row_size
// param5: int col_size
__global__ void matrixMul(float* A, float* B, float* out, int row_size, int col_size)
""".strip()

# text_input = """
# // function for vector addition
# // param1 float* A,
# // param2 float* B
# // param3 float* out
# __global__ void matrixAdd(float* A, float* B, float* out)
# """.strip()



generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(1)
print(generator(text_input, max_length=MAX_SIZE, num_return_sequences=1)[0]["generated_text"])
print("\n\n#----------------------------#\n")

batch = tokenizer(text_input, return_tensors="pt", max_length=MAX_SIZE, truncation=True)
generated_ids = model.generate(**batch, max_new_tokens=MAX_SIZE)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

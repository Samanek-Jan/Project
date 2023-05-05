from collections import OrderedDict
import torch
import transformers
from transformers import AutoConfig, CodeGenForCausalLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, pipeline, set_seed

from src.codeGen.config import MODEL_NAME

MAX_SIZE = 500
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, model_max_length=MAX_SIZE, add_bos_token=True)
tokenizer.add_special_tokens({
    "pad_token" : "</s>"
})


configuration = AutoConfig.from_pretrained(MODEL_NAME)
# model = CodeGenForCausalLM.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_config(configuration)
model_dict = torch.load(f"/tmp/xsaman02/CodeGen/codegen.current.pt", map_location="cpu")
model_state_dict = OrderedDict()
for key, val in model_dict["model_dict"].items():
    model_state_dict[".".join(key.split(".")[1:])] = val
model.load_state_dict(model_state_dict)
# model.load_state_dict(model_dict["model_dict"])

# print(DEVICE)

text_input = """
// function for matrix multiplication
// param1: float* A
// param2: float* B
// param3: float* out
// param4: int row_size
// param5: int col_size
__global__ void matrixMul(float* A, float* B, float* out, int row_size, int col_size)
""".strip()

# text_input = """
# Hi, how are you?
# """.strip()

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(3)
print(generator(text_input, max_length=512, num_return_sequences=1)[0]["generated_text"].replace("{\t", "{\n\t").replace("}\t", "}\n\t").replace(";\t", ";\n\t").replace("{ ", "{\n ").replace("} ", "}\n ").replace("; ", ";\n "))
# batch = tokenizer(text_input, return_tensors="pt", max_length=MAX_SIZE, truncation=True)
# generated_ids = model.generate(**batch, max_new_tokens=MAX_SIZE, do_sample=False)

# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

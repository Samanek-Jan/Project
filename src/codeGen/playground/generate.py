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
model_dict = torch.load(f"../../../models/codeGen/codegen.evaluated.pt", map_location="cpu")
model_state_dict = OrderedDict()
# for key, val in model_dict["model_dict"].items():
#     model_state_dict[".".join(key.split(".")[1:])] = val
# model.load_state_dict(model_state_dict)
model.load_state_dict(model_dict["model_dict"])

# print(DEVICE)

text_input = """
// function for optimized matrix multiplication using shared memory
__global__ void optimizedMatrixMul(float* A, float* B, float* out, int row_size, int col_size)
""".strip()

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(1)
print(generator(text_input, max_length=800, num_return_sequences=1)[0]["generated_text"].replace("{\t", "{\n\t").replace("}\t", "}\n\t").replace(";\t", ";\n\t").replace("{ ", "{\n ").replace("} ", "}\n ").replace("; ", ";\n "))

set_seed(2)
print(generator(text_input, max_length=800, num_return_sequences=1)[0]["generated_text"].replace("{\t", "{\n\t").replace("}\t", "}\n\t").replace(";\t", ";\n\t").replace("{ ", "{\n ").replace("} ", "}\n ").replace("; ", ";\n "))

set_seed(3)
print(generator(text_input, max_length=800, num_return_sequences=1)[0]["generated_text"].replace("{\t", "{\n\t").replace("}\t", "}\n\t").replace(";\t", ";\n\t").replace("{ ", "{\n ").replace("} ", "}\n ").replace("; ", ";\n "))

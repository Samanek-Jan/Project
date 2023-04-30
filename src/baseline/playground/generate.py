from collections import OrderedDict
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, pipeline, set_seed
from src.baseline.model import Model
from src.baseline.search import BeamSearch, GreedySearch

MAX_SIZE = 500
DEVICE = "cpu"
name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, model_max_length=MAX_SIZE, add_bos_token=True)
tokenizer.add_special_tokens({
    "pad_token" : "<pad>"
})


model_dict = torch.load("/tmp/xsaman02/baseline/baseline.current.pt", map_location=DEVICE)
configuration = model_dict.get("configuration").get("configuration")
model = Model(len(tokenizer), configuration.get("d_model"), None, tokenizer.pad_token_id, configuration)
model.load_state_dict(model_dict["model_dict"])

searcher = GreedySearch(model, tokenizer, beam_size=2, max_length=512)


text_input = """
// function for matrix multiplication
// param1: float* A
// param2: float* B
// param3: float* out
// param4: int row_size
// param5: int col_size
__global__ void matrixMultiplication(float* A, float* B, float* out, int row_size, int col_size)
""".strip()

# text_input = """
# "supplement code:
# // Function for adding vectors
# // Takes vector v1 and v2 and sum element-wise to the out vector
# // Each vector has size of parameter size
# __global__ void addVectors(float* v1, float* v2, float* out, int size)
# """.strip()

batch = tokenizer(text_input, return_tensors="pt", max_length=MAX_SIZE, truncation=True)

sentences = searcher(batch["input_ids"], batch["attention_mask"])

print(sentences[0])

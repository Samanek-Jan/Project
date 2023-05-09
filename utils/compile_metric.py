from functools import partial
import os, sys
import subprocess
from typing import Set
from pymongo import MongoClient
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
from random import shuffle

MODEL_PATH = ""
TEST_FILE = "metric_file.cu"
RANDOM_SAMPLE = True
SAMPLE_RATIO = 0.2

error_weights = {
    "missing_token" : 0.2,
    "already_defined_errors" : 0.3,
    "wrong_val" : 0.4,
    "missing_type" : 0.6,
    "syntax_error" : 1.0
}


db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
validation_db = db["validation"]
train_db = db["validation"]


def load_model_dict(model_path):
    model_dict = torch.load(model_path, map_location="cpu")
    return model_dict

def get_kernel_prefixes(kernel : str) -> Set[str]:
    prefixes = set()
    one_line_kernel = kernel.replace("\n", " ")
    cuda_header_prefix_re = re.compile("__(host|global|device)__")
    
    prefixes.update(cuda_header_prefix_re.findall(one_line_kernel))
    return prefixes

def compile(file_content):
    
    with open(TEST_FILE, "w") as fd:
        fd.write(file_content)
    
    completedProcess = subprocess.run(["nvcc", TEST_FILE, "-o", f"{TEST_FILE}.o", "-std=c++17"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completedProcess.stdout.decode('utf-8')
    stderr = completedProcess.stderr.decode('utf-8')
    if os.path.isfile(f"{TEST_FILE}.o"):
        os.remove(f"{TEST_FILE}.o")
    
    return completedProcess.returncode, stdout, stderr

def calculate_error_metric(error_output : str, start_line_idx : int) -> float:
    error_lines = error_output.splitlines()
    escaped_test_file = re.escape(TEST_FILE)
    missing_token_re_list = [f"{escaped_test_file}\((\d+)\):.+ is undefined.*", f"{escaped_test_file}\((\d+)\):.*cannot determine which instance of overloaded function .+ is intended.*"]
    wrong_val_re_list = [f"{escaped_test_file}\((\d+)\):.*value of type .+ cannot be used to .+"]
    syntax_error_re_list = [f"{escaped_test_file}\((\d+)\):.*expected a .+", f"{escaped_test_file}\((\d+)\):.*unrecognized token.*", f"{escaped_test_file}\((\d+)\):.*return value type does not match the function type.*"]
    missing_type_re_list = [f"{escaped_test_file}\((\d+)\):.*explicit type is missing.*", f"{escaped_test_file}\((\d+)\):.*type name is not allowed.*"]
    inlude_error_re_list = [f"{escaped_test_file}\((\d+)\):.* error: (\S+): No such file or directory.*"]
    already_defined_error_re_list = [f"{escaped_test_file}\((\d+)\):.* error: .+ has already been .*"]
    
    error_analysis = {}
    
    def determine_line_error(line : int, error_type : str):
        if line < start_line_idx:
            return

        error_type_weight = error_weights.get(error_type)
        if error_type_weight == None:
            raise Exception(f"Invalid error_type: {error_type}")
        if error_analysis.get(line) == None:
            error_analysis[line] = error_type_weight
        else:
            error_analysis[line] = max(error_analysis.get(line), error_type_weight)
        
    for i, line in enumerate(error_lines, 1):
        res = None
        for regex in missing_token_re_list:
            if (res := re.match(regex, line)):
                line = int(res[1])
                determine_line_error(line, "missing_token")
                break
        
        if res is not None:
            continue
        
        for regex in wrong_val_re_list:
            if res := re.match(regex, line):
                line = int(res[1])
                determine_line_error(line, "wrong_val")
                break

        if res is not None:
            continue
        
        for regex in syntax_error_re_list:
            if res := re.match(regex, line):
                line = int(res[1])
                determine_line_error(line, "syntax_error")
                break
        
        if res is not None:
            continue
        
        for regex in missing_type_re_list:
            if res := re.match(regex, line):
                line = int(res[1])
                determine_line_error(line, "missing_type")
                break
        
        if res is not None:
            continue
        
        for regex in inlude_error_re_list:
            if res := re.match(regex, line):
                line = int(res[1])
                determine_line_error(line, "include_error")
                break
    
        for regex in already_defined_error_re_list:
            if res := re.match(regex, line):
                line = int(res[1])
                determine_line_error(line, "already_defined_errors")
                break
    
    
    return sum(error_analysis.values()) / len(error_lines)


if __name__ == "__main__":
    model_d = load_model_dict("/tmp/xsaman02/models/t5-small/t5-small_pretrained.best.pt")
    # kernel_d = cache_targets()
    
    assert len(model_d.get("source_sentences")) == len(model_d.get("target_sentences")) and len(model_d.get("target_sentences")) == len(model_d.get("pred_sentences"))
    
    result_d = {
        "__global__" : {
            "metric_score" : 0,
            "compiled" : 0,
            "not_compiled" : 0
        },
        "__device__" : {
            "metric_score" : 0,
            "compiled" : 0,
            "not_compiled" : 0
        },
        "__host__" : {
            "metric_score" : 0,
            "compiled" : 0,
            "not_compiled" : 0
        }
    }
    
    origin_n = len(model_d.get("source_sentences"))
    n = round(origin_n * SAMPLE_RATIO)
    
    if SAMPLE_RATIO < 1:
        print("Evaluating {} (found in DB) {}/{} samples".format("random" if RANDOM_SAMPLE else "first", n , origin_n))
        
    samples = list(zip(model_d.get("source_sentences"), model_d.get("target_sentences"), model_d.get("pred_sentences")))
    if RANDOM_SAMPLE:
        shuffle(samples)
    
    i = 0
    for src, tgr, prd in tqdm(samples):
        kernel = train_db.find_one({"body" : tgr})
        if kernel is None:
            kernel = validation_db.find_one({"body" : tgr})
            if kernel is None:
                continue
            
        last_additional_content = kernel.get("validation").get("iterations")[-1].get("additional_content")
        header_cuda_prefixes = kernel.get("metadata").get("header_cuda_prefixes")
        
        # REMOVE when not evaluating T5 model
        src = "//" + src 
        prd = prd.replace(";", ";\n").replace("{", "{\n").replace("}", "}\n").replace("int", "\nint").replace("float", "\nfloat")
        
        start_line_idx = (last_additional_content + src).count("\n")
        
        final_kernel_str = f"""
        {last_additional_content}
        {src}
        {prd}
        """.strip()
        
        retval, stdout, stderr = compile(final_kernel_str)
        i += 1    
        for prefix in header_cuda_prefixes:
            if retval == 0:
                result_d[prefix]["compiled"] += 1
            else:
                result_d[prefix]["not_compiled"] += 1
                result_d[prefix]["metric_score"] += calculate_error_metric(stderr, start_line_idx)
        
        if i >= n:
            break
    
    for prefix, results in result_d.items():
        print(f"Prefix: {prefix}")
        compiled = results.get("compiled") 
        not_compiled = results.get("not_compiled")
        total = compiled + not_compiled
        print("Final score (inverted): {:.4f}".format(1-(results.get("metric_score")/total)))
        print(f"Compiled:     {compiled} ({compiled/total:.2%})")
        print(f"Not compiled: {not_compiled} ({not_compiled/total:.2%})")
        print("----------------------------------------------------")

    get_total = lambda d: d.get("compiled") + d.get("not_compiled")
    get_weighted_score = lambda d, w: (1-(d.get("metric_score")/(d.get("compiled") + d.get("not_compiled"))))*w
    
    g_d = result_d.get("__global__")
    d_d = result_d.get("__device__")
    h_d = result_d.get("__host__")
    
    
    # Weights
    total_total = get_total(g_d) + get_total(d_d) + get_total(h_d)
    g_w = get_total(g_d) / total_total
    d_w = get_total(d_d) / total_total
    h_w = get_total(h_d) / total_total
    
    
    print("Total inverted score {:.3f}".format(get_weighted_score(g_d, g_w) + get_weighted_score(d_d, d_w) + get_weighted_score(h_d, h_w)))
    

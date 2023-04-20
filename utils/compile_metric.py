import os, sys
import subprocess
from pymongo import MongoClient
from tqdm import tqdm
import re

MODEL_PATH = ""
TEST_FILE = "metric_file.cu"

error_weights = {
    "missing_token" : 0.2,
    "already_defined" : 0.3,
    "wrong_val" : 0.4,
    "missing_type" : 0.6,
    "syntax_error" : 1.0
}


db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
validation_db = db["repo_metadata"]


def load_model():
    model = None
    ...
    return model

def cache_targets():
    d = {}
    for kernel in validation_db.find({}):
        d[kernel.get("body")] : kernel
    
    return d


def compile(file_content):
    
    with open(TEST_FILE, "w") as fd:
        fd.write(file_content)
    
    completedProcess = subprocess.run(["nvcc", TEST_FILE, "-o", f"{TEST_FILE}.o", "-std=c++17"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completedProcess.stdout.decode('utf-8')
    stderr = completedProcess.stderr.decode('utf-8')
    if os.path.isfile(os.path.join(os.getcwd(), f"{TEST_FILE}.o")):
        os.remove(f"{TEST_FILE}.o")
    
    return completedProcess.returncode, stdout, stderr

def calculate_error_metric(error_output : str) -> float:
    error_lines = error_output.splitlines()
    missing_token_re_list = [f"{TEST_FILE}(\d+):.*identifier .+ is undefined.*", f"{TEST_FILE}(\d+):.*cannot determine which instance of overloaded function .+ is intended.*"]
    wrong_val_re_list = [f"{TEST_FILE}(\d+):.*value of type .+ cannot be used to initialize an entity of type.*"]
    syntax_error_re_list = [f"{TEST_FILE}(\d+):.*expected a .+", f"{TEST_FILE}(\d+):.*unrecognized token.*", f"{TEST_FILE}(\d+):.*return value type does not match the function type.*"]
    missing_type_re_list = [f"{TEST_FILE}(\d+):.*explicit type is missing.*", f"{TEST_FILE}(\d+):.*type name is not allowed.*"]
    inlude_error_re_list = [f"{TEST_FILE}(\d+):.* error: (\S+): No such file or directory.*"]
    already_defined_error_re_list = [f"{TEST_FILE}(\d+):.* error: variable .+ has already been defined"]
    
    error_analysis = {}
    
    def determine_line_error(line : int, error_type : str):
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
            if (res := re.match(regex, line)) and res[1]:
                line = res[1]
                determine_line_error(line, "missing_token")
                break
        
        if res is not None:
            continue
        
        for regex in wrong_val_re_list:
            if res := re.match(regex, line):
                line = res[1]
                determine_line_error(line, "wrong_val")
                break

        if res is not None:
            continue
        
        for regex in syntax_error_re_list:
            if res := re.match(regex, line):
                line = res[1]
                determine_line_error(line, "syntax_error")
                break
        
        if res is not None:
            continue
        
        for regex in missing_type_re_list:
            if res := re.match(regex, line):
                line = res[1]
                determine_line_error(line, "missing_type")
                break
        
        if res is not None:
            continue
        
        for regex in inlude_error_re_list:
            if res := re.match(regex, line):
                line = res[1]
                determine_line_error(line, "include_error")
                break
    
        for regex in already_defined_error_re_list:
            if res := re.match(regex, line):
                error_analysis["already_defined_errors"].append({
                    "line" : line,
                    "line_idx" : i,
                    "error_regex" : regex,
                    "identifier" : res[1]
                })
                break
    
    
    return sum(error_analysis.values()) / len(error_lines)


if __name__ == "__main__":
    model_d = load_model()
    kernel_d = cache_targets()
    
    assert len(model_d.get("source_sentences")) == len(model_d.get("target_sentences")) and len(model_d.get("target_sentences")) == len(model_d.get("pred_sentences"))
    
    metric_score = 0
    n = len(model_d.get("source_sentences"))
    
    for src, tgr, prd in tqdm(zip(model_d.get("source_sentences"), model_d.get("target_sentences"), model_d.get("pred_sentences"))):
        kernel = kernel_d.get(prd)
        if kernel == None:
            continue
        
        last_additional_content = kernel.get("validation").get("iterations")[-1].get("additional_content")
        
        final_kernel_str = f"""
        {last_additional_content}
        {src}
        {prd}
        """.strip()
        
        retval, stdout, stderr = compile(final_kernel_str)
        
        if retval == 0:
            continue
        
        metric_score += calculate_error_metric(stderr)
    
    print(f"Final score {metric_score/n}")
        
        
        
        
        
    



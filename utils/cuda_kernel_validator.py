from ast import Dict
from typing import Collection, Tuple
from pymongo import MongoClient
import sys, os
import subprocess
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId
import re

MAX_COMPILE_TRIES = 15

def get_args():
    argparser = ArgumentParser()
    argparser.add_argument("--use_db", "-d", type=bool, default=True)
    argparser.add_argument("--in_folder", "-i", type=str, default=None)
    argparser.add_argument("--out_file", "-o", type=str, default="cuda_kernel_validator.log")
    
    args = argparser.parse_args()
    
    if not args.use_db and args.in_folder is None:
        raise IOError("You must specify --in_folder or --use_db")
    
    if not args.use_db and not os.path.isdir(args.in_folder):
        raise IOError("Specified in folder must be a directory")
        
    return args

def compile(file_content):
    tmp_file = "cuda_test_file.cu"
    
    with open(tmp_file, "w") as fd:
        fd.write(file_content)
    
    completedProcess = subprocess.run(["nvcc", tmp_file, "-o", f"{tmp_file}.o"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completedProcess.stdout.decode('utf-8')
    stderr = completedProcess.stderr.decode('utf-8')
    if os.path.isfile(f"{tmp_file}.o"):
        os.remove(f"{tmp_file}.o")
    
    return completedProcess.returncode, stdout, stderr
    
    
def analyze_error(error_output : str) -> Dict:
    error_lines = error_output.splitlines()
    missing_token_re_list = [r".*identifier \"(\S+)\" is undefined.*"]
    wrong_val_re_list = [r".*value of type \"(.+)\" cannot be used to initialize an entity of type \"(.+)\".*"]
    syntax_error_re_list = [r".*expected a \"\S+\"", r".*unrecognized token.*", r".*return value type does not match the function type.*"]
    missing_type_re_list = [r".*explicit type is missing.*", r".*type name is not allowed.*"]
    inlude_error_re_list = [r".* error: (\S+): No such file or directory.*"]
    
    error_analysis = {
        "missing_tokens" : [],
        "wrong_vals" : [],
        "syntax_errors" : [],
        "missing_types" : [],
        "include_errors" : []
    }
    
    for i, line in enumerate(error_lines, 1):
        for regex in missing_token_re_list:
            if res := re.match(regex, line):
                error_analysis["missing_tokens"].append({
                    "line" : line,
                    "line_idx" : i,
                    "error_regex" : regex,
                    "identifier" : res[1]
                })
                break
            
        if res is not None:
            continue
        
        for regex in wrong_val_re_list:
            if res := re.match(regex, line):
                error_analysis["wrong_vals"].append({
                    "line" : line,
                    "line_idx" : i,
                    "error_regex" : regex,
                    "given_type" : res[1],
                    "required_type" : res[2]
                })
                break
    
        if res is not None:
            continue
        
        
        for regex in syntax_error_re_list:
            if res := re.match(regex, line):
                error_analysis["syntax_errors"].append({
                    "line" : line,
                    "line_idx" : i,
                    "error_regex" : regex,
                })
                break
        
        if res is not None:
            continue
        
        for regex in missing_type_re_list:
            if res := re.match(regex, line):
                error_analysis["missing_types"].append({
                    "line" : line,
                    "line_idx" : i,
                    "error_regex" : regex,
                })
                break
        
        if res is not None:
            continue
        
        for regex in inlude_error_re_list:
            if res := re.match(regex, line):
                error_analysis["include_errors"].append({
                    "line" : line,
                    "line_idx" : i,
                    "error_regex" : regex,
                    "file" : res[1]
                })
                break
    
    return error_analysis

def apply_error_patch(error_analysis : Dict, file_metadata : Dict, db_collection):
    # error_analysis = {
    #     "missing_tokens" : [],
    #     "wrong_vals" : [],
    #     "syntax_errors" : [],
    #     "missing_types" : [],
    #     "include_errors" : []
    # }
    
    if error_analysis["syntax_errors"]:
        return None, False
    
    if error_analysis["include_errors"]:
        return None, False
    
    if error_analysis["missing_tokens"]:
        missing_tokens_obj_list = error_analysis["missing_tokens"]
        for token_obj in missing_tokens_obj_list:
            token_name = token_obj["identifier"]
            return find_missing_token(token_name, file_metadata, db_collection)
        
    return None, False
            
    

def find_missing_token(token_name, metadata, files_metadata):
    return search_file_for_token(metadata["repo_name"], metadata["filename"], token_name, files_metadata)
    
    
def search_file_for_token(repo_name : str, file_name : str, token_name : str, files_metadata):
    file_name = file_name.split("/")[-1]
    matching_headers = list(files_metadata.find({"repo_name" : repo_name, "filename" : file_name}))
    
    for header in matching_headers:
        for global_var_obj in header["global_vars"]:
            if global_var_obj["name"] == token_name:
                return global_var_obj["full_line"], True
    
    libraries = set()
    for header in matching_headers:
        library_names = [include["full_line"].strip() for include in header["includes"] if include["is_custom_include"]]
        for library_name in library_names:
            library_proposal, found = search_file_for_token(repo_name, library_name, token_name, files_metadata)
            if found:
                return library_proposal
            elif library_proposal is not None:
                libraries.update(library_proposal)
        
        third_party_libraries = [include["full_line"].strip() for include in header["includes"] if not include["is_custom_include"]]
        libraries.update(set(third_party_libraries))
    
    libraries = set(libraries)
    if len(libraries) == 0:
        return None, False
    
    return libraries, False
        
        
def validate_kernel(kernel : Dict, files_metadata) -> Dict:

    kernel_str = "\n{}{}\n".format(kernel["header"], kernel["body"])
    additional_content = """
int main() {
    return 0;
}"""

    kernel_validation = {
        "iterations" : [],
        "max_iterations" : MAX_COMPILE_TRIES,
        "compiled" : None,
    }
    
    used_libraries = set("cuda/helper_cuda.h")
    
    metadata : Dict = files_metadata.find_one({"_id" : ObjectId(kernel["file_metadata_id"])})
    if not metadata:
        raise MemoryError("Did not find kernel file metadata")
    
    for i in range(1, MAX_COMPILE_TRIES+1):
        
        kernel_validation_iteration = {
            "index" : i,
            "stderr" : None,
            "stdout" : None,
            "retval" : None,
            "additional_content" : None,
        }
        
        retval, stdout, stderr = compile("{}\n{}".format(additional_content, kernel_str))
        kernel_validation_iteration["stderr"] = stderr
        kernel_validation_iteration["stdout"] = stdout
        kernel_validation_iteration["retval"] = retval
        kernel_validation_iteration["additional_content"] = additional_content
        
        error_analyses = analyze_error(stderr)
        kernel_validation_iteration["error_analyses"] = error_analyses
        
        kernel_validation["iterations"].append(kernel_validation_iteration)
        
        if retval == 0:
            kernel_validation["compiled"] = True
            break
        
        new_additional_content, found = apply_error_patch(error_analyses, metadata, files_metadata)
        if not found and new_additional_content is not None:
            new_additional_content = new_additional_content.difference(used_libraries)
            if len(new_additional_content) == 0:
                kernel_validation["compiled"] = False
                break
            used_libraries.update(new_additional_content)
            new_additional_content = "\n".join(new_additional_content)
        elif new_additional_content is None:
            kernel_validation["compiled"] = False
            break
            
        additional_content = new_additional_content + "\n" + additional_content
        
    return kernel_validation
            
def get_nvcc_version():
    completedProcess = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completedProcess.returncode != 0:
        raise SystemError(f"Could not get NVCC info.\Retval: {completedProcess.returncode}\n Stderr: {completedProcess.stderr}\n")

    return completedProcess.stdout.decode("utf-8")
    

def validate_db():
    db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
    train = db["train"]
    validate = db["validation"]
    file_metadatas = db["file_metadata"]
    
    nvcc_info = get_nvcc_version()
    compiled = 0
    not_compiled = 0
    
    print("Validating train part")
    for kernel in tqdm(tuple(train.find())):
        validation_result = validate_kernel(kernel, file_metadatas)
        validation_result["nvcc_info"] = nvcc_info
        if validation_result["compiled"]:
            compiled += 1
        else:
            not_compiled += 1
        
        new_vals = {
            "$set" : {"validation" : validation_result}
        }
        train.update_one({"_id" : kernel["_id"]}, new_vals)
    
    print(f"Compiled successfully: {compiled}")
    print(f"Compilation failed   : {not_compiled}")
    
    compiled = 0
    not_compiled = 0
    
    print("Validating validate part")
    for kernel in tqdm(tuple(validate.find())):
        validation_result = validate_kernel(kernel, file_metadatas)
        validation_result["nvcc_info"] = nvcc_info
        if validation_result["compiled"]:
            compiled += 1
        else:
            not_compiled += 1
        new_vals = {
            "$set" : {"validation" : validation_result}
        }
        validate.update_one({"_id" : kernel["_id"]}, new_vals)
    
    print(f"Compiled successfully: {compiled}")
    print(f"Compilation failed   : {not_compiled}")
    
        
if __name__ == "__main__":
    validate_db()

    
    
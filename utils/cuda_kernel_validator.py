from ast import Dict
from typing import Collection
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
    missing_token_re_list = [re.compile(r"indentifier \"(\S+)\" is undefined")]
    wrong_val_re_list = [re.compile(r"value of type \"(.+)\" cannot be used to initialize an entity of type \"(.+)\"")]
    syntax_error_re_list = [re.compile(r"expected a \"\S+\""), re.compile(r"unrecognized token"), re.compile(r"return value type does not match the function type")]
    missing_type_re_list = [re.compile(r"explicit type is missing"), re.compile(r"type name is not allowed")]
    inlude_error_re_list = [re.compile(r"(\S+): No such file or directory")]
    
    error_analysis = {
        "missing_tokens" : [],
        "wrong_vals" : [],
        "syntax_errors" : [],
        "missing_types" : [],
        "include_errors" : []
    }
    
    for i, line in enumerate(error_lines, 1):
        for regex in missing_token_re_list:
            if res := regex.match(line):
                error_analysis["missing_tokens"].append({
                    "line" : line,
                    "idx" : i,
                    "error_regex" : regex.pattern,
                    "identifier" : res[1]
                })
                break
            
        if res is not None:
            continue
        
        for regex in wrong_val_re_list:
            if res := regex.match(line):
                error_analysis["wrong_vals"].append({
                    "line" : line,
                    "idx" : i,
                    "error_regex" : regex.pattern,
                    "given_type" : res[1],
                    "required_type" : res[2]
                })
                break
    
        if res is not None:
            continue
        
        
        for regex in syntax_error_re_list:
            if res := regex.match(line):
                error_analysis["syntax_errors"].append({
                    "line" : line,
                    "idx" : i,
                    "error_regex" : regex.pattern,
                })
                break
        
        if res is not None:
            continue
        
        for regex in missing_type_re_list:
            if res := regex.match(line):
                error_analysis["missing_types"].append({
                    "line" : line,
                    "idx" : i,
                    "error_regex" : regex.pattern,
                })
                break
        
        if res is not None:
            continue
        
        for regex in inlude_error_re_list:
            if res := regex.match(line):
                error_analysis["include_errors"].append({
                    "line" : line,
                    "idx" : i,
                    "error_regex" : regex.pattern,
                    "file" : res[1]
                })
                break
    
    return error_analysis

def apply_error_patch(error_analysis : Dict, file_metadata : Dict, db_collection) -> str:
    # error_analysis = {
    #     "missing_tokens" : [],
    #     "wrong_vals" : [],
    #     "syntax_errors" : [],
    #     "missing_types" : [],
    #     "include_errors" : []
    # }
    
    if error_analysis["syntax_errors"] is not []:
        return None
    
    if error_analysis["include_errors"] is not []:
        return None
    
    if error_analysis["missing_tokens"]:
        missing_tokens_obj_list = error_analysis["missing_tokens"]
        for token_obj in missing_tokens_obj_list:
            token_name = token_obj["identifier"]
            return find_missing_token(token_name, file_metadata, db_collection)
        
    return None
            
    

def find_missing_token(token_name, metadata, files_metadata):
    additional_content, _ = search_file_for_token(metadata["repo_name"], metadata["filename"], token_name, files_metadata)
    return additional_content
    
    
def search_file_for_token(repo_name : str, file_name : str, token_name : str, files_metadata):
    file_name = file_name.split("/")[-1]
    matching_headers = files_metadata.findMany({"repo_name" : repo_name, "filename" : file_name})
    
    for header in matching_headers:
        for global_var_obj in header["global_vars"]:
            if global_var_obj["name"] == token_name:
                return global_var_obj["full_line"], True
    
    libraries = ""
    for header in matching_headers:
        custom_libraries = [include["full_line"].strip() for include in header["includes"] if include["is_custom_include"]]
        for custom_library in custom_libraries:
            library_proposal, found = search_file_for_token(repo_name, custom_library["include_name"], token_name, files_metadata)
            if found:
                return library_proposal
            else:
                libraries += "\n" + library_proposal
        
        third_party_libraries = "\n".join([include["full_line"].strip() for include in header["includes"] if not include["is_custom_include"]])
        libraries += "\n" + third_party_libraries
    
    libraries_str = "\n".join(set([libraries.splitlines()]))
    if libraries_str.strip() == "":
        return None, False
    
    return libraries_str, False
        
        
def validate_kernel(kernel : Dict, files_metadata) -> Dict:

    kernel_str = "\n{}{}\n".format(kernel["header"], kernel["body"])
    additional_content = """
int main() \{
    return 0;
\}"""

    kernel_validation = {
        "iterations" : [],
        "max_tries" : MAX_COMPILE_TRIES,
        "retval" : None,
        "error" : None,
        "additional_content" : None,
        "compiled" : None
    }
    
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

        kernel_validation["iterations"].append(kernel_validation_iteration)
        
        if retval == 0:
            kernel_validation["retval"] = retval
            kernel_validation["stderr"] = stderr
            kernel_validation["additional_content"] = additional_content
            kernel_validation["compiled"] = True
            break
        
        error_analyses = analyze_error(stderr)
        new_additional_content = apply_error_patch(error_analyses, metadata, files_metadata)
        if new_additional_content is None:
            # Unknown error
            kernel_validation["retval"] = retval
            kernel_validation["stdout"] = stdout
            kernel_validation["stderr"] = stderr
            kernel_validation["additional_content"] = additional_content
            kernel_validation["compiled"] = False
            break
        
        additional_content = new_additional_content + "\n" + additional_content
        
    return kernel_validation
            
            
    

def validate_db():
    db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
    train = db["train"]
    validate = db["validate"]
    file_metadatas = db["file_metadata"]
    error_dict = {}
    
    print("Validating train part")
    for kernel in tqdm(tuple(train.find())):
        validation_result = validate_kernel(kernel, file_metadatas)
        kernel["validation_result"] = validation_result
        train.update_one({"_id" : kernel["_id"]}, kernel)
    
    print("Validating validate part")
    for kernel in tqdm(tuple(validate.find())):
        validation_result = validate_kernel(kernel, file_metadatas)
        kernel["validation_result"] = validation_result
        validate.update_one({"_id" : kernel["_id"]}, kernel)
            
    return error_dict
        
        
def validate_files(in_folder):
    train = os.path.join(in_folder, "train")
    validate = os.path.join(in_folder, "valid")
    error_dict = {}
    
    train_files = os.listdir(train)
    validate_files = os.listdir(validate)
    
    for file in tqdm(train_files):
        full_path = os.path.join(train, file)
        with open(full_path, "r") as fd:
            cuda_snippets = json.load(fd)
        
        for cuda_snippet in cuda_snippets:
            retval, stdout, stderr = compile("{}\n{}\n".format(cuda_snippet["header"], cuda_snippet["body"]))
            if retval != 0:
                error_dict[file] = {
                    "part" : "validate",
                    "retval": retval,
                    "cause": stderr,
                    "stdout": stdout
                }

    for file in tqdm(validate_files):
        full_path = os.path.join(train, file)
        with open(full_path, "r") as fd:
            cuda_snippets = json.load(fd)
        
        for cuda_snippet in cuda_snippets:
            retval, stdout, stderr = compile("{}\n{}\n".format(cuda_snippet["header"], cuda_snippet["body"]))
            if retval != 0:
                error_dict[file] = {
                    "part" : "train",
                    "retval": retval,
                    "cause": stderr,
                    "stdout": stdout
                }
                
    return error_dict

if __name__ == "__main__":
    validate_db()

    
    
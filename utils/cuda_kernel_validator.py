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
    ...

def apply_error_patch(error_analysis : Dict, file_metadata : Dict, db_collection) -> str:
    ...

def validate_kernel(kernel : Dict, file_metadatas) -> Dict:

    is_valid_kernel = False
    kernel_str = "\n{}{}\n".format(kernel["header"], kernel["body"])
    additional_content = """
int main() \{
    return 0;
\}"""

    kernel_validation = {
        "error_stack" : [],
        "stdout_stack" : [],
        "retval_stack" : [],
        "final_retval" : None,
        "final_error" : None
    }
    
    metadata : Dict = file_metadatas.findOne({"_id" : u"{}".format(kernel["file_metadata_id"])})
    
    while True:
        
        retval, stdout, stderr = compile("{}\n{}".format(additional_content, kernel_str))
        kernel_validation["error_stack"].append(stderr)
        kernel_validation["stdout_stack"].append(stdout)
        kernel_validation["retval_stack"].append(retval)
        
        if retval == 0:
            kernel_validation["final_retval"] = retval
            kernel_validation["final_error"] = stderr
            break
        
        error_analyses = analyze_error(stderr)
        additional_content = apply_error_patch(error_analyses, metadata, file_metadatas)
        if additional_content is None:
            # Unknown error
            kernel_validation["final_retval"] = retval
            kernel_validation["final_error"] = stderr
            break
        
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
        train.update_one({"_id" : u"{}".format(kernel["_id"])}, kernel)
    
    print("Validating validate part")
    for kernel in tqdm(tuple(validate.find())):
        validation_result = validate_kernel(kernel, file_metadatas)
        kernel["validation_result"] = validation_result
        validate.update_one({"_id" : u"{}".format(kernel["_id"])}, kernel)
            
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
    args = get_args()
    error_dict = {}
    with open(args.out_file, 'w') as fd:
        if args.use_db:
            # Log is in DB
            validate_db()
        else:
            error_dict = validate_files(args.in_folder)
            json.dump(error_dict, fd, indent=2)
    
    
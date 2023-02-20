from pymongo import MongoClient
import sys, os
import subprocess
import json
from argparse import ArgumentParser
from tqdm import tqdm

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
    main_function_prefix = """
#define BLOCK_SIZE 64
#define block_size 64
#define blockSize 64

int main() {
    return 0;
}
    """
    
    with open(tmp_file, "w") as fd:
        fd.write(main_function_prefix)
        fd.write(file_content)
    
    completedProcess = subprocess.run(["nvcc", tmp_file, "-o", f"{tmp_file}.o"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completedProcess.stdout.decode('utf-8')
    stderr = completedProcess.stderr.decode('utf-8')
    if os.path.isfile(f"{tmp_file}.o"):
        os.remove(f"{tmp_file}.o")
    
    return completedProcess.returncode, stdout, stderr
    
    

def validate_db():
    db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
    train = db["train"]
    validate = db["validate"]
    error_dict = {}
    validation = db["validation"]
    validation.drop()
    
    print("Validating train part")
    for obj in tqdm(tuple(train.find())):
        retval, stdout, stderr = compile("\n{}{}\n".format(obj["header"], obj["body"]))
        if retval != 0:
            validation.insert(
                {
                "snippet_id" : obj["_id"],
                "part" : "train",
                "retval": retval,
                "cause": stderr,
                "stdout": stdout
                }
            )    
    
    print("Validating validate part")
    for obj in tqdm(tuple(validate.find())):
        retval, stdout, stderr = compile("\n{}{}\n".format(obj["header"], obj["body"]))
        if retval != 0:
            validation.insert(
                {
                "snippet_id" : obj["_id"],
                "part" : "train",
                "retval": retval,
                "cause": stderr,
                "stdout": stdout
                }
            )
            
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
    
    
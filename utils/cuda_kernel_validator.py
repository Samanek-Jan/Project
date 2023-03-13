from ast import Dict
from typing import Collection, List, Tuple
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

from data.parser.parser import Parser

MAX_ATTEMPTS = 15
db = MongoClient("mongodb://localhost:27017")["cuda_snippets"]
train_db = db["train"]
validation_db = db["validation"]
files_metadata_db = db["file_metadata"]

train_db.create_index("_id")
train_db.create_index("repo_name")
train_db.create_index("kernel_name")
train_db.create_index("validation.compiled")

validation_db.create_index("_id")
validation_db.create_index("repo_name")
validation_db.create_index("kernel_name")
validation_db.create_index("validation.compiled")

def get_nvcc_version():
    completedProcess = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completedProcess.returncode != 0:
        raise SystemError(f"Could not get NVCC info.\Retval: {completedProcess.returncode}\n Stderr: {completedProcess.stderr}\n")

    return completedProcess.stdout.decode("utf-8")
    

compiled = 0
not_compiled = 0
nvcc_info = get_nvcc_version()

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
    
    completedProcess = subprocess.run(["nvcc", tmp_file, "-o", f"{tmp_file}.o", "-std=c++17"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

def apply_error_patch(error_analysis : Dict, file_metadata : Dict):    
    # if error_analysis["syntax_errors"]:
    #     return None, False
    
    # if error_analysis["include_errors"]:
    #     return None, False
    
    tokens_content = ""
    if error_analysis["missing_tokens"]:
        searched_tokens = set()
        tokens_content = ""
        missing_tokens_obj_list = error_analysis["missing_tokens"]
        for token_obj in missing_tokens_obj_list:
            token_name = token_obj["identifier"]
            if token_name in searched_tokens:
                continue
            searched_tokens.add(token_name)
            proposal = search_global_vars(token_name, file_metadata)
            if proposal is not None:
                tokens_content = f"{proposal}\n\n{tokens_content}"
                continue
            
            proposal = search_db(token_name, file_metadata["repo_name"])
            if proposal is not None:
                tokens_content = f"{proposal}\n\n{tokens_content}"
                continue
            
            proposal = search_custom_libs(token_name, file_metadata)
            if proposal is not None:
                tokens_content = f"{proposal}\n\n{tokens_content}"
                continue
    
    if tokens_content == "":
        third_party_libs, _ = get_third_party_libraries(file_metadata)
        tokens_content = "\n".join(third_party_libs)
        return tokens_content if tokens_content != "" else None, False
    else:
        remove_namespaces = Parser().remove_namespaces
        return remove_namespaces(tokens_content), True
            

def search_custom_library_for_token(token_name : str, custom_metadata : dict, searched_libs : set = set()):
    for global_var in custom_metadata["global_vars"]:
        if token_name == global_var["name"]:
            return global_var["full_line"], searched_libs
        
    proposal = search_full_text(token_name, custom_metadata)
    if proposal is not None:
        return proposal, searched_libs
        
    custom_libraries = list(files_metadata_db.find({"repo_name" : custom_metadata["repo_name"], "filename" : {"$in" : [include["include_name"] for include in custom_metadata["includes"] if include["is_custom_include"]]}}))
    
    for custom_library in custom_libraries:
        if custom_library["filename"] in searched_libs:
            continue
        searched_libs.update([custom_library["filename"]])
        proposal, sub_searched_libs = search_custom_library_for_token(token_name, custom_library, searched_libs)
        searched_libs.update(sub_searched_libs)
        if proposal is not None:
            return proposal, searched_libs
    
    return None, searched_libs


def get_third_party_libraries(custom_library : dict, searched_libs : set = set()):
    third_party_libraries = []
    for library in custom_library["includes"]:
        if library["include_name"].split("/")[-1] in searched_libs:
            continue
        
        if library["is_custom_include"]:
            library = files_metadata_db.find_one({"repo_name" : custom_library["repo_name"], "filename" : library["include_name"].split("/")[-1]})
            if library is None:
                continue
            searched_libs.update([library["filename"]])
            third_party_libs_addon, searched_libs_addon = get_third_party_libraries(library, searched_libs)
            third_party_libraries.extend(third_party_libs_addon)
            searched_libs.update(searched_libs_addon)
        else:
            third_party_libraries.append(library["full_line"].strip())
    
    return list(set(third_party_libraries)), searched_libs

    
def search_full_text(token_name : str, file_metadata : dict) -> str:
    class_re = re.compile("(\W|^\s*)class\W")
    struct_re = re.compile("(\W|^\s*)struct\W")
    token_re = re.compile(f"\W{token_name}\W")
    
    full_content_lines : List[str] = file_metadata["full_content"].splitlines()
    for i, line in enumerate(full_content_lines):
        if not token_re.match(line):
            continue
        
        non_comment_match = lambda reg, line: reg.match(line) is not None and not line.lstrip().startswith("//") and not line.lstrip().startswith("*") and not line.rstrip().endswith("*/") and not line.rstrip().endswith(";")
        
        proposal = []
        is_class = non_comment_match(class_re, line)
        is_struct = non_comment_match(struct_re, line)
        if is_class or is_struct:
            proposal.append(line)
            j = i - 1
            # Get prefix (template, rest of definition, ...)
            while j >= 0:
                j_line = full_content_lines[j]
                if j_line.lstrip().startswith("//") or \
                   j_line.rstrip().endswith("*/") or   \
                   j_line.rstrip().endswith("}") or    \
                   j_line.strip() == "":
                    break
                
                proposal.insert(0, j_line)
            
            # Get rest of body
            count_brackets = Parser().count_brackets
            brackets_count = count_brackets(line)
            j = i + 1
            while j < len(full_content_lines):
                j_line = full_content_lines[j]
                brackets_count = count_brackets(j_line)
                if brackets_count == 0:
                    last_bracket_idx = j_line.rfind("}")
                    if last_bracket_idx != -1:
                        j_line = j_line[:last_bracket_idx]
                    proposal.append(j_line[:last_bracket_idx])
                    break
                elif brackets_count < 0:
                    for _ in range(abs(brackets_count)):
                        last_bracket_idx = j_line.rfind("}")
                        j_line = j_line[:last_bracket_idx]
                    proposal.append(j_line)
                    break
                proposal.append(j_line)
            
            return Parser().remove_namespaces("\n".join(proposal))
    return None

def search_global_vars(token_name : str, file_metadata : dict):
    # 1. Try to find missing token in file global vars        
    for global_var in file_metadata["global_vars"]:
        if token_name == global_var["name"]:
            return global_var["full_line"]
            
    return None

def search_custom_libs(token_name : str, file_metadata : dict):
    proposal, _ = search_custom_library_for_token(token_name, file_metadata)
    # custom_libraries = list(files_metadata_db.find({"repo_name" : file_metadata["repo_name"], "filename" : {"$in" : [include["include_name"] for include in file_metadata["includes"] if include["is_custom_include"]]}}))
    
    # # 2. Search for missing token in included custom libraries
    # for custom_library in custom_libraries:
    #     proposal, _ = search_custom_library_for_token(token_name, custom_library)
    #     if proposal is not None:
    #         return proposal
        
    return proposal

def search_db(token_name : str, repo_name : str):
    global compiled
    global not_compiled
    
    # Search in train part
    kernel = train_db.find_one({"repo_name" : repo_name, "kernel_name" : token_name, "$or" : [{"validation.compiled" : {"$exists" : False}}, {"validation.compiled" : True}]})
    if kernel != None:
        if kernel.get("validation"):
            return "{}\n{}".format(kernel["header"], kernel["body"])
        else:
            validation_result = validate_kernel(kernel)
            validation_result["nvcc_info"] = nvcc_info
            if validation_result["compiled"]:
                compiled += 1
            else:
                not_compiled += 1
            
            new_vals = {
                "$set" : {"validation" : validation_result}
            }
            train_db.update_one({"_id" : kernel["_id"]}, new_vals)
            if validation_result["compiled"]:
                add_content = "\n".join(validation_result["iterations"][-1]["additional_content"].splitlines()[:-4])
                return "{}\n{}\n{}".format(add_content, kernel["header"], kernel["body"])
            else:
                return None
                
    
    # Search in validation part
    kernel = validation_db.find_one({"repo_name" : repo_name, "kernel_name" : token_name, "$or" : [{"validation.compiled" : {"$exists" : False}}, {"validation.compiled" : True}]})
    if kernel != None:
        if kernel.get("validation"):
            return "{}\n{}".format(kernel["header"], kernel["body"])
        else:
            validation_result = validate_kernel(kernel)
            validation_result["nvcc_info"] = nvcc_info
            if validation_result["compiled"]:
                compiled += 1
            else:
                not_compiled += 1
            
            new_vals = {
                "$set" : {"validation" : validation_result}
            }
            validation_db.update_one({"_id" : kernel["_id"]}, new_vals)
            if validation_result["compiled"]:
                add_content = "\n".join(validation_result["iterations"][-1]["additional_content"].splitlines()[:-4])
                return "{}\n{}\n{}".format(add_content, kernel["header"], kernel["body"])
            else:
                return None
    return None
             
def validate_kernel(kernel : Dict) -> Dict:

    kernel_str = "\n{}{}\n".format(kernel["header"], kernel["body"])
    additional_content = """
int main() {
    return 0;
}"""

    kernel_validation = {
        "iterations" : [],
        "max_iterations" : MAX_ATTEMPTS,
        "compiled" : None,
    }
    
    metadata : Dict = files_metadata_db.find_one({"_id" : ObjectId(kernel["file_metadata_id"])})
    if not metadata:
        raise MemoryError("Did not find kernel file metadata")
    
    applied_third_party_libs = False
    for attempt_idx in range(MAX_ATTEMPTS):        
        kernel_validation_iteration = {
            "attempt_idx" : attempt_idx,
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
                
        patch_proposal, found = apply_error_patch(error_analyses, metadata)
        if found == True or (not applied_third_party_libs and patch_proposal is not None):
            if not found:
                applied_third_party_libs = True
            additional_content = f"{patch_proposal}\n\n{additional_content}"
        else:
            kernel_validation["compiled"] = False
            break

    return kernel_validation

def validate_db():    
    pbar = tqdm(tuple(train_db.find()))
    global compiled
    global not_compiled
    
    print("Validating train part")
    for kernel in pbar:
        # Already validated in recursion function
        if train_db.find_one({"_id" : kernel["_id"], "validation" : {"$exists" : True}}):
            continue
        
        validation_result = validate_kernel(kernel)
        validation_result["nvcc_info"] = nvcc_info
        if validation_result["compiled"]:
            compiled += 1
        else:
            not_compiled += 1
        
        pbar.set_description_str(f"Compiled ratio: {compiled/(compiled+not_compiled):.2%}")
        
        new_vals = {
            "$set" : {"validation" : validation_result}
        }
        train_db.update_one({"_id" : kernel["_id"]}, new_vals)
    
    print(f"Compiled successfully: {compiled}")
    print(f"Compilation failed   : {not_compiled}")
    
    compiled = 0
    not_compiled = 0
    pbar = tqdm(tuple(validation_db.find()))
    
    print("Validating validate part")
    for kernel in pbar:
        # Already validated in recursion function
        if train_db.find_one({"_id" : kernel["_id"], "validation" : {"$exists" : True}}):
            continue
        
        validation_result = validate_kernel(kernel)
        validation_result["nvcc_info"] = nvcc_info
        if validation_result["compiled"]:
            compiled += 1
        else:
            not_compiled += 1

        pbar.set_description_str(f"Compiled ratio: {compiled/(compiled+not_compiled):.2%}")

        new_vals = {
            "$set" : {"validation" : validation_result}
        }
        validation_db.update_one({"_id" : kernel["_id"]}, new_vals)
    
    print(f"Compiled successfully: {compiled}")
    print(f"Compilation failed   : {not_compiled}")
    
        
if __name__ == "__main__":
    train_db.update_many({}, {"$unset" : {"validation" : ""}})
    validation_db.update_many({}, {"$unset" : {"validation" : ""}} )
    
    validate_db()

    
    
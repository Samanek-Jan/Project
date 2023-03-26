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

files_metadata_db.create_index("filename")
files_metadata_db.create_index("repo_name")

TMP_FILE = "cuda_test_file.cu"

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
    
    with open(TMP_FILE, "w") as fd:
        fd.write(file_content)
    
    completedProcess = subprocess.run(["nvcc", TMP_FILE, "-o", f"{TMP_FILE}.o", "-std=c++17"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completedProcess.stdout.decode('utf-8')
    stderr = completedProcess.stderr.decode('utf-8')
    if os.path.isfile(f"{TMP_FILE}.o"):
        os.remove(f"{TMP_FILE}.o")
    
    return completedProcess.returncode, stdout, stderr

    
def analyze_error(error_output : str) -> Dict:
    error_lines = error_output.splitlines()
    missing_token_re_list = [r".*identifier \"(\S+)\" is undefined.*", r".*cannot determine which instance of overloaded function \"(\S+)\" is intended.*"]
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
    
    missing_tokens_set = set()
    
    for i, line in enumerate(error_lines, 1):
        for regex in missing_token_re_list:
            if (res := re.match(regex, line)) and res[1] not in missing_tokens_set:
                missing_tokens_set.add(res[1])
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

def apply_error_patch(error_analysis : Dict, file_metadata : Dict, queried_kernel_ids : set):    
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
            # Try to search in file and connected libraries            
            proposal = search_custom_libs(token_name, file_metadata)
            if proposal is not None:
                tokens_content = f"{proposal}\n\n{tokens_content}"
                continue
            
            # Try to search DB for same-named kernel
            proposal = search_db(token_name, file_metadata["repo_name"], queried_kernel_ids)
            if proposal is not None:
                tokens_content = f"{proposal}\n\n{tokens_content}"
                continue
    
    if tokens_content == "":
        third_party_libs_set = get_third_party_libraries(file_metadata)
        tokens_content = "\n".join(third_party_libs_set)
        return tokens_content if tokens_content != "" else None, False
    else:
        remove_namespaces = Parser().remove_namespaces_and_tags
        return remove_namespaces(tokens_content, remove_tags=False), True
            

def search_library_for_token(token_name : str, custom_metadata : dict, searched_libs : set = set()):
    for global_var in custom_metadata["global_vars"]:
        if token_name == global_var["name"]:
            return global_var["full_line"], searched_libs
        
    proposal = search_full_text(token_name, custom_metadata)
    if proposal is not None:
        return proposal, searched_libs
        
    libraries = files_metadata_db.find({"repo_name" : custom_metadata["repo_name"], "filename" : {"$in" : [include["include_name"].split("/")[-1] for include in custom_metadata["includes"]]}})
    
    for library in libraries:
        if str(library["_id"]) in searched_libs:
            continue
        searched_libs.add(str(library["_id"]))
        proposal, sub_searched_libs = search_library_for_token(token_name, library, searched_libs)
        searched_libs.update(sub_searched_libs)
        if proposal is not None:
            return proposal, searched_libs
    
    return None, searched_libs
             

def get_third_party_libraries(file_metadata : dict):
    repo_name = file_metadata["repo_name"]
    libraries : set = set()
    for library in file_metadata["includes"]:
        if library["include_name"].split("/")[-1] in libraries:
            continue
        
        custom_libs = [*list(train_db.find({"repo_name" : repo_name, "filename" : library["include_name"].split("/")[-1]})) \
                      ,*list(validation_db.find({"repo_name" : repo_name, "filename" : library["include_name"].split("/")[-1]}))]
        if len(custom_libs) == 0:
            libraries.add(library["include_name"].split("/")[-1])
        else:
            for custom_lib in custom_libs:
                libraries.update(get_third_party_libraries(custom_lib))
    
    return libraries
    

    
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
            
            return Parser().remove_namespaces_and_tags("\n".join(proposal))
    return None

def search_custom_libs(token_name : str, file_metadata : dict):
    proposal, _ = search_library_for_token(token_name, file_metadata)
    return proposal

def search_db(token_name : str, repo_name : str, queried_kernel_ids : set):
    global compiled
    global not_compiled
    
    # Search in train part
    kernel = train_db.find_one({"repo_name" : repo_name, "kernel_name" : token_name, "$or" : [{"validation.compiled" : {"$exists" : False}}, {"validation.compiled" : True}]})
    if kernel != None:
        if kernel.get("validation"):
            return "{}\n{}".format(kernel["header"], kernel["body"])
        elif str(kernel["_id"]) not in queried_kernel_ids:
            queried_kernel_ids.add(str(kernel["_id"]))
            
            validation_result = validate_kernel(kernel, queried_kernel_ids)
            validation_result["nvcc_info"] = nvcc_info
            
            new_vals = {
                "$set" : {"validation" : validation_result}
            }
            train_db.update_one({"_id" : kernel["_id"]}, new_vals)
            if validation_result["compiled"]:
                compiled += 1
                add_content = "\n".join(validation_result["iterations"][-1]["additional_content"].splitlines()[:-4])
                return "{}\n{}\n{}".format(add_content, kernel["header"], kernel["body"])
            else:
                not_compiled += 1
                return None
        return None
                
    
    # Search in validation part
    kernel = validation_db.find_one({"repo_name" : repo_name, "kernel_name" : token_name, "$or" : [{"validation.compiled" : {"$exists" : False}}, {"validation.compiled" : True}]})
    if kernel != None:
        if kernel.get("validation"):
            return "{}\n{}".format(kernel["header"], kernel["body"])
        elif str(kernel["_id"]) not in queried_kernel_ids:
            queried_kernel_ids.add(str(kernel["_id"]))
            validation_result = validate_kernel(kernel, queried_kernel_ids)
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
             
def validate_kernel(kernel : Dict, queried_kernel_ids : set = set()) -> Dict:

    kernel_str = "\n{}{}\n".format(kernel["header"], kernel["body"])
    additional_content = """
#include <cstdint>

using namespace std;

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
        raise MemoryError("Did not find kernel file metadata (kernel id: {})".format(str(kernel["_id"])))
    
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
                
        patch_proposal, found = apply_error_patch(error_analyses, metadata, queried_kernel_ids)
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
        if validation_db.find_one({"_id" : kernel["_id"], "validation" : {"$exists" : True}}):
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
    
    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)
    
    
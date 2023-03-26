import re
import sys, os
import random
from typing import Dict, Generator, Iterable, List
from copy import deepcopy
from tqdm import tqdm
from pymongo import MongoClient
import random


MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "cuda_snippets"

GPU_FILE_SUFFIXES = set(["cu", "hu", "cuh"])
HEADER_FILE_SUFFIXES = set(["h", "hpp", "hu", "cuh"])
COMPATIBLE_FILE_SUFFIXES = set([*GPU_FILE_SUFFIXES, *HEADER_FILE_SUFFIXES, "cpp", "cc", "rc"])
DATA_FILE_SUFFIX = ".data.json"

IN_FOLDER = "/tmp/xsaman02/raw"
# IN_FOLDER = "../../../data/raw"
TRAIN_RATIO = 0.85

class Parser:

    def __init__(self):
        
        self.bracket_counter                            : int   = 0
        self.parsed_object_list                         : List  = []
        self.is_current_file_gpu                        : bool  = False
        self.is_parsing_comment                         : bool  = False
        self.filename                                   : str   = "",
        self.searched_cuda_header_token_substitutions   : dict  = {"__DH__" : "__device____host__", "__dh__" : "__device____host__"}
        self.searched_cuda_header_token_set             : set   = set(["__device__", "__host__", "__global__", *self.searched_cuda_header_token_substitutions.keys()])
        

    def process_file(self, filename : str) -> List:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        # # self.logger.debug(f'Processing {filename}')

        if self.__is_file_valid(filename):
            self.filename = filename
            self.is_current_file_gpu = filename.split(".")[-1] in GPU_FILE_SUFFIXES
            return self.__process_file(filename)
        return [], {}

    def process_files(self, filenames : Iterable[str]) -> Generator[List, None, None]:
        """ Parse all given files

        Args:
            filenames (Iterable[str]): input files to be parsed

        Returns:
            Generator[List[ParsedObject]]: generator of list of lexical objects for each file
        """
        for filename in filenames:
            self.filename = filename
            yield self.process_file(filename)
            
    def process_str(self, content : str, filename : str) -> List:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        # self.logger.debug(f'Processing {filename}')
        return self.__process_str(content, filename)

    def __is_file_valid(self, filename : str) -> bool:
        """ Checking if file exists

        Args:
            filename (str): path to input file

        Returns:
            bool: flag indicating whether file exists
        """
        return os.path.isfile(filename) and filename.split(".")[-1] in COMPATIBLE_FILE_SUFFIXES

    def __process_file(self, filename : str) -> List:
        with open(filename, 'r', encoding='latin-1') as fd:
            content = fd.read()
            return self.__process_str(content, os.path.split(filename)[1])
    
    def remove_namespaces_and_tags(self, content : str, remove_namespaces=True, remove_tags=True):
        lines = content.splitlines(keepends=True)
        namespace_r = re.compile("((\S+)::)")
        tag_r = re.compile("(<\/?\S{1,20}\/?>)")
        clean_content = ""
        for line in lines:
            if remove_namespaces:
                match = namespace_r.search(line, endpos=300)
                while match is not None:
                    line = line.replace(match[1], "")
                    match = namespace_r.search(line)
            
            if remove_tags:
                match = tag_r.search(line)
                while match is not None:
                    line = line.replace(match[1], "")
                    match = tag_r.search(line)
                
            clean_content += line.rstrip() + "\n"

        return clean_content
    
    def __process_str(self, content : str, filename : str) -> List:
        
        kernels = []
        cuda_headers = self.__get_cuda_headers(content)
        
        for cuda_header in cuda_headers:
            end_comment_idx = cuda_header["start_idx"]
            body_start_idx = cuda_header["end_idx"]
            
            comment = self.__parse_comment(content, end_comment_idx)
            has_generated_comment = False
            if comment == "":
                comment = self.__generate_default_comment(cuda_header["header"])
                has_generated_comment = True
                
            body = self.__parse_body(content, body_start_idx)
            if body == "":
                continue
            
            # Get start space indent
            start_space_indent_size = 0
            for c in cuda_header["header"]:
                if c == " ":
                    start_space_indent_size += 1
                elif c == "\t":
                    start_space_indent_size += 2
                else:
                    break
            
            kernels.append({
                "comment"               : self.remove_namespaces_and_tags(comment),
                "header"                : self.__clean_header(self.remove_namespaces_and_tags(cuda_header["header"]), start_space_indent_size),
                "body"                  : self.__clean_body(self.remove_namespaces_and_tags(body, remove_tags=False), start_space_indent_size),
                "kernel_name"           : cuda_header["kernel_name"],
                "type"                  : "function",
                "is_from_cuda_file"     : self.is_current_file_gpu,
                "has_generated_comment" : has_generated_comment,
                "filename"              : filename
            })
        
        file_metadata = None
        if len(kernels) > 0 or filename.split(".")[-1] in HEADER_FILE_SUFFIXES:
            file_metadata = self.__get_file_metadata(content, filename)
        
        return kernels, file_metadata
    
    def __get_file_metadata(self, content : str, filename : str):
        file_metadata = {
            "filename" : filename.split("/")[-1],
        }
    
        includes, global_vars = self.__get_includes_and_global_vars(content)
        file_metadata["includes"] = includes
        file_metadata["global_vars"] = global_vars
        file_metadata["full_content"] = content
        file_metadata["is_header"] = filename.split(".")[-1] in HEADER_FILE_SUFFIXES
        
        return file_metadata
        
    def __get_includes_and_global_vars(self, content : str):
        includes = []
        global_vars = []
        lines = content.splitlines()
        include_re = r"^\s*#include\s*(<|\")\s*(\S+)\s*(\"|>)\s*$"
        define_re = r"^\s*#define\s+(\S+)\s+(.+)$"
        using_alias_re = r"^\s*using\s+(\S+)\s*=\s*.+$"
        global_var_with_val_re = r"^(\S.+)(?:(\s+(\S+))\s*=)(.+)\s*;\s*$"
        
        for i, line in enumerate(lines, 1):
            if (match := re.match(include_re, line)):
                includes.append({
                    "full_line" : self.remove_namespaces_and_tags(match[0], remove_tags=False),
                    "include_name" : self.remove_namespaces_and_tags(match[2].strip()),
                    "line" : i
                })
            elif (match := re.match(define_re, line)):
                global_vars.append({
                    "full_line" : self.remove_namespaces_and_tags(match[0], remove_tags=False),
                    "name" : match[1].strip(),
                    "value" : match[2].strip(),
                    "line" : i,
                    "type" : "define"
                })
            elif (match := re.match(using_alias_re, line)):
                if line.rfind(";") == -1:
                    continue
                global_vars.append({
                    "full_line" : self.remove_namespaces_and_tags(match[0]),
                    "name" : match[1].strip(),
                    "value" : None,
                    "line" : i,
                    "type" : "using_alias"
                })
            elif (match := re.match(global_var_with_val_re, line)):
                global_vars.append({
                    "full_line" : self.remove_namespaces_and_tags(match[0], remove_tags=False),
                    # "type" : self.remove_namespaces_and_tags(match[1].strip()),
                    "name" : self.remove_namespaces_and_tags(match[3].strip().lstrip("*").lstrip("&")),
                    "value" : match.group(4).strip(),
                    "line" : i,
                    "type" : "global_var"
                })
            # elif (match := re.match(global_var_re, line)):
            #     global_vars.append({
            #         "full_line" : match[0],
            #         "type" : match[1],
            #         "name" : match[3],
            #         "value" : None,
            #         "line" : i
            #     })
        
        return includes, global_vars
    
    def __split_name(self, name : str) -> List:
        # Split name to multiple words
        words = []
        word = ""
        for c in name:
            if c.isupper() and word != "":
                words.append(word)
                word = c.lower()
            elif c == "_":
                words.append(word)
                word = ""
            else:
                word += c
                
        if word != "":
            words.append(word)
        
        return words
    
    def __parse_header_params(self, params : str) -> Dict:
        # Get parameters
        params = map(lambda x: x.strip(), params.split(","))
        params_dict = {}
        for param in params:
            param = param.split(" ")
            param_name = param[-1]
            if param_name != "":
                params_dict[param_name] = "".join(param[:-1])
        return params_dict
    
    def __generate_default_comment(self, header : str) -> str:
        
        # Get kernel name
        oneline_header = header.replace("\n", " ")
        non_templated_kernel_header = r"^\s*(\S+\s)*\s*(__.+__\s*)+\s+(.+)\s+(\S+)\s*\((.*)\).*"
        templated_kernel_header = r"^\s*template\s*<(.*)>\s*(\S+\s)*\s*(__.+__\s*)+\s+(.+)\s+(\S+)\s*\((.*)\).*"
        templated_kernel_header2 = r"^\s*template\s*<(.*)>\s*(\S+\s)*\s*(.+)\s+(__.+__\s*)+\s+(\S+)\s*\((.*)\).*"
        
        
        description_call_set = ["Function", "Method", "Kernel"]
        description_template_call_set = ["Templated", "Generic"]
        
        if (res := re.match(non_templated_kernel_header, oneline_header)):
            splitted_name = self.__split_name(res[4])
            params_dict = self.__parse_header_params(res[5])
            generated_comment = "// {} for {}\n".format(random.choice(description_call_set), " ".join(splitted_name))
            return_type = res[3]
        elif (res := re.match(templated_kernel_header, oneline_header)):
            splitted_name = self.__split_name(res[5])
            params_dict = self.__parse_header_params(res[6])
            return_type = res[4]
            generated_comment = "// {}{} for {}\n".format(random.choice(description_template_call_set) + " " if random.random() < 0.5 else "", random.choice(description_call_set), " ".join(splitted_name))
        elif (res := re.match(templated_kernel_header2, oneline_header)):
            splitted_name = self.__split_name(res[5])
            params_dict = self.__parse_header_params(res[6])
            return_type = res[3]
            generated_comment = "// {}{} for {}\n".format(random.choice(description_template_call_set) + " " if random.random() < 0.5 else "", random.choice(description_call_set), " ".join(splitted_name))
            
        else:
            raise Exception(f"parser.__generate_default_comment: Error parsing header.\nHeader: {header}")
        
        generated_comment += self.__params_to_str(params_dict)
        if random.random() < 0.5 and oneline_header.find("__global__") == -1:
            generated_comment += f"\n// returns {return_type}"

        return generated_comment.strip() + "\n"
        
    def __params_to_str(self, params_dict, prefix="// "):
        print_types = random.random() < 0.5
        s = ""
        for i, (name, t) in enumerate(params_dict.items(), 1):
            s += "{}{}. param. {}{},\n".format(prefix, i, t+" " if print_types else "", name)
        
        return s.rstrip()
    
    def __get_kernel_name(self, header : str):
        copy_header = header.replace("\n", " ")
        r = re.compile("(?:([^\(]+)\()")
        res = r.match(copy_header)
        if res is None:
            raise ValueError(f"parser.__get_kernel_name: Could not parse kernel name. Header: {header}")
        
        return res[1].split(" ")[-1].strip()
    
    def __get_cuda_headers(self, content : str) -> List:
        cuda_prefix_function_regex = re.compile("({})+".format("|".join(self.searched_cuda_header_token_set)))
        template_regex = re.compile("^\s*template\s*<")
        lines = content.splitlines(keepends=True)
        cuda_headers = []
        
        found_cuda_header = False
        
        content_idx = 0
        cuda_header = None
        for line in lines:
            template_res = None
            header_res = None
            
            if (cuda_header is not None and found_cuda_header) or (header_res := cuda_prefix_function_regex.search(line, endpos=300)) or (template_res := re.match(template_regex, line)):
                if not found_cuda_header and header_res:
                    found_cuda_header = True
                
                if cuda_header is None and (template_res or header_res):                    
                    cuda_header = {
                        "start_idx" : content_idx,
                    }
                
            if cuda_header is not None and (start_body_idx := line.find("{")) != -1:
                end_header_idx = content_idx + start_body_idx
                cuda_header["end_idx"] = end_header_idx
                if found_cuda_header:                                                
                    # Cropping header by the const suffixes
                    kernel_header = content[cuda_header["start_idx"]:cuda_header["end_idx"]]
                    end_header_idx = kernel_header.rfind(")")
                    if end_header_idx != -1: # Should always be true
                        kernel_header = kernel_header[:end_header_idx+1]
                    for substitution, val in self.searched_cuda_header_token_substitutions.items():
                        kernel_header = kernel_header.replace(substitution, val)
                    
                    cuda_header["header"] = kernel_header
                    cuda_header["kernel_name"] = self.__get_kernel_name(cuda_header["header"])
                    cuda_headers.append(cuda_header)
                    
                found_cuda_header = False                        
                cuda_header = None
                    
            elif cuda_header is not None and line.find(";") != -1:
                found_cuda_header = False                        
                cuda_header = None
                
            content_idx += len(line)
                
        return cuda_headers
    
    
    def __parse_body(self, content, body_start_idx):
        i = body_start_idx
        self.is_parsing_comment = False
        bracket_count = 0
        body = ""
        while i < len(content):
            line = self.__get_line(content, i)
            bracket_count += self.count_brackets(line)
            if bracket_count == 0:
                last_bracket_idx = line.rfind("}")
                line = line[:last_bracket_idx]
                body = content[body_start_idx:i + len(line)+1]
                break
            if bracket_count < 0:
                for _ in range(-bracket_count):
                    last_bracket_idx = line.rfind("}")
                    if last_bracket_idx == -1:
                        raise ValueError("parser.__process_str: Failed to retract body brackets")
                    line = line[:last_bracket_idx]
                body = content[body_start_idx:i + len(line)+1]
                break
            i += len(line)
            
        return body
 
    def __get_line_back(self, content: str, end_idx : int) -> str:
        start_idx = end_idx
        while start_idx > 0:
            start_idx -= 1
            if content[start_idx] == '\n':
                return content[start_idx+1: end_idx]
        return content[start_idx: end_idx]
    
    def __parse_comment(self, content : str, end_idx : int) -> str:
        
        line = self.__get_line_back(content, end_idx-1)
        if not line.strip().startswith("//") and not line.strip().endswith("*/"):
            return ""
        
        comment_idx = end_idx - len(line) - 1
        line = line.strip()
        comment = [line]
        if line.startswith("//"):
            while (line := self.__get_line_back(content, comment_idx)) != "" and line.strip().startswith("//"):
                comment.append(line.strip())
                comment_idx -= (len(line) + 1)
                "\n".join(comment[::-1])
        elif line.find("/*") == -1:
            while (line := self.__get_line_back(content, comment_idx)).find("/*") == -1:
                comment.append(line.strip())
                comment_idx -= (len(line) + 1)
            comment.append(line.strip())
            return "\n".join(self.__clean_comment(comment[::-1]))
        
        return "\n".join(self.__clean_comment(comment[::-1]))
    
    def __clean_comment(self, comment : List[str]) -> List[str]:

        cleaned_comment = []
        for line in comment:
            i = 0
            while i < len(line):
                c = line[i]
                # Delete tags in the beginning of the line
                if c == "<" and (r := line.find(">", i, i+10)) != -1:
                    i = r
                
                elif c.isalnum():
                    cleaned_comment.append("// " + line[i:].strip())
                    break
                
                i += 1
        return cleaned_comment

    def __clean_header(self, header : str, start_space_indent : int) -> str: 
        return self.__adjust_indent(header, start_space_indent)       
    
    def __clean_body(self, body : str, start_space_indent : int) -> str:
        return self.__adjust_indent(body, start_space_indent)

    def __adjust_indent(self, content : str, start_space_indent : int = None):
        content_lines = content.splitlines()
        if content_lines == []:
            return content_lines
        
        cleaned_content_lines = []
        transform_tab_to_spaces = lambda line: line.replace("\t", "  ")
        
        if start_space_indent is None:
            first_line = transform_tab_to_spaces(content_lines[0])
            start_space_indent = len(first_line) - len(first_line.lstrip())
        
        for line in content_lines:
            line = transform_tab_to_spaces(line).rstrip()
            space_indent_size = len(line) - len(line.lstrip())
            if space_indent_size > start_space_indent:
                line = " " * (space_indent_size - start_space_indent) + line.lstrip()
            else:
                line = " " * (space_indent_size + start_space_indent) + line.lstrip()
            cleaned_content_lines.append(line)
        
        return "\n".join(cleaned_content_lines)
        
    def __get_line(self, content : str, start_idx : int) -> str:
        end_idx = start_idx
        while end_idx < len(content):
            c = content[end_idx]
            if c == '\n':
                return content[start_idx: end_idx + 1]
            end_idx += 1
            
        return content[start_idx: end_idx]
    
    def count_brackets(self, line : str):
        is_in_comment_block = self.is_parsing_comment
        bracket_sum = 0
        d = {"{" : 1, "}" : -1}
        
        for i, c in enumerate(line):
            if c == "/" and not is_in_comment_block:
                if line[i+1] == "/":
                    return bracket_sum
                elif line[i+1] == "*":
                    is_in_comment_block = True
                continue
                    
            elif is_in_comment_block:
                if c == "*" and line[i+1] == "/":
                    is_in_comment_block = False
                continue
            
            else:
                bracket_sum += d.get(c, 0)

        self.is_parsing_comment = is_in_comment_block
        return bracket_sum

# ----------------------------------------------------------------
# ------------------------ END OF PARSER -------------------------
# ----------------------------------------------------------------

def get_database():
    client = MongoClient(MONGODB_CONNECTION_STRING)
    return client[DATABASE_NAME]

def fetch_files(in_folder : str, root_path = "", repo_name = None, root=True) -> List[str]:
    wanted_files = []
    files = [file for file in os.listdir(in_folder)]
    if root:
        files = tqdm(files, leave=False)


    for file in files:
        if root:
            files.set_description(f"Fetching files ({len(wanted_files)})")
        
        full_path = os.path.join(in_folder, file)
        if os.path.isdir(full_path):
            if repo_name is None:
                repo_name = file
            wanted_files.extend(fetch_files(full_path, os.path.join(root_path, file), repo_name, True))
        
        elif file.split(".")[-1] in COMPATIBLE_FILE_SUFFIXES:
            wanted_files.append(
                {
                    "full_path" : full_path,
                    "root_path" : os.path.join(root_path, file),
                }
            )
            
    return wanted_files


def parse_folder() -> None:
    
    if not os.path.isdir(IN_FOLDER):
        raise Exception("in folder '%s' does not exist" % IN_FOLDER)
    
    elif TRAIN_RATIO < 0 or TRAIN_RATIO > 1:
        raise Exception("train ratio parameter out of bounds")
    
    print("Connecting to DB...", end="\r")
    db = get_database()
    train = db["train"]
    validation = db["validation"]
    invalid_files = db["invalid_files"]
    file_metadatas = db["file_metadata"]
    # Start new collections
    train.drop()
    validation.drop()
    invalid_files.drop()
    file_metadatas.drop()

    parser = Parser()
    
    train_data_counter = 0
    train_char_counter = 0
    
    valid_data_counter = 0
    valid_char_counter = 0

    scaling_const = 100
    variable_train_ratio = TRAIN_RATIO * scaling_const
    
    train_document_idx = 0
    valid_document_idx = 0
    
    repos = os.listdir(IN_FOLDER)
    repo_pbar = tqdm(repos, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=True)
    
    for repo_name in repo_pbar:
        repo_path = os.path.join(IN_FOLDER, repo_name)
        if not os.path.isdir(repo_path):
            continue
        
        repo_pbar.set_description(repo_name)
        
        wanted_files = fetch_files(repo_path)
        pbar = tqdm(wanted_files, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False)
        
        for file_obj in pbar:
            file_obj["root_path"] = os.path.join(repo_name, file_obj.get("root_path"))
            pbar.set_description("Processing")
            pbar.set_postfix_str(file_obj.get("root_path"))

            try:
                kernels, file_metadata = parser.process_file(file_obj.get("full_path"))
                if file_metadata is None:
                    continue
                
                file_metadata["repo_name"] = repo_name
                del file_obj["full_path"] 
                file_metadata = {
                    **file_metadata,
                    **file_obj
                }
                
                insert_result = file_metadatas.insert_one(file_metadata)
                
                for kernel in kernels:
                    kernel["file_metadata_id"] = str(insert_result.inserted_id)
                    kernel["repo_name"] = repo_name
                    is_train_data = random.random() * scaling_const < variable_train_ratio
                    
                    if is_train_data or kernel.get("has_generated_comment", False):
                        variable_train_ratio -= (1-TRAIN_RATIO)
                        train_data_counter += 1
                        train_char_counter += len(kernel.get("comment", "") + kernel.get("header", "") + kernel.get("body", ""))
                        kernel["index"] = train_document_idx
                        train.insert_one(kernel)
                        train_document_idx += 1
                    else:
                        variable_train_ratio += TRAIN_RATIO
                        valid_data_counter += 1
                        valid_char_counter += len(kernel.get("comment", "") + kernel.get("header", "") + kernel.get("body", ""))
                        kernel["index"] = valid_document_idx
                        validation.insert_one(kernel)
                        valid_document_idx += 1

            except Exception as e:
                invalid_files.insert_one({"file" : file_obj.get("root_path"), "exception" : str(e)})
            
    print("Train data stats:")
    print("\tTotal data: %d" % train_data_counter)
    print("\tTotal char. len: %d" % train_char_counter)          

    print("Valid data stats:")
    print("\tTotal data: %d" % valid_data_counter)
    print("\tTotal char. len: %d" % valid_char_counter)
    
    if train_data_counter == 0 or valid_data_counter == 0:
        return 
    
    print("Train ratio by data: {:.2%}".format(train_data_counter / (train_data_counter + valid_data_counter)))
    print("Train ratio by char: {:.2%}".format(train_char_counter / (train_char_counter + valid_char_counter)))


if __name__ == "__main__":
    
    parse_folder()
    
    
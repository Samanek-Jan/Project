import json
import re
import sys, os
import random
from typing import Dict, Generator, Iterable, List
from copy import deepcopy
from tqdm import tqdm
from pymongo import MongoClient
import random

from data.parser.parser_exceptions import BracketCountErrorException, InvalidCharacterException, InvalidTypeException, ParsingFunctionException, ProcessingObjectException
from data.parser.parsing_object import PARSED_FUNCTION_TEMPLATE, PARSING_OBJECTS, PARSING_TYPES


MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "cuda_snippets"

GPU_FILE_SUFFIXES = set(["cu", "c", "hu"])
COMPATIBLE_FILE_SUFFIXES = set([*GPU_FILE_SUFFIXES, "cpp", "h", "cc", "hpp", "rc"])
DATA_FILE_SUFFIX = ".data.json"

skipped_files = []

class Parser:

    def __init__(self):
        
        self.bracket_counter      : int                = 0
        self.parsed_object_list     : List               = []
        self.is_current_file_gpu  : bool               = False
        self.is_parsing_comment   : bool               = False
        self.filename             : str                = ""
        

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
        return []

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
            return self.__process_str(content, filename)
        
    
    def __process_str(self, content : str, filename : str) -> List:
        
        parsed_objects = []
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
            
            parsed_objects.append({
                "comment"       : comment,
                "header"        : cuda_header["header"],
                "body"          : body,
                "type"          : "function",
                "is_cuda"        : True,
                "is_from_cuda_file" : self.is_current_file_gpu,
                "has_generated_comment" : has_generated_comment,
                "filename" : filename
            })
                    
        return parsed_objects
    
    def __generate_default_comment(self, header : str) -> str:
        
        # Get kernel name
        oneline_header = header.replace("\n", " ")
        res = re.search(r"(^|.*\s+)(\S+)\s+(\S+)\s*\((.*)\).*", oneline_header)
        
        if res is None:
            raise Exception("parser.__generate_default_comment: Error finding kernel name.")
        
        # Split name to multiple words
        name = res[3]
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
            
        # Get parameters
        params = map(lambda x: x.strip(), res[4].split(","))
        params_dict = {}
        for param in params:
            param = param.split(" ")
            param_name = param[-1]
            params_dict[param_name] = "".join(param[:-1])
        
        generated_comment = """
// {}{} for {}
{}
{}
        """.format(res[1] + " " if random.random() > 0.5 else "", random.choice(["Function", "Method", "Kernel"]), " ".join(words) , self.__params_to_str(params_dict), f"// returns {res[2]}" if random.random() > 0.5 else "")
        
        return generated_comment.strip() + "\n"
        
    def __params_to_str(self, params_dict, prefix="// "):
        print_types = random.random() < 0.5
        s = ""
        for i, (name, t) in enumerate(params_dict.items(), 1):
            s += "{}{}. param. {}{},\n".format(prefix, i, t+" " if print_types else "", name)
        
        return s.rstrip()
        
    
    def __get_cuda_headers(self, content : str) -> List:
        # cuda_function_header_regex = r"^\s*(template<.+>)?\s*(__device__|__host__|__global__)+\s+\S+\s+\S+\(.*\)\s*$"
        cuda_prefix_function_regex = r"(__device__|__host__|__global__)+"
        template_regex = r"^\s*template<.+>\s*$"
        lines = content.split("\n")
        cuda_headers = []
        
        content_idx = 0
        cuda_header = None
        for i, line in enumerate(lines):
            if cuda_header != None or re.match(cuda_prefix_function_regex, line):
                if cuda_header is None and i > 0 and re.match(template_regex, lines[i-1].strip()):
                    start_header_idx = content_idx - len(lines[i-1]) - 1
                    cuda_header = {
                        "start_idx" : start_header_idx,
                    }
                elif cuda_header is None:
                    cuda_header = {
                        "start_idx" : content_idx,
                    }  
                
                if cuda_header is not None and (start_body_idx := line.find("{")) != -1:
                    end_header_idx = content_idx + start_body_idx
                    cuda_header["end_idx"] = end_header_idx
                    cuda_header["header"] = content[cuda_header["start_idx"]:cuda_header["end_idx"]]
                    cuda_headers.append(cuda_header)
                    cuda_header = None
                elif cuda_header is not None and line.find(";") != -1:
                    cuda_header = None
                
            content_idx += len(line)+1 # plus newline
                
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
    
    def line_start_with(self, content : str, idx : int) -> bool:
        while idx > 0:
            idx -= 1
            if content[idx] == "\n":
                return True
            elif content[idx].isalnum():
                return False
            
        return True
 
            
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
            return "\n".join(self.__transform_comment(comment[::-1]))
        
        return "\n".join(self.__transform_comment(comment[::-1]))
    
    def __transform_comment(self, comment : List[str]) -> List[str]:

        transformer_comment = []
        for line in comment:
            for i, c in enumerate(line):
                if c.isalnum():
                    transformer_comment.append("// " + line[i:].strip())
                    break
        return transformer_comment
        
        
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
        
        for i, c in enumerate(line[:-1]):
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

def get_databases():
    client = MongoClient(MONGODB_CONNECTION_STRING)
    return client[DATABASE_NAME]

def fetch_files(in_folder : str, root=True) -> List[str]:
    wanted_files = []
    files = [file for file in os.listdir(in_folder)]
    if root:
        files = tqdm(files, leave=False)


    for file in files:
        if root:
            files.set_description("Fetching files")
        
        full_path = os.path.join(in_folder, file)
        if os.path.isdir(full_path):
            wanted_files.extend(fetch_files(full_path, False))
        
        elif file.split(".")[-1] in COMPATIBLE_FILE_SUFFIXES:
            wanted_files.append(full_path)
            
    return wanted_files


def parse_folder(in_folder : str, 
                 train_folder : str, 
                 valid_folder : str, 
                 train_ratio : float
                 ) -> None:
    
    if not os.path.isdir(in_folder):
        raise Exception("in folder '%s' does not exist" % in_folder)
    
    elif not os.path.exists(train_folder):
        raise Exception("train folder '%s' does not exist" % train_folder)
    
    elif not os.path.exists(valid_folder):
        raise Exception("valid folder '%s' does not exist" % valid_folder)
    
    elif train_ratio < 0 or train_ratio > 1:
        raise Exception("train ratio parameter out of bounds")
    
    print("Connecting to DB...", end="\r")
    db = get_databases()
    train = db["train"]
    validation = db["validation"]
    
    # Start new collections
    train.drop()
    validation.drop()
    
    wanted_files = fetch_files(in_folder)
    parser = Parser()
    pbar = tqdm(wanted_files, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    train_data_counter = 0
    train_char_counter = 0
    
    valid_data_counter = 0
    valid_char_counter = 0

    scaling_const = 100
    variable_train_ratio = train_ratio * scaling_const
    
    for file in pbar:
        pbar.set_postfix_str("/".join(file.split("/")[len(in_folder.split("/")):]))

        try:
            parsed_objects = parser.process_file(file)
            if len(parsed_objects) == 0:
                continue
            
            
            for obj in parsed_objects:
                is_train_data = random.random() * scaling_const < variable_train_ratio
                if is_train_data or obj.get("has_generated_comment", False):
                    variable_train_ratio -= (1-train_ratio)
                    train_data_counter += 1
                    train_char_counter += len(obj.get("comment", "") + obj.get("header", "") + obj.get("body", ""))
                    train.insert_one(obj)
                    # out_file = os.path.join(out_folder, "{}_{}{}".format(train_files_counter, file.split("/")[-1], DATA_FILE_SUFFIX))
                else:
                    variable_train_ratio += train_ratio
                    valid_data_counter += 1
                    valid_char_counter += len(obj.get("comment", "") + obj.get("header", "") + obj.get("body", ""))
                    # out_file = os.path.join(out_folder, "{}_{}{}".format(valid_files_counter, file.split("/")[-1], DATA_FILE_SUFFIX))
                    validation.insert_one(obj)

            
            # with open(out_file, "w") as fd:
            #     json.dump(parsed_objects, fd, indent=2)
        except Exception as e:
            skipped_files.append({"file" : file, "exception" : str(e)})    
        
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


def clear_folders(train_folder, valid_folder):
    [os.remove(os.path.join(train_folder, file)) for file in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, file))]
    [os.remove(os.path.join(valid_folder, file)) for file in os.listdir(valid_folder) if os.path.isfile(os.path.join(valid_folder, file))]
    

if __name__ == "__main__":
    
    # parser = Parser()
    # test_file = "data/raw/oneflow/stack_op.cpp"
    # parsed_objs = parser.process_file(test_file)
    # print(json.dumps(parsed_objs, indent=2))
    # sys.exit(0)
    
    in_folder = "../raw"
    train_folder = "../processed/train"
    valid_folder = "../processed/valid"
    train_ratio = 0.8
    
    # print("Cleaning folders...", end="\r")
    # clear_folders(train_folder, valid_folder)
    
    parse_folder(in_folder, train_folder, valid_folder, train_ratio)
    
    with open("skipped_files.log", "w") as fd:
        json.dump(skipped_files, fd, indent=2)
    
    
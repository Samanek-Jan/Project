import json
import re
import sys, os
import random
from typing import Dict, Generator, Iterable, List
from copy import deepcopy
from tqdm import tqdm

from data.parser.parser_exceptions import BracketCountErrorException, InvalidCharacterException, InvalidTypeException, ParsingFunctionException, ProcessingObjectException
from data.parser.parsing_object import PARSING_OBJECTS, PARSING_TYPES

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
        
    def __reset(self):
        self.is_parsing_comment = False
    
    def __process_str(self, content : str, filename : str) -> List:
        
        self.__reset()
        
        content_idx = 0
        line_idx = 0
        comment = []
        parsed_objects = []
        
        while content_idx < len(content):
            line = self.__get_line(content, content_idx)
            line_idx += 1
            stripped_line = line.strip()
            
            if self.__is_comment(line):
                comment += line.rstrip()
                content_idx += len(line)
            elif stripped_line == "":
                comment.clear()
                content_idx += len(line)
            else:
            # elif len(comment) > 0:
                header = self.__check_for_header(content, content_idx)
                if header is None:
                    content_idx += len(line)
                    comment.clear()
                    continue
                
                content_idx += len(header)
                body : str = self.__check_for_body(content, content_idx)
                if body is None:
                    comment.clear()
                    continue
                content_idx += len(body)
                parsing_type = self.__determine_code_type(header)
                parsing_object = PARSING_OBJECTS.get(parsing_type, None)
                parsing_object["comment"] = ".\n".join(comment)
                parsing_object["header"] = header
                parsing_object["type"] = parsing_type
                parsing_object["body"] = body
                parsing_object["is_from_cuda_file"] = self.is_current_file_gpu
                parsing_object["is_cuda"] = self.__is_cuda_function(header, parsing_type)
                
                if parsing_type == PARSING_TYPES.CLASS:
                    inner_parser = Parser()
                    body_start_idx = body.find("{") + 1
                    body_end_idx = body.rfind("}")
                    parsed_objects.extend(inner_parser.process_str(body[body_start_idx : body_end_idx]))
                elif parsing_type == PARSING_OBJECTS.OBJECT:
                    inner_parser = Parser()
                    body_start_idx = body.find("{") + 1
                    body_end_idx = body.rfind("}")
                    inner_parsed_objects = inner_parser.process_str(body[body_start_idx : body_end_idx])
                    if len(inner_parsed_objects) > 0:
                        parsed_objects.extend(inner_parsed_objects)
                    else:
                        parsed_objects.append(parsing_object)
                    
                    comment = []
        
            # else:
            #     content_idx += len(line)
        return parsed_objects
          
    def __is_cuda_function(self, header : str, code_type : PARSING_TYPES):
        if code_type != PARSING_TYPES.FUNCTION:
            return False
        
        return header.find("__device__") != -1 or header.find("__global__") != 1 or header.find("__host__") != 1
        
            
    def __determine_code_type(self, header : str) -> PARSING_TYPES:

        class_regex = re.compile(r"^.*\s+(class)\s+.*$")
        struct_regex = re.compile(r"^.*\s+(class)\s+.*$")
        function_regex = re.compile(r"^(\S+\s+)+\S+\s*\((\s*\S+\s+\S+\s*,?)+\)\s*$")
        
        
        if class_regex.match(header) is not None or struct_regex.match(header) is not None:
            return PARSING_TYPES.CLASS
        header = header.replace("\n", " ")
        if function_regex.match(header) is not None:
            return PARSING_TYPES.FUNCTION

        return PARSING_TYPES.OBJECT
                
    def __check_for_body(self, content : str, body_start_idx : int) -> str:
        body_idx = body_start_idx
        if content[body_idx] != "{":
            raise InvalidCharacterException("Start body character is not '{'")
        
        bracket_counter = 0
        while body_idx < len(content):
            line = self.__get_line(content, body_idx)
            bracket_counter += self.count_brackets(line)
            if bracket_counter == 0:
                return content[body_start_idx : body_idx + len(line)]
            if bracket_counter < 0:
                raise BracketCountErrorException("Got problems with brackets!")
            else:
                body_idx += len(line)
                
        return None
        
    def __check_for_header(self, content : str, content_idx : int, max_header_size = 160) -> str:
        
        body_start = content.find("{", content_idx, content_idx + max_header_size)
        if body_start == -1:
            return None
        
        header = content[content_idx : body_start]
        header_lines = header.split("\n")
        for header_line in header_lines:
            header_line = header_line.strip()
            if self.__is_comment(header_line) or header_line == "":
                return None
            
        return header 
    
    def __get_line(self, content : str, idx : int):
        
        nl_char_index = idx
        while nl_char_index < len(content) and content[nl_char_index] != "\n":
            nl_char_index += 1
            
        return content[idx: nl_char_index+1]
    
    def __is_comment(self, line : str) -> bool:
        if line.lstrip().startswith("//"):
            return True
        elif line.lstrip().startswith("/*"):
            self.is_parsing_comment = True
            return True
        elif self.is_parsing_comment and line.find("*/") != -1:
            self.is_parsing_comment = False
            return True
        
        return False
    
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

def fetch_files(in_folder : str) -> List[str]:
    wanted_files = []
    files = [file for file in os.listdir(in_folder)]

    for file in files:
        full_path = os.path.join(in_folder, file)
        if os.path.isdir(full_path):
            wanted_files.extend(fetch_files(full_path))
        
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
    
    print("Fetching files...", end="\r")
    wanted_files = fetch_files(in_folder)
    parser = Parser()
    pbar = tqdm(wanted_files, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    train_files_counter = 0
    train_data_counter = 0
    train_char_counter = 0
    
    valid_files_counter = 0
    valid_data_counter = 0
    valid_char_counter = 0
    
    data_counter = lambda parsed_obj: 1 if parsed_obj["inner_objects"] == [] else len(parsed_obj["inner_objects"])
    
    for file in pbar:
        pbar.set_postfix_str("/".join(file.split("/")[len(in_folder.split("/")):]))
        is_train_data = random.random() < train_ratio
        out_folder = train_folder if is_train_data else valid_folder

        try:
            parsed_objects = parser.process_file(file)
            if len(parsed_objects) == 0:
                continue
            elif is_train_data:
                train_files_counter += 1
                train_data_counter += sum([data_counter(parsed_obj) for parsed_obj in parsed_objects])
                train_char_counter += len("".join([obj.get("comment", "") + obj.get("header", "") + obj.get("body", "") for obj in parsed_objects]))
                out_file = os.path.join(out_folder, "{}_{}{}".format(train_files_counter, file.split("/")[-1], DATA_FILE_SUFFIX))
            else:
                valid_files_counter += 1
                valid_data_counter += sum([data_counter(parsed_obj) for parsed_obj in parsed_objects])
                valid_char_counter += len("".join([obj.get("comment", "") + obj.get("header", "") + obj.get("body", "") for obj in parsed_objects]))
                out_file = os.path.join(out_folder, "{}_{}{}".format(valid_files_counter, file.split("/")[-1], DATA_FILE_SUFFIX))

            
            with open(out_file, "w") as fd:
                json.dump(parsed_objects, fd, indent=2)
        except Exception as e:
            # print("Skipped file\n")
            skipped_files.append({"file" : file, "exception" : str(e)})    
        
    print("Train data stats:")
    print("\tTotal files: %d" % train_files_counter)
    print("\tTotal data: %d" % train_data_counter)
    print("\tTotal char. len: %d" % train_char_counter)          

    print("Valid data stats:")
    print("\tTotal files: %d" % valid_files_counter)
    print("\tTotal data: %d" % valid_data_counter)
    print("\tTotal char. len: %d" % valid_char_counter)
    
    if train_files_counter == 0 or valid_files_counter == 0:
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
    
    print("Cleaning folders...", end="\r")
    clear_folders(train_folder, valid_folder)
    
    parse_folder(in_folder, train_folder, valid_folder, train_ratio)
    
    with open("skipped_files.log", "w") as fd:
        json.dump(skipped_files, fd, indent=2)
    
    
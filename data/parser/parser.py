from configparser import ParsingError
from genericpath import isdir
import json
import sys, os
import logging
import random
from typing import Dict, Generator, Iterable, List
from data.parser.class_parser import ClassParser
from data.parser.function_parser import FunctionParser
from data.parser.object_parser import ObjectParser
from data.parser.parser_exceptions import BracketCountErrorException, InvalidStateException, InvalidTypeException, ParsingFunctionException
from data.parser.struct_parser import StructParser
from copy import deepcopy
from tqdm import tqdm

READY_AUTOMATA_STATE = "r"
DOXYGEN_AUTOMATA_STATE = "d"
ONE_LINE_COMMENT_AUTOMATA_STATE = "olc"
CODE_AUTOMATA_STATE = "c"
INVALID_AUTOMATA_STATE = "i"
FINISH_AUTOMATA_STATE = "f"

class_keyword = "class"
struct_keyword = "struct"
GPU_FILE_SUFFIXES = set(["cu", "c", "hu"])

class Parser:

    def __init__(self):
        # self.logger = logging.getLogger('Parser').setLevel(logging.INFO)
        self.parsers = {
            "class": ClassParser(),
            "struct": StructParser(),
            "function": FunctionParser(),
            "object": ObjectParser(),
        }
        
        self.automata_states = {
            READY_AUTOMATA_STATE            : self.__ready_parsing_state, # Ready to parse state
            DOXYGEN_AUTOMATA_STATE          : self.__parsing_doxygen_state, # Doxygen state
            ONE_LINE_COMMENT_AUTOMATA_STATE : self.__parsing_one_line_comment_state, # One line comment state
            CODE_AUTOMATA_STATE             : self.__parsing_code_state, # Code state
            INVALID_AUTOMATA_STATE          : self.__parsing_invalid_state, # Invalid status state
            FINISH_AUTOMATA_STATE           : self.__object_ready_state, # Finish block state (Should never reach here)
        }
        
        self.current_status       : str                = READY_AUTOMATA_STATE # basically enum of automata states
        self.current_code_block   : List[str]          = []
        self.current_parsing_type : str                = None    # One of [None, class, function, struct]
        self.bracket_counter      : int                = 0
        self.parsedObjectList     : List               = []
        self.is_current_file_gpu  : bool               = False
        self.line_counter         : int                = 1
        

    def process_file(self, filename : str) -> List:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        # # self.logger.debug(f'Processing {filename}')

        if self.__is_file_valid(filename):
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
            yield self.process_file(filename)
            
    def process_str(self, content : str, filename : str) -> List:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        # self.logger.debug(f'Processing {filename}')
        return self.__process_str(content.split("\n"), filename)

    def __is_file_valid(self, filename : str) -> bool:
        """ Checking if file exists

        Args:
            filename (str): path to input file

        Returns:
            bool: flag indicating whether file exists
        """
        return os.path.isfile(filename)

    def __process_file(self, filename : str) -> List:
        with open(filename, 'r') as fd:
            lines = fd.readlines()
            return self.__process_str(lines, filename)
    
    def __process_str(self, lines : Iterable[str], filename : str) -> List:
        
        """ Processing a file content
        
        Args:
            filename (str): path to input file
            
        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        self.parsedObjectList.clear()
        self.current_status = "r"
        self.bracket_counter = 0
        self.line_counter = 1
        
        for _, line in enumerate(lines):
            self.current_status = self.automata_states[self.current_status](line)
            self.line_counter += 1
            
        return self.parsedObjectList
                
    def __ready_parsing_state(self, line : str) -> str:
        line = line.strip()
        if line.startswith("/*"):                
            return self.__parsing_doxygen_state(line) # Return doxygen state
        
        elif line.startswith("//"):
            return self.__parsing_one_line_comment_state(line) # Return one line comment state
        
        elif len(line.strip()) > 0:
            return self.__parsing_code_state(line)

        else:
            self.current_code_block.clear()                
            return READY_AUTOMATA_STATE # Return ready for parsing state
    
    def __parsing_doxygen_state(self, line: str) -> str:
        self.current_code_block.append(line.rstrip())
        return READY_AUTOMATA_STATE if line.rstrip().endswith("*/") else DOXYGEN_AUTOMATA_STATE # If doxygen ends, return code state else return doxygen state
            
        
    def __parsing_one_line_comment_state(self, line: str) -> str:
        self.current_code_block.append(line.rstrip())
        return READY_AUTOMATA_STATE # Return ready state
    
    def __parsing_code_state(self, line: str) -> str:
        
        if self.current_parsing_type != None:

            self.current_code_block.append(line.rstrip())            
            self.bracket_counter += self.count_brackets(line)
            
            if self.bracket_counter == 0:
                return self.__object_ready_state()

            elif self.bracket_counter < 0:
                message = f"Error counting brackets with result {self.bracket_counter} on line {self.line_counter}\n"
                # self.logger.error(message)
                raise BracketCountErrorException(message)
            
            return CODE_AUTOMATA_STATE
        
        else:    
            if line.strip() == "":
                self.current_parsing_type = "object"
                return self.__object_ready_state()
            
            tokens = line.split(" ")
            self.current_code_block.append(line.rstrip())
            self.bracket_counter += self.count_brackets(line)
            
            # Is class
            if self.current_parsing_type == None and class_keyword in tokens:
                self.current_parsing_type = "class"
            
            # Is struct
            elif self.current_parsing_type == None and struct_keyword in tokens:
                self.current_parsing_type = "struct"
            
            # Is probably definition of a function
            elif self.current_parsing_type == None and self.bracket_counter > 0:
                self.current_parsing_type = "function"
            
            # Only declaration
            elif line.find(";") != -1:
                self.current_parsing_type = "object" if self.current_parsing_type == None else self.current_parsing_type
                return self.__object_ready_state()
            
            # Error parsing brackets
            elif self.bracket_counter < 0:
                message = f"Error counting brackets with result {self.bracket_counter} on line {self.line_counter}\n"
                # self.logger.error(message)
                raise BracketCountErrorException(message)
            
            return CODE_AUTOMATA_STATE


    def __parsing_invalid_state(self, line: str) -> str:
        self.current_parsing_type = None  
        self.current_code_block.clear()
        
        if self.bracket_counter > 0:
            self.bracket_counter += self.count_brackets(line)
            if self.bracket_counter == 0:
                return READY_AUTOMATA_STATE
            else:
                return INVALID_AUTOMATA_STATE
        
        elif self.bracket_counter < 0:
            message = f"Error counting brackets with result {self.bracket_counter} on line {self.line_counter}\n"
            # self.logger.error(message)
            raise BracketCountErrorException(message)
        
        else:
            return READY_AUTOMATA_STATE if line.strip() == "" else INVALID_AUTOMATA_STATE
        
    def __object_ready_state(self, *args):
        if self.current_parsing_type == None:
            message = f"No type of parsed object on line {self.line_counter}"
            # self.logger.error(message)
            raise ParsingError(message)
        
        self.parsedObjectList.append(self.parsers[self.current_parsing_type].process(self.current_code_block, self.is_current_file_gpu))
        self.current_status = READY_AUTOMATA_STATE
        self.current_code_block.clear()
        self.current_parsing_type = None
        
        return READY_AUTOMATA_STATE
                
        
    def count_brackets(self, line : str):
        d = {"{" : 1, "}" : -1}
        return sum([d[c] for c in line if c in d])

# ----------------------------------------------------------------
# ------------------------ END OF PARSER -------------------------
# ----------------------------------------------------------------

COMPATIBLE_FILE_SUFFIXES = set(["c", "cpp", "cc", "h", "hpp", "cu", "hu"])
DATA_FILE_SUFFIX = ".data.json"

skipped_files = []

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
    
    for file in pbar:
        pbar.set_postfix_str("/".join(file.split("/")[len(in_folder.split("/")):]))
        is_train_data = random.random() < train_ratio
        out_folder = train_folder if is_train_data else valid_folder

        try:
            parsed_objects = parser.process_file(file)
            if is_train_data:
                train_files_counter += 1
                train_data_counter += len(parsed_objects)
                train_char_counter += len("".join([obj.get("comment", "") + obj.get("header", "") + obj.get("body", "") for obj in parsed_objects]))
                out_file = os.path.join(out_folder, "{}_{}{}".format(train_files_counter, file.split("/")[-1], DATA_FILE_SUFFIX))
            else:
                valid_files_counter += 1
                valid_data_counter += len(parsed_objects)
                valid_char_counter += len("".join([obj.get("comment", "") + obj.get("header", "") + obj.get("body", "") for obj in parsed_objects]))
                out_file = os.path.join(out_folder, "{}_{}{}".format(valid_files_counter, file.split("/")[-1], DATA_FILE_SUFFIX))

            
            with open(out_file, "w") as fd:
                json.dump(parsed_objects, fd)
        except Exception as e:
            # print("Skipped file\n")
            skipped_files.append(file)      
            
    print("Train data stats:")
    print("\tTotal files: %d" % train_files_counter)
    print("\tTotal data: %d" % train_data_counter)
    print("\tTotal char. len: %d" % train_char_counter)          

    print("Valid data stats:")
    print("\tTotal files: %d" % valid_files_counter)
    print("\tTotal data: %d" % valid_data_counter)
    print("\tTotal char. len: %d" % valid_char_counter)
    
    print("Train ratio by data: {:.2%}".format(train_data_counter / (train_data_counter + valid_data_counter)))
    print("Train ratio by char: {:.2%}".format(train_char_counter / (train_char_counter + valid_char_counter)))
    

if __name__ == "__main__":
    print("Processing Raw data. Don't forget to clear folders!")
    in_folder = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/raw"
    train_folder = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/processed/train"
    valid_folder = "/mnt/c/Users/jansa/Škola/Ing_2023_zima/Diplomka/Project/data/processed/valid"
    train_ratio = 0.8
    parse_folder(in_folder, train_folder, valid_folder, train_ratio)
    
    if len(skipped_files) > 0:
        with open("skipped_files.log", "w") as fd:
            fd.write('\n'.join(skipped_files) + '\n')
    
    
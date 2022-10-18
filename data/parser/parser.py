from configparser import ParsingError
from io import TextIOWrapper
import re
import sys, os
import logging
from typing import Generator, Iterable, List
from data.parser.class_parser import ClassParser
from data.parser.function_parser import FunctionParser
from data.parser.parser_exceptions import BracketCountErrorException, InvalidStateException, ParsingFunctionException
from data.parser.struct_parser import StructParser
from data.parser.parsing_object import ParsedObject
from copy import deepcopy

READY_AUTOMATA_STATE = "r"
DOXYGEN_AUTOMATA_STATE = "d"
ONE_LINE_COMMENT_AUTOMATA_STATE = "olc"
CODE_AUTOMATA_STATE = "c"
INVALID_AUTOMATA_STATE = "i"
FINISH_AUTOMATA_STATE = "f"

DATA_FILE_SUFFIX = "data.json"

class_keyword = "class"
struct_keyword = "struct"

class Parser:

    def __init__(self):
        self.logger = logging.getLogger('Parser')
        self.parsers = {
            "class": ClassParser(),
            "struct": StructParser(),
            "function": FunctionParser()
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
        self.current_parsing_type : str                = None # One of [None, class, function, struct]
        self.bracket_counter      : int                = 0
        self.parsedObjectList     : List[ParsedObject] = []
        

    def process_file(self, filename : str) -> List[ParsedObject]:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        self.logger.debug(f'Processing {filename}')

        if self.__is_file_valid(filename):
            return self.__process_file(filename)
        
    def process_str(self, content : str, filename : str) -> List[ParsedObject]:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        self.logger.debug(f'Processing {filename}')
        return self.__process_str(content.split("\n"), filename)

    def process_files(self, filenames : Iterable[str]) -> Generator[List[ParsedObject], None, None]:
        """ Parse all given files

        Args:
            filenames (Iterable[str]): input files to be parsed

        Returns:
            Generator[List[ParsedObject]]: generator of list of lexical objects for each file
        """
        for filename in filenames:
            yield self.process_file(filename)

    def __is_file_valid(self, filename : str) -> bool:
        """ Checking if file exists

        Args:
            filename (str): path to input file

        Returns:
            bool: flag indicating whether file exists
        """
        return os.path.isfile(filename)

    def __process_file(self, filename : str) -> List[ParsedObject]:
        with open(filename, 'r') as fd:
            lines = fd.readlines()
            return self.__process_str(lines, filename)
    
    def __process_str(self, lines : Iterable[str], filename : str) -> List[ParsedObject]:
        
        """ Processing a file content
        
        Args:
            filename (str): path to input file
            
        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        self.parsedObjectList.clear()
        
        for _, line in enumerate(lines):
            self.current_status = self.automata_states[self.current_status](line)
            
        return self.parsedObjectList
                
    def __ready_parsing_state(self, line : str) -> str:
        line = line.strip()
        if line.startswith("/*"):
            # Parsed unimportant characters before
            if self.current_code_block is not []:
                self.current_code_block = []
                
            return self.__parsing_doxygen_state(line) # Return doxygen state
        
        elif line.startswith("//"):
            return self.__parsing_one_line_comment_state(line) # Return one line comment state
        
        elif len(line.strip()) > 0:
            return self.__parsing_code_state(line)

        else:
            # Parsed unimportant characters before
            if self.current_code_block is not []:
                self.current_code_block = []
                
            return READY_AUTOMATA_STATE # Return ready for parsing state
    
    def __parsing_doxygen_state(self, line: str) -> str:
        self.current_code_block.append(line.rstrip())
        return READY_AUTOMATA_STATE if line.rstrip().endswith("*/") else DOXYGEN_AUTOMATA_STATE # If doxygen ends, return code state else return doxygen state
            
        
    def __parsing_one_line_comment_state(self, line: str) -> str:
        self.current_code_block.append(line.rstrip())
        return READY_AUTOMATA_STATE # Return code state
    
    def __parsing_code_state(self, line: str) -> str:
        if self.current_parsing_type != None:

            self.current_code_block.append(line.rstrip())            
            self.bracket_counter += count_brackets(line)
            
            if self.bracket_counter == 0:
                return self.__object_ready_state()

            elif self.bracket_counter < 0:
                message = f"Error counting brackets with result {self.bracket_counter} on line \"{line}\"\n"
                self.logger.error(message)
                raise BracketCountErrorException(message)
            
            return CODE_AUTOMATA_STATE
        
        else:    
            if line.strip() == "":
                return INVALID_AUTOMATA_STATE
            
            tokens = line.split(" ")
            
            # Is class
            if class_keyword in tokens:
                self.current_parsing_type = "class"
            
            # Is struct
            elif struct_keyword in tokens:
                self.current_parsing_type = "struct"
            
            self.current_code_block.append(line.rstrip())            
            self.bracket_counter += count_brackets(line)
            
            if self.bracket_counter > 0 and self.current_parsing_type == None:
                self.current_parsing_type = "function"
            
            # Only declaration
            elif line.rstrip().endswith(";"):
                self.current_parsing_type = "function" if self.current_parsing_type == None else self.current_parsing_type
                return self.__object_ready_state()
            
            # Error parsing brackets
            elif self.bracket_counter < 0:
                message = f"Error counting brackets with result {self.bracket_counter} on line \"{line}\"\n"
                self.logger.error(message)
                raise BracketCountErrorException(message)
            
            return CODE_AUTOMATA_STATE


    def __parsing_invalid_state(self, line: str) -> str:
        self.current_parsing_type = None
        self.current_code_block.clear()
        
        if self.bracket_counter > 0:
            self.bracket_counter += count_brackets(line)
            if self.bracket_counter == 0:
                return READY_AUTOMATA_STATE
            else:
                return INVALID_AUTOMATA_STATE
        
        elif self.bracket_counter < 0:
            message = f"Error counting brackets with result {self.bracket_counter} on line \"{line}\"\n"
            self.logger.error(message)
            raise BracketCountErrorException(message)
        
        else:
            return READY_AUTOMATA_STATE if line.strip() == "" else INVALID_AUTOMATA_STATE
        
    def __object_ready_state(self, *args):
        if self.current_parsing_type == None:
            message = "No type of parsed object"
            self.logger.error(message)
            raise ParsingError(message)
        
        self.parsedObjectList.append(self.parsers[self.current_parsing_type].process(self.current_code_block))
        self.current_status = READY_AUTOMATA_STATE
        self.current_code_block.clear()
        self.current_parsing_type = None
        
        return READY_AUTOMATA_STATE
                
        
def count_brackets(line : str):
    d = {"{" : 1, "}" : -1}
    return sum([d[c] for c in line if c in d])
    
from io import TextIOWrapper
import sys, os
import logging
from typing import Generator, Iterable, List
from data.parser.parser_exceptions import ParsingFunctionException
from parsing_object import ParsedObject, ParsedClass, ParsedFunction, ParsedStruct
from copy import deepcopy


class Parser:

    def __init__(self):
        self.logger = logging.getLogger('Parser')
        
        self.behaviour_enum = {
            "r"   : self.__ready_parsing_state, # Ready to parse state
            "d"   : self.__parsing_doxygen_state, # Doxygen state
            "olc" : self.__parsing_one_line_comment_state, # One line comment state
            "c"   : self.__parsing_code_state, # Code state
            "i"   : self.__parsing_invalid_state, # Invalid status state
            "f"   : lambda l: None, # Finish block state (Should never reach here)
        }
        self.current_status = "r"
        self.current_code_block = []
        

    def process_one(self, filename : str) -> List[ParsedObject]:
        """ Parse and process given file

        Args:
            filename (str): input filename to be parsed

        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """
        
        self.logger.debug(f'Processing {filename}')

        if self.__is_file_valid(filename):
            return self.__process(filename)

    def process_all(self, filenames : Iterable[str]) -> Generator[List[ParsedObject]]:
        """ Parse all given files

        Args:
            filenames (Iterable[str]): input files to be parsed

        Returns:
            Generator[List[ParsedObject]]: generator of list of lexical objects for each file
        """
        for filename in filenames:
            yield self.process_one(filename)

    def __is_file_valid(self, filename : str) -> bool:
        """ Checking if file exists

        Args:
            filename (str): path to input file

        Returns:
            bool: flag indicating whether file exists
        """
        return os.path.isfile(filename)

    def __process(self, filename : str) -> List[ParsedObject]:
        """ Processing a file content
        
        Args:
            filename (str): path to input file
            
        Returns:
            List[ParsedObject]: Lexical list of parsed objects
        """

        with open(filename, 'r') as fd:
            lines = fd.readlines()
            
        for i, line in enumerate(lines):
            self.current_status = self.behaviour_enum[self.current_status](line)
            
            if self.current_status == "r": # Code block is ready state
                ...


    def __ready_parsing_state(self, line : str) -> str:
        line = line.strip()
        if line.startswith("/*"):
            self.current_code_block.append(line)
            return "d" # Return doxygen state
        
        elif line.startswith("//"):
            self.current_code_block.append(line)
            return "olc" # Return one line comment state
        
        elif len(line) > 0:
            self.current_code_block.append(line)
            return "c" # Return code state

        else:
            return "r" # Return ready for parsing state
    
    def __parsing_doxygen_state(self, line: str) -> str:
        self.current_code_block.append(line)
        return "c" if line.rstrip().endswith("*/") else "d" # If doxygen ends, return code state else return doxygen state
            
        
    def __parsing_one_line_comment_state(self, line: str) -> str:
        if line.lstrip().startswith("//"):    
            self.current_code_block.append(line)
            return "olc" # Return one line comment state
        else:
            return "c" # Return code state
        
    def __parsing_code_state(self, line: str) -> str:
        ...
        
    def __parsing_invalid_state(self, line: str) -> str:
        ...
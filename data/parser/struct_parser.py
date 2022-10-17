from copy import deepcopy
import logging
from typing import List
from data.parser.parser_exceptions import ParsingStructException

from data.parser.parsing_object import ParsedStruct


class StructParser:

    logger = logging.getLogger('StructParser')
    parsed_struct = ParsedStruct()

    def process(self, lines : List[str]):
        self.logger.debug('Processing struct')
        
        idx = self.__process_comment(lines)
        lines = lines[idx:]
        self.parsed_struct.code = "".join(lines)

        # Reset and return
        parsed_struct = deepcopy(self.parsed_struct)
        self.parsed_struct = ParsedStruct()
        return parsed_struct


    def __process_comment(self, lines : List[str]) -> int:
        """
        Parse the doxygen / comment of the function and fills parsed_struct object
        
        params: 
        lines - lines of function code
        
        return - index of next line after comment
        """

        if len(lines) == 0:
            self.throw_exception(
                "No function content found",
                "No function content found"
            )

        first_line = lines[0].lstrip()
        # Parsing doxygen
        if first_line.startswith('/*'):
            line_ids = range(0, len(lines))
            
            for line_idx in line_ids:
                line = lines[line_idx].rstrip()
                if line.endswith('*/'):
                    self.parsed_struct.comment = "".join(lines[:line_idx+1])
                    return line_idx + 1

            self.throw_exception(
                "Error parsing comment:\n {}\n".format("".join(lines)),
                "Error parsing comment"
            )


        # Parsing one line comments
        elif first_line.startswith('//'):
            line_ids = range(0, len(lines))
            for line_idx in line_ids:
                line = lines[line_idx].lstrip()
                if not line.startswith('//'):
                    self.parsed_struct.comment = "".join(lines[:line_idx+1])
                    return line_idx + 1
            
            self.throw_exception(
                "Error parsing comment:\n {}\n".format("".join(lines)),
                "Error parsing comment"
            )


        # Invalid start token
        else:
            self.throw_exception(
                "Invalid start token in comment:\n {}\n".format("".join(lines)),
                "Invalid start token in comment"
            )

    def throw_exception(self, debugger_error : str, exception_error : str) -> None:
        """
        Throw exception
        
        params: 
        exception_type - exception type
        debugger_error - exception error
        exception_error - exception error
        
        return - None
        """

        self.logger.error(debugger_error)
        raise ParsingStructException(exception_error)

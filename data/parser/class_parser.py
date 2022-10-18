from copy import deepcopy
import logging
from typing import List

from data.parser.parser_exceptions import ParsingClassException
from data.parser.parsing_object import ParsedClass


class ClassParser:

    logger = logging.getLogger('ClassParser')
    parsed_class = ParsedClass()

    def process(self, lines : List[str]):
        self.logger.debug('Processing struct')
        
        idx = self.__process_comment(lines)
        lines = lines[idx:]
        self.parsed_class.code = "\n".join(lines)

        # Reset and return
        parsed_class = deepcopy(self.parsed_class)
        self.parsed_class = ParsedClass()
        return parsed_class


    def __process_comment(self, lines : List[str]) -> int:
        """
        Parse the doxygen / comment of the function and fills parsed_class object
        
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
            for i, line in enumerate(lines[1:], 1):
                line = lines[i].rstrip()
                if line.endswith('*/'):
                    self.parsed_class.comment = "\n".join(lines[:i])
                    return i

            self.throw_exception(
                "Error parsing comment:\n {}\n".format("\n".join(lines)),
                "Error parsing comment"
            )


        # Parsing one line comments
        elif first_line.startswith('//'):
            for i, line in enumerate(lines[1:], 1):
                line = line.lstrip()
                if not line.startswith('//'):
                    self.parsed_class.comment = "\n".join(lines[:i])
                    return i
            
            self.throw_exception(
                "Error parsing comment:\n {}\n".format("\n".join(lines)),
                "Error parsing comment"
            )


        # Invalid start token
        else:
            self.throw_exception(
                "Invalid start token in comment:\n {}\n".format("\n".join(lines)),
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
        raise ParsingClassException(exception_error)

from copy import deepcopy
import logging
from typing import List
from data.parser.parser_exceptions import ParsingFunctionException

from data.parser.parsing_object import ParsedFunction


class FunctionParser:

    logger = logging.getLogger('FunctionParser')
    parsed_function = ParsedFunction()
    gpu_prefixes = ["__global__", "__device__", "__host__", "__constant__"]

    def process(self, lines : List[str]):
        self.logger.debug('Processing function')
        
        idx = self.__process_comment(lines)
        lines = lines[idx:]
        body = self.__process_header(lines)
        if len(body) > 0:
            self.parsed_function.code = body

        # Reset and return
        parsed_function = deepcopy(self.parsed_function)
        self.parsed_function = ParsedFunction()
        return parsed_function


    def __process_comment(self, lines : List[str]) -> int:
        """
        Parse the doxygen / comment of the function and fills parsed_function object
        
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
                    self.parsed_function.comment = "\n".join(lines[:line_idx+1])
                    return line_idx + 1

            self.throw_exception(
                "Error parsing comment:\n {}\n".format("\n".join(lines)),
                "Error parsing comment"
            )


        # Parsing one line comments
        elif first_line.startswith('//'):
            line_ids = range(0, len(lines))
            for line_idx in line_ids:
                line = lines[line_idx].lstrip()
                if not line.startswith('//'):
                    self.parsed_function.comment = "\n".join(lines[:line_idx+1])
                    return line_idx + 1
            
            self.throw_exception(
                "Error parsing comment:\n {}\n".format("\n".join(lines)),
                "Error parsing comment"
            )


        # Invalid start token
        else:
            self.throw_exception(
                "Invalid start token in comment:\n {}\n".format("".join(lines)),
                "Invalid start token in comment"
            )



    def __process_header(self, lines : List[str]) -> str:
        """
        Parse the header of the function and fills parsed_function object
        
        params: 
        lines - lines of function code
        
        return - rest of the content
        """

        if len(lines) == 0:
            self.throw_exception(
                "Error parsing function header:\n {}\n".format("\n".join(lines)),
                "Error parsing function header"
            )

        body = "\n".join(lines)
        body_start_idx = body.find("{")
        
        # No body. Just declaration
        if body_start_idx == -1:
            self.parsed_function.header = body
            self.parsed_function.is_gpu = any((gpu_prefix in body) for gpu_prefix in self.gpu_prefixes)
            return ""
        
        header = body[:body_start_idx]
        # Set the parsed header
        self.parsed_function.header = header
        # Check if function is for GPUs
        self.parsed_function.is_gpu = any((gpu_prefix in header) for gpu_prefix in self.gpu_prefixes)
        
        return body[body_start_idx:]

    
    
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
        raise ParsingFunctionException(exception_error)

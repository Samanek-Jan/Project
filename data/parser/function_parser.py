from copy import deepcopy
import logging
from typing import List
from data.parser.parser_exceptions import ParsingFunctionException
from data.parser.parsing_object import PARSED_FUNCTION_TEMPLATE


class FunctionParser:

    def __init__(self):
        self.logger = logging.getLogger('FunctionParser')
        self.gpu_prefixes = ["__global__", "__device__", "__host__", "__constant__"]

    def process(self, lines : List[str]):
        self.logger.debug('Processing function')
        self.parsed_function = deepcopy(PARSED_FUNCTION_TEMPLATE)
        
        content = "\n".join(lines)
        content = self.__process_comment(content)
        body = self.__process_header(content)
        self.parsed_function["body"] = body
        self.parsed_function["type"] = "function"
        return self.parsed_function


    def __process_comment(self, content : str) -> str:

        if len(content) == 0:
            self.__throw_exception(
                "No function content found",
                "No function content found"
            )

        lines = content.split("\n")
        first_line = lines[0].lstrip()
        char_count = 0

        # Parsing doxygen
        if first_line.startswith('/*'):
            
            for line in lines:
                end_comment = line.find('*/')
                if end_comment > -1:
                    self.parsed_function["comment"] += line.rstrip()
                    char_count += end_comment                    
                    return content[char_count:].strip()
                else:
                    self.parsed_function["comment"] += line.rstrip()
                    char_count += len(line)

            self.__throw_exception(
                "Function does not have any body:\n {}\n".format("\n".join(content)),
                "Function does not have any body"
            )


        # Parsing one line comments
        if first_line.startswith('//'):
            
            for line in lines:
                if not line.lstrip().startswith("//"):
                    self.parsed_function["comment"] += line.strip()
                    return content[char_count:].strip()
                else:
                    self.parsed_function["comment"] += line.strip()

                char_count += len(line)

            self.__throw_exception(
                "Function does not have any body:\n {}\n".format("\n".join(content)),
                "Function does not have any body"
            )
            
        else:
            self.parsed_function["comment"] = ""
            return content.strip()

    def __process_header(self, content : str) -> str:

        if len(content) == 0:
            self.__throw_exception(
                "Parsing function has no header",
                "Parsing function has no header"
            )

        body_start_idx = content.find("{")
        
        # No body. Just declaration
        if body_start_idx == -1:
            self.parsed_function["header"] = content
            self.parsed_function["is_gpu"] = any((gpu_prefix in set(content.split(" "))) for gpu_prefix in self.gpu_prefixes)
            return ""
        
        header = content[:body_start_idx]
        # Set the parsed header
        self.parsed_function["header"] = header
        # Check if function is for GPUs
        self.parsed_function["is_gpu"] = any((gpu_prefix in header) for gpu_prefix in self.gpu_prefixes)
        
        return content[body_start_idx:]

    
    
    def __throw_exception(self, debugger_error : str, exception_error : str) -> None:
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

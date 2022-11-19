from copy import deepcopy
import logging
from typing import List
from data.parser.parser_exceptions import ParsingStructException
from data.parser.parsing_object import MIN_INNER_LEN_PARSING, PARSED_OBJECT_TEMPLATE
import data.parser.parser as parser


class ObjectParser:

    logger = logging.getLogger('ObjectParser')

    def process(self, lines : List[str], is_gpu : bool, filename : str):
        self.parsed_object = deepcopy(PARSED_OBJECT_TEMPLATE) 
        self.logger.debug('Processing struct')
        content = "\n".join(lines)
        
        body = self.__process_comment(content)
        
        self.parsed_object["body"] = body
        self.parsed_object["type"] = "object"
        self.parsed_object["is_gpu"] = is_gpu
        
        if len(body) > MIN_INNER_LEN_PARSING:
            start_bracket = body.find("{")
            end_bracket = body.rfind("}")
            if start_bracket > -1 and end_bracket > start_bracket:
                inner_parser = parser.Parser()
                body = body[start_bracket+1 : end_bracket]
                self.parsed_object["inner_objects"] = inner_parser.process_str(body, filename)
        
        return self.parsed_object


    def __process_comment(self, content : str) -> str:
        """ Parse comment from given content

        Args:
            content (str): code of structure with optional comment

        Returns:
            str: rest of code without comment
        """

        if len(content.strip()) == 0:
            self.__throw_exception(
                "No class content found",
                "No class content found"
            )

        lines = content.split("\n")
        char_count = 0
        comment = []
        
        if lines[0].lstrip().startswith("/*"):
            
            for line in lines:
                end_comment = line.find("*/")
                if end_comment > -1:
                    char_count += end_comment
                    comment.append(line[:end_comment])
                    self.parsed_object["comment"] = "\n".join(comment)
                    return content[char_count:].strip()      
                else:
                    char_count += len(line)
                    comment.append(line.rstrip())
            
        elif lines[0].lstrip().startswith("//"):
            for line in lines:
                if not line.lstrip().startswith("//"):
                    self.parsed_object["comment"] = "\n".join(comment)
                    return content[char_count:].strip()
                else:
                    char_count += len(line)
                    comment.append(line.rstrip())        
        else:
            self.parsed_object["comment"] = ""
            return content.strip()
        
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
        raise ParsingStructException(exception_error)

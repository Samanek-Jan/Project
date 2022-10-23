from copy import deepcopy
import logging
from typing import List
from data.parser.parser_exceptions import ParsingStructException
from data.parser.parsing_object import PARSED_OBJECT_TEMPLATE


class ObjectParser:

    logger = logging.getLogger('ObjectParser')

    def process(self, lines : List[str]):
        self.parsed_object = deepcopy(PARSED_OBJECT_TEMPLATE) 
        self.logger.debug('Processing struct')
        content = "\n".join(lines)
        
        body = self.__process_comment(content)
        
        self.parsed_object["body"] = body
        self.parsed_object["type"] = "object"
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
                    return content[char_count:]           
                else:
                    char_count += len(line)
                    comment.append(line.rstrip())
            
        elif lines[0].lstrip().startswith("//"):
            for line in lines:
                if not line.lstrip().startswith("//"):
                    self.parsed_object["comment"] = "\n".join(comment)
                    return content[char_count:]  
                else:
                    char_count += len(line)
                    comment.append(line.rstrip())        
        else:
            self.parsed_object["comment"] = ""
            return content
        
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

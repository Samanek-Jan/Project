from copy import deepcopy
import logging
from typing import List

from data.parser.parser_exceptions import ParsingClassException
from data.parser.parsing_object import PARSED_CLASS_TEMPLATE


class ClassParser:

    logger = logging.getLogger('ClassParser')
    

    def process(self, lines : List[str]):
        self.logger.debug('Processing struct')
        self.parsed_class = deepcopy(PARSED_CLASS_TEMPLATE)
        content = "\n".join(lines)
        
        body = self.__process_comment(content)
        
        if body.strip() == "":
            return None
        
        self.parsed_class["body"] = body
        self.parsed_class["type"] = "class"
        return self.parsed_class


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
                    self.parsed_class["comment"] = "\n".join(comment)
                    return content[char_count:].strip()         
                else:
                    char_count += len(line)
                    comment.append(line.rstrip())
            
        elif lines[0].lstrip().startswith("//"):
            for line in lines:
                if not line.lstrip().startswith("//"):
                    self.parsed_class["comment"] = "\n".join(comment)
                    return content[char_count:].strip()
                else:
                    char_count += len(line)
                    comment.append(line.rstrip())        
        else:
            self.parsed_class["comment"] = ""
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
        raise ParsingClassException(exception_error)


from enum import Enum

class ParsingTypes(Enum):
    CLASS = "class",
    FUNCTION = "function",
    OBJECT = "object",

PARSING_TYPES = Enum("ParsingTypes", ["CLASS", "FUNCTION", "OBJECT"])

MIN_INNER_LEN_PARSING = 500

PARSED_OBJECT_TEMPLATE = {
    "comment"       : "",
    "body"          : "",
    "type"          : "",
    "is_cuda"        : False,
    "is_from_cuda_file" : False,
}

PARSED_CLASS_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE,
}

PARSED_FUNCTION_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE,
    "header" : ""
}

PARSING_OBJECTS = {
    PARSING_TYPES.CLASS : PARSED_CLASS_TEMPLATE,
    PARSING_TYPES.FUNCTION : PARSED_FUNCTION_TEMPLATE,
    PARSING_TYPES.OBJECT : PARSED_OBJECT_TEMPLATE
}
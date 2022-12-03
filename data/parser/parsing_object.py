
PARSING_TYPES = ["object", "struct", "class", "function"]
MIN_INNER_LEN_PARSING = 500

PARSED_OBJECT_TEMPLATE = {
    "comment"       : "",
    "body"          : "",
    "type"          : "",
    "inner_objects" : [],
    "is_gpu"        : False
}

PARSED_CLASS_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE
}

PARSED_STRUCT_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE
}

PARSED_FUNCTION_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE,
    "header" : ""
}
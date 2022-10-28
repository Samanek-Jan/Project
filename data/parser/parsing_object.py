
PARSING_TYPES = ["object", "struct", "class", "function"]

PARSED_OBJECT_TEMPLATE = {
    "comment" : "",
    "body"    : "",
    "type"    : "",
    "is_gpu"  : False
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
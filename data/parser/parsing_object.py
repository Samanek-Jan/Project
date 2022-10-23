
PARSED_OBJECT_TEMPLATE = {
    "comment" : "",
    "body"    : "",
    "type"    : "" 
}

PARSED_CLASS_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE
}

PARSED_STRUCT_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE
}

PARSED_FUNCTION_TEMPLATE = {
    **PARSED_OBJECT_TEMPLATE,
    "header" : "",
    "is_gpu" : False
}
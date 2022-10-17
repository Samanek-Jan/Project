from typing import Iterable


class ParsedObject:
    comment : str
    code : str

class ParsedFunction(ParsedObject):
    is_gpu : bool
    header : str

class ParsedStruct(ParsedObject):
    ...


class ParsedClass(ParsedObject):
    ...
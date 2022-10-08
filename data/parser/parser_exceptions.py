

class ParsingFunctionException(Exception):

    def __init__(self, message):
        super().__init__(message)

class ParsingStructException(Exception):

    def __init__(self, message):
        super().__init__(message)

class ParsingClassException(Exception):

    def __init__(self, message):
        super().__init__(message)
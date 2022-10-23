

class ParsingFunctionException(Exception):

    def __init__(self, message):
        super().__init__(message)

class ParsingStructException(Exception):

    def __init__(self, message):
        super().__init__(message)

class ParsingClassException(Exception):

    def __init__(self, message):
        super().__init__(message)
        
class InvalidStateException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
        
class BracketCountErrorException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
        
class InvalidTypeException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
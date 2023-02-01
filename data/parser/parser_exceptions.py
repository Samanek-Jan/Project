
class ProcessingObjectException(Exception):
    def __init__(self, message):
        super().__init__(message)
        
class ParsingFunctionException(Exception):

    def __init__(self, message):
        super().__init__(message)

class ParsingStructException(Exception):

    def __init__(self, message):
        super().__init__(message)

class ParsingClassException(Exception):

    def __init__(self, message):
        super().__init__(message)
        
        
class BracketCountErrorException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
        
class InvalidTypeException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
        
class InvalidParameterException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
        
class InvalidCharacterException(Exception):
    
    def __init__(self, message):
        super().__init__(message)
        
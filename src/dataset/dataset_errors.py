
class WrongParameterError(Exception):
    
    def __init__(self, msg):
        super(WrongParameterError, self).__init__(msg)
''' IO implementatations '''

def guessed_imp(filespec):
    return IOImplementation.from_filespec(filespec)

class IOImplementation(object):
    pass

default_io = IOImplementation


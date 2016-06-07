"""
Python 3 compatibility tools.

Copied from numpy/compat/py3k

Please prefer the routines in externals/six.py when possible

BSD license
"""

__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
           'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested',
           'asstr', 'open_latin1', 'StringIO', 'BytesIO']

import sys

if sys.version_info[0] >= 3:
    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    bytes = bytes
    unicode = str
    asunicode = str

    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')

    def asstr(s):
        if isinstance(s, str):
            return s
        return s.decode('latin1')

    def isfileobj(f):
        return isinstance(f, io.FileIO)

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode, encoding='iso-8859-1')
    strchar = 'U'
    ints2bytes = lambda seq: bytes(seq)
    ZEROB = bytes([0])
    FileNotFoundError = FileNotFoundError
else:
    import StringIO
    StringIO = BytesIO = StringIO.StringIO
    bytes = str
    unicode = unicode
    asbytes = str
    asstr = str
    strchar = 'S'

    def isfileobj(f):
        return isinstance(f, file)

    def asunicode(s):
        if isinstance(s, unicode):
            return s
        return s.decode('ascii')

    def open_latin1(filename, mode='r'):
        return open(filename, mode=mode)
    ints2bytes = lambda seq: ''.join(chr(i) for i in seq)
    ZEROB = chr(0)

    class FileNotFoundError(IOError):
        pass


def getexception():
    return sys.exc_info()[1]


def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)


def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)

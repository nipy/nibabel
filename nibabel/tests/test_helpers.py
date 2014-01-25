""" Helper functions for tests
"""

from io import BytesIO
from ..openers import Opener
from ..tmpdirs import InTemporaryDirectory


def bytesio_filemap(klass):
    """ Return bytes io filemap for this image class `klass` """
    file_map = klass.make_file_map()
    for name, fileholder in file_map.items():
        fileholder.fileobj = BytesIO()
        fileholder.pos = 0
    return file_map


def bytesio_round_trip(img):
    """ Save then load image from bytesio
    """
    klass = img.__class__
    bytes_map = bytesio_filemap(klass)
    img.to_file_map(bytes_map)
    return klass.from_file_map(bytes_map)


def bz2_mio_error():
    """ Return True if writing mat 4 file fails

    Writing an empty string can fail for bz2 objects in python 3.3:

    http://bugs.python.org/issue16828

    This in turn causes scipy to give this error when trying to write bz2 mat
    files.

    This won't cause a problem for scipy releases after Jan 24 2014 because of
    commit 98ef522d99 (in scipy)
    """
    try:
        import scipy.io
    except ImportError:
        return True
    with InTemporaryDirectory():
        with Opener('test.mat.bz2', 'wb') as fobj:
            try:
                scipy.io.savemat(fobj, {'a': 1}, format='4')
            except ValueError:
                return True
    return False

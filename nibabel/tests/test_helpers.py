""" Helper functions for tests
"""

from io import BytesIO


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

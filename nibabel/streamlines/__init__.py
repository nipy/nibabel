from nibabel.openers import Opener

from nibabel.streamlines.utils import detect_format


def load(fileobj):
    ''' Load a file of streamlines, return instance associated to file format

    Parameters
    ----------
    fileobj : string or file-like object
       If string, a filename; otherwise an open file-like object
       pointing to a streamlines file (and ready to read from the beginning
       of the streamlines file's header)

    Returns
    -------
    obj : instance of ``StreamlineFile``
       Returns an instance of a ``StreamlineFile`` subclass corresponding to
       the format of the streamlines file ``fileobj``.
    '''
    fileobj = Opener(fileobj)
    streamlines_file = detect_format(fileobj)
    return streamlines_file.load(fileobj)

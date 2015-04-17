from .header import Field
from .base_format import Streamlines, LazyStreamlines

from nibabel.streamlines.trk import TrkFile
#from nibabel.streamlines.tck import TckFile
#from nibabel.streamlines.vtk import VtkFile

# List of all supported formats
FORMATS = {".trk": TrkFile,
           #".tck": TckFile,
           #".vtk": VtkFile,
           }


def is_supported(fileobj):
    ''' Checks if the file-like object if supported by NiBabel.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a streamlines file (and ready to read from the beginning of the
        header)

    Returns
    -------
    is_supported : boolean
    '''
    return detect_format(fileobj) is not None


def detect_format(fileobj):
    ''' Returns the StreamlinesFile object guessed from the file-like object.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a streamlines file (and ready to read from the beginning of the
        header)

    Returns
    -------
    streamlines_file : StreamlinesFile object
        Object that can be used to manage a streamlines file.
        See 'nibabel.streamlines.StreamlinesFile'.
    '''
    for format in FORMATS.values():
        try:
            if format.is_correct_format(fileobj):
                return format

        except IOError:
            pass

    import os
    from ..externals.six import string_types
    if isinstance(fileobj, string_types):
        _, ext = os.path.splitext(fileobj)
        return FORMATS.get(ext, None)

    return None


def load(fileobj, ref, lazy_load=False):
    ''' Loads streamlines from a file-like object in voxel space.

    Parameters
    ----------
    fileobj : string or file-like object
       If string, a filename; otherwise an open file-like object
       pointing to a streamlines file (and ready to read from the beginning
       of the streamlines file's header).

    ref : filename | `Nifti1Image` object | 2D array (4,4)
        Reference space where streamlines will live in `fileobj`.

    lazy_load : boolean (optional)
        Load streamlines in a lazy manner i.e. they will not be kept
        in memory.

    Returns
    -------
    obj : instance of `Streamlines`
       Returns an instance of a `Streamlines` class containing data and metadata
       of streamlines loaded from `fileobj`.
    '''
    streamlines_file = detect_format(fileobj)

    if streamlines_file is None:
        raise TypeError("Unknown format for 'fileobj': {0}!".format(fileobj))

    return streamlines_file.load(fileobj, ref, lazy_load=lazy_load)


def save(streamlines, filename, ref=None):
    ''' Saves a `Streamlines` object to a file

    Parameters
    ----------
    streamlines : `Streamlines` object
       Streamlines to be saved.

    filename : str
       Name of the file where the streamlines will be saved. The format will
       be guessed from `filename`.

    ref : filename | `Nifti1Image` object | 2D array (4,4) (optional)
        Reference space where streamlines will live in `fileobj`.
    '''
    streamlines_file = detect_format(filename)

    if streamlines_file is None:
        raise TypeError("Unknown streamlines file format: '{0}'!".format(filename))

    streamlines_file.save(streamlines, filename, ref)


def convert(in_fileobj, out_filename, ref):
    ''' Converts a streamlines file to another format.

    Parameters
    ----------
    in_fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a streamlines file (and ready to read from the beginning of the
        header).

    out_filename : str
       Name of the file where the streamlines will be saved. The format will
       be guessed from `out_filename`.

    ref : filename | `Nifti1Image` object | 2D array (4,4)
        Reference space where streamlines live in `fileobj`.
    '''
    streamlines = load(in_fileobj, ref, lazy_load=True)
    save(streamlines, out_filename)

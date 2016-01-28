import os
from ..externals.six import string_types

from .header import Field
from .compact_list import CompactList
from .tractogram import Tractogram, LazyTractogram

from .trk import TrkFile
#from .tck import TckFile
#from .vtk import VtkFile

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
        to a tractogram file (and ready to read from the beginning of the
        header)

    Returns
    -------
    tractogram_file : ``TractogramFile`` class
        Returns an instance of a `TractogramFile` class containing data and
        metadata of the tractogram contained from `fileobj`.
    '''
    for format in FORMATS.values():
        try:
            if format.is_correct_format(fileobj):
                return format

        except IOError:
            pass

    if isinstance(fileobj, string_types):
        _, ext = os.path.splitext(fileobj)
        return FORMATS.get(ext.lower())

    return None


def load(fileobj, lazy_load=False, ref=None):
    ''' Loads streamlines from a file-like object in voxel space.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object
        pointing to a streamlines file (and ready to read from the beginning
        of the streamlines file's header).

    lazy_load : boolean (optional)
        Load streamlines in a lazy manner i.e. they will not be kept
        in memory.

    ref : filename | `Nifti1Image` object | 2D array (4,4) (optional)
        Reference space where streamlines will live in `fileobj`.

    Returns
    -------
    tractogram_file : ``TractogramFile``
        Returns an instance of a `TractogramFile` class containing data and
        metadata of the tractogram loaded from `fileobj`.
    '''
    tractogram_file = detect_format(fileobj)

    if tractogram_file is None:
        raise ValueError("Unknown format for 'fileobj': {}".format(fileobj))

    return tractogram_file.load(fileobj, lazy_load=lazy_load)


def save(tractogram_file, filename):
    ''' Saves a tractogram to a file.

    Parameters
    ----------
    tractogram_file : ``TractogramFile`` object
        Tractogram to be saved on disk.

    filename : str
        Name of the file where the tractogram will be saved. The format will
        be guessed from `filename`.
    '''
    tractogram_file.save(filename)


def save_tractogram(tractogram, filename, **kwargs):
    ''' Saves a tractogram to a file.

    Parameters
    ----------
    tractogram : ``Tractogram`` object
        Tractogram to be saved.

    filename : str
        Name of the file where the tractogram will be saved. The format will
        be guessed from `filename`.
    '''
    tractogram_file_class = detect_format(filename)

    if tractogram_file_class is None:
        raise ValueError("Unknown tractogram file format: '{}'".format(filename))

    tractogram_file = tractogram_file_class(tractogram, **kwargs)
    save(tractogram_file, filename)

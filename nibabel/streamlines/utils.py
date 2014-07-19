import os

from ..externals.six import string_types

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

    if isinstance(fileobj, string_types):
        _, ext = os.path.splitext(fileobj)
        return FORMATS.get(ext, None)

    return None


def load(fileobj, lazy_load=True, anat=None):
    ''' Loads streamlines from a file-like object in their native space.

    Parameters
    ----------
    fileobj : string or file-like object
       If string, a filename; otherwise an open file-like object
       pointing to a streamlines file (and ready to read from the beginning
       of the streamlines file's header)

    Returns
    -------
    obj : instance of ``Streamlines``
       Returns an instance of a ``Streamlines`` class containing data and metadata
       of streamlines loaded from ``fileobj``.
    '''

    # TODO: Ask everyone what should be the behavior if the anat is provided.
    # if anat is None:
    #     warnings.warn("WARNING: Streamlines will be loaded in their native space (i.e. as they were saved).")

    streamlines_file = detect_format(fileobj)

    if streamlines_file is None:
        raise TypeError("Unknown format for 'fileobj': {0}!".format(fileobj))

    return streamlines_file.load(fileobj, lazy_load=lazy_load)


def save(streamlines, filename):
    ''' Saves a ``Streamlines`` object to a file

    Parameters
    ----------
    streamlines : Streamlines object
       Streamlines to be saved (metadata is obtained with the function ``get_header`` of ``streamlines``).

    filename : string
       Name of the file where the streamlines will be saved. The format will be guessed from ``filename``.
    '''

    streamlines_file = detect_format(filename)

    if streamlines_file is None:
        raise TypeError("Unknown format for 'filename': {0}!".format(filename))

    streamlines_file.save(streamlines, filename)


def convert(in_fileobj, out_filename):
    ''' Converts one streamlines format to another.

    It does not change the space in which the streamlines are.

    Parameters
    ----------
    in_fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a streamlines file (and ready to read from the beginning of the
        header)

    out_filename : string
       Name of the file where the streamlines will be saved. The format will
       be guessed from ``out_filename``.
    '''
    streamlines = load(in_fileobj, lazy_load=True)
    save(streamlines, out_filename)


# TODO
def change_space(streamline_file, new_point_space):
    pass

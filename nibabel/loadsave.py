# module imports
import os

from nibabel.filename_parser import types_filenames
from nibabel import volumeutils as vu
from nibabel import spm2analyze as spm2
from nibabel import nifti1
from nibabel import minc
from nibabel.spatialimages import ImageFileError


def load(filename, *args, **kwargs):
    ''' Load file given filename, guessing at file type

    Parameters
    ----------
    filename : string
       specification of file to load
    *args
    **kwargs
       arguments to pass to image load function

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type

    '''
    fname = filename
    for ending in ('.gz', '.bz2'):
        if filename.endswith(ending):
            fname = fname[:-len(ending)]
            break
    froot, ext = os.path.splitext(fname)
    if ext == '.nii':
        return nifti1.load(filename, *args, **kwargs)
    if ext == '.mnc':
        return minc.load(filename, *args, **kwargs)
    if not ext in ('.img', '.hdr'):
        raise RuntimeError('Cannot work out file type of "%s"' %
                           filename)
    # might be nifti pair or analyze of some sort
    files_types = (('image','.img'), ('header','.hdr'))
    filenames = types_filenames(filename, files_types)
    hdr = nifti1.Nifti1Header.from_fileobj(
        vu.allopen(filenames['header']),
        check=False)
    magic = hdr['magic']
    if magic in ('ni1', 'n+1'):
        return nifti1.load(filename, *args, **kwargs)
    return spm2.load(filename, *args, **kwargs)


def save(img, filename):
    ''' Save an image to file without changing format

    Parameters
    ----------
    img : ``SpatialImage``
       image to save
    filename : str
       filename (often implying filenames) to which to save `img`.

    Returns
    -------
    None
    '''
    try:
        img.to_filename(filename)
    except ImageFileError:
        pass
    else:
        return
    

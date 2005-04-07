# module imports
import os

from nibabel.filename_parser import types_filenames, splitext_addext
from nibabel import volumeutils as vu
from nibabel import spm2analyze as spm2
from nibabel import nifti1
from nibabel import minc
from nibabel.spatialimages import ImageFileError
from nibabel.imageclasses import class_map, ext_map


def load(filename):
    ''' Load file given filename, guessing at file type

    Parameters
    ----------
    filename : string
       specification of file to load

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type

    '''
    froot, ext, trailing = splitext_addext(filename, ('.gz', '.bz2'))
    try:
        img_type = ext_map[ext]
    except KeyError:
        raise ImageFileError('Cannot work out file type of "%s"' %
                             filename)
    if ext in ('.nii', '.mnc'):
        klass = class_map[img_type]['class']
        return klass.from_filename(filename)
    # might be nifti pair or analyze of some sort
    files_types = (('image','.img'), ('header','.hdr'))
    filenames = types_filenames(filename, files_types)
    hdr = nifti1.Nifti1Header.from_fileobj(
        vu.allopen(filenames['header']),
        check=False)
    magic = hdr['magic']
    if magic == 'ni1':
        return nifti1.Nifti1Pair.from_filename(filename)
    elif magic == 'n+1':
        return nifti1.Nifti1Image.from_filename(filename)
    return spm2.Spm2AnalyzeImage.from_filename(filename)


def save(img, filename):
    ''' Save an image to file adapting format to `filename`

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
    froot, ext, trailing = splitext_addext(filename, ('.gz', '.bz2'))
    img_type = ext_map[ext]
    klass = class_map[img_type]['class']
    converted = klass.from_image(img)
    converted.to_filename(filename)

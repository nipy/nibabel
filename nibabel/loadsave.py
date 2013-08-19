# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# module imports
import numpy as np

from .filename_parser import types_filenames, splitext_addext
from .volumeutils import BinOpener, Opener
from . import spm2analyze as spm2
from . import nifti1
from .minc1 import Minc1Image
from .minc2 import Minc2Image
from .freesurfer import MGHImage
from .fileholders import FileHolderError
from .spatialimages import ImageFileError
from .imageclasses import class_map, ext_map


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
    if ext in ('.nii', '.mgh', '.mgz'):
        klass = class_map[img_type]['class']
    elif ext == '.mnc':
        # Look for HDF5 signature for MINC2
        # http://www.hdfgroup.org/HDF5/doc/H5.format.html
        with Opener(filename) as fobj:
            signature = fobj.read(4)
            klass = Minc2Image if signature == b'\211HDF' else Minc1Image
    else:
        # might be nifti pair or analyze of some sort
        files_types = (('image','.img'), ('header','.hdr'))
        filenames = types_filenames(filename, files_types)
        with BinOpener(filenames['header']) as fobj:
            hdr = nifti1.Nifti1Header.from_fileobj(fobj, check=False)
        if hdr['magic'] in (b'ni1', b'n+1'):
            # allow goofy nifti single magic for pair
            klass = nifti1.Nifti1Pair
        else:
            klass =  spm2.Spm2AnalyzeImage
    return klass.from_filename(filename)


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


def read_img_data(img, prefer='scaled'):
    """ Read data from image associated with files

    Parameters
    ----------
    img : ``SpatialImage``
       Image with valid image file in ``img.file_map``.  Unlike the
       ``img.get_data()`` method, this function returns the data read
       from the image file, as specified by the *current* image header
       and *current* image files. 
    prefer : str, optional
       Can be 'scaled' - in which case we return the data with the
       scaling suggested by the format, or 'unscaled', in which case we
       return, if we can, the raw data from the image file, without the
       scaling applied.

    Returns
    -------
    arr : ndarray
       array as read from file, given parameters in header

    Notes
    -----
    Summary: please use the ``get_data`` method of `img` instead of this
    function unless you are sure what you are doing.

    In general, you will probably prefer ``prefer='scaled'``, because
    this gives the data as the image format expects to return it. 

    Use `prefer` == 'unscaled' with care; the modified Analyze-type
    formats such as SPM formats, and nifti1, specify that the image data
    array is given by the raw data on disk, multiplied by a scalefactor
    and maybe with the addition of a constant.  This function, with
    ``unscaled`` returns the data on the disk, without these
    format-specific scalings applied.  Please use this funciton only if
    you absolutely need the unscaled data, and the magnitude of the
    data, as given by the scalefactor, is not relevant to your
    application.  The Analyze-type formats have a single scalefactor +/-
    offset per image on disk. If you do not care about the absolute
    values, and will be removing the mean from the data, then the
    unscaled values will have preserved intensity ratios compared to the
    mean-centered scaled data.  However, this is not necessarily true of
    other formats with more complicated scaling - such as MINC.
    """
    if prefer not in ('scaled', 'unscaled'):
        raise ValueError('Invalid string "%s" for "prefer"' % prefer)
    image_fileholder = img.file_map['image']
    hdr = img.get_header()
    try:
        fileobj = image_fileholder.get_prepare_fileobj()
    except FileHolderError:
        raise ImageFileError('No image file specified for this image')
    with fileobj:
        if prefer == 'unscaled':
            try:
                return hdr.raw_data_from_fileobj(fileobj)
            except AttributeError:
                pass
        return hdr.data_from_fileobj(fileobj)


def which_analyze_type(binaryblock):
    """ Is `binaryblock` from NIfTI1, NIfTI2 or Analyze header?

    Parameters
    ----------
    binaryblock : bytes
        The `binaryblock` is 348 bytes that might be NIfTI1, NIfTI2, Analyze, or
        None of the the above.

    Returns
    -------
    hdr_type : str
        * a nifti1 header (pair or single) -> return 'nifti1'
        * a nifti2 header (pair or single) -> return 'nifti2'
        * an Analyze header -> return 'analyze'
        * None of the above -> return None

    Notes
    -----
    Algorithm:

    * read in the first 4 bytes from the file as 32-bit int ``sizeof_hdr``
    * if ``sizeof_hdr`` is 540 or byteswapped 540 -> assume nifti2
    * Check for 'ni1', 'n+1' magic -> assume nifti1
    * if ``sizeof_hdr`` is 348 or byteswapped 348 assume Analyze
    * Return None
    """
    hdr = np.ndarray(shape=(), dtype=nifti1.header_dtype, buffer=binaryblock)
    bs_hdr = hdr.byteswap()
    sizeof_hdr = hdr['sizeof_hdr']
    bs_sizeof_hdr = bs_hdr['sizeof_hdr']
    if 540 in (sizeof_hdr, bs_sizeof_hdr):
        return 'nifti2'
    if hdr['magic'] in (b'ni1', b'n+1'):
        return 'nifti1'
    if 348 in (sizeof_hdr, bs_sizeof_hdr):
        return 'analyze'
    return None

# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# module imports
""" Utilities to load and save image objects """

import numpy as np

from .filename_parser import types_filenames, splitext_addext
from .volumeutils import BinOpener, Opener
from .analyze import AnalyzeImage
from .spm2analyze import Spm2AnalyzeImage
from .nifti1 import Nifti1Image, Nifti1Pair, header_dtype as ni1_hdr_dtype
from .nifti2 import Nifti2Image, Nifti2Pair
from .minc1 import Minc1Image
from .minc2 import Minc2Image
from .freesurfer import MGHImage
from .fileholders import FileHolderError
from .spatialimages import ImageFileError
from .imageclasses import class_map, ext_map
from .arrayproxy import is_proxy


def load(filename, **kwargs):
    ''' Load file given filename, guessing at file type

    Parameters
    ----------
    filename : string
       specification of file to load
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type
    '''
    return guessed_image_type(filename).from_filename(filename, **kwargs)


def guessed_image_type(filename):
    """ Guess image type from file `filename`

    Parameters
    ----------
    filename : str
        File name containing an image

    Returns
    -------
    image_class : class
        Class corresponding to guessed image type
    """
    froot, ext, trailing = splitext_addext(filename, ('.gz', '.bz2'))
    lext = ext.lower()
    try:
        img_type = ext_map[lext]
    except KeyError:
        raise ImageFileError('Cannot work out file type of "%s"' %
                             filename)
    if lext in ('.mgh', '.mgz', '.par'):
        klass = class_map[img_type]['class']
    elif lext == '.mnc':
        # Look for HDF5 signature for MINC2
        # https://www.hdfgroup.org/HDF5/doc/H5.format.html
        with Opener(filename) as fobj:
            signature = fobj.read(4)
            klass = Minc2Image if signature == b'\211HDF' else Minc1Image
    elif lext == '.nii':
        with BinOpener(filename) as fobj:
            binaryblock = fobj.read(348)
        ft = which_analyze_type(binaryblock)
        klass = Nifti2Image if ft == 'nifti2' else Nifti1Image
    else: # might be nifti 1 or 2 pair or analyze of some sort
        files_types = (('image','.img'), ('header','.hdr'))
        filenames = types_filenames(filename, files_types)
        with BinOpener(filenames['header']) as fobj:
            binaryblock = fobj.read(348)
        ft = which_analyze_type(binaryblock)
        if ft == 'nifti2':
            klass = Nifti2Pair
        elif ft == 'nifti1':
            klass = Nifti1Pair
        else:
            klass = Spm2AnalyzeImage
    return klass


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
    # Special-case Nifti singles and Pairs
    if type(img) == Nifti1Image and ext in ('.img', '.hdr'):
        klass = Nifti1Pair
    elif type(img) == Nifti2Image and ext in ('.img', '.hdr'):
        klass = Nifti2Pair
    elif type(img) == Nifti1Pair and ext == '.nii':
        klass = Nifti1Image
    elif type(img) == Nifti2Pair and ext == '.nii':
        klass = Nifti2Image
    else:
        img_type = ext_map[ext]
        klass = class_map[img_type]['class']
    converted = klass.from_image(img)
    converted.to_filename(filename)


@np.deprecate_with_doc('Please use ``img.dataobj.get_unscaled()`` '
                       'instead')
def read_img_data(img, prefer='scaled'):
    """ Read data from image associated with files

    We've deprecated this function and will remove it soon. If you want
    unscaled data, please use ``img.dataobj.get_unscaled()`` instead.  If you
    want scaled data, use ``img.get_data()`` (which will cache the loaded
    array) or ``np.array(img.dataobj)`` (which won't cache the array). If you
    want to load the data as for a modified header, save the image with the
    modified header, and reload.

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
    hdr = img.header
    if not hasattr(hdr, 'raw_data_from_fileobj'):
        # We can only do scaled
        if prefer == 'unscaled':
            raise ValueError("Can only do unscaled for Analyze types")
        return np.array(img.dataobj)
    # Analyze types
    img_fh = img.file_map['image']
    img_file_like = (img_fh.filename if img_fh.fileobj is None
                     else img_fh.fileobj)
    if img_file_like is None:
        raise ImageFileError('No image file specified for this image')
    # Check the consumable values in the header
    hdr = img.header
    dao = img.dataobj
    default_offset = hdr.get_data_offset() == 0
    default_scaling = hdr.get_slope_inter() == (None, None)
    # If we have a proxy object and the header has any consumed fields, we load
    # the consumed values back from the proxy
    if is_proxy(dao) and (default_offset or default_scaling):
        hdr = hdr.copy()
        if default_offset and dao.offset != 0:
            hdr.set_data_offset(dao.offset)
        if default_scaling and (dao.slope, dao.inter) != (1, 0):
            hdr.set_slope_inter(dao.slope, dao.inter)
    with BinOpener(img_file_like) as fileobj:
        if prefer == 'scaled':
            return hdr.data_from_fileobj(fileobj)
        return hdr.raw_data_from_fileobj(fileobj)


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
    hdr = np.ndarray(shape=(), dtype=ni1_hdr_dtype, buffer=binaryblock)
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

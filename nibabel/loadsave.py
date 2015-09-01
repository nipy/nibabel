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

from .filename_parser import splitext_addext
from .openers import ImageOpener
from .spatialimages import ImageFileError
from .imageclasses import all_image_classes
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

    sniff = None
    for image_klass in all_image_classes:
        is_valid, sniff = image_klass.is_image(filename, sniff)
        if is_valid:
            return image_klass.from_filename(filename, **kwargs)

    raise ImageFileError('Cannot work out file type of "%s"' %
                         filename)


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

    # Save the type as expected
    try:
        img.to_filename(filename)
    except ImageFileError:
        pass
    else:
        return

    # Be nice to users by making common implicit conversions
    froot, ext, trailing = splitext_addext(filename, img._compressed_exts)
    lext = ext.lower()

    # Special-case Nifti singles and Pairs
    from .nifti1 import Nifti1Image, Nifti1Pair  # Inline imports, as this module
    from .nifti2 import Nifti2Image, Nifti2Pair  # really shouldn't reference any image type
    if type(img) == Nifti1Image and lext in ('.img', '.hdr'):
        klass = Nifti1Pair
    elif type(img) == Nifti2Image and lext in ('.img', '.hdr'):
        klass = Nifti2Pair
    elif type(img) == Nifti1Pair and lext == '.nii':
        klass = Nifti1Image
    elif type(img) == Nifti2Pair and lext == '.nii':
        klass = Nifti2Image
    else:  # arbitrary conversion
        valid_klasses = [klass for klass in all_image_classes
                         if klass.is_valid_extension(ext)]
        try:
            klass = valid_klasses[0]
        except IndexError:  # if list is empty
            raise ImageFileError('Cannot work out file type of "%s"' %
                                 filename)
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
    with ImageOpener(img_file_like) as fileobj:
        if prefer == 'scaled':
            return hdr.data_from_fileobj(fileobj)
        return hdr.raw_data_from_fileobj(fileobj)

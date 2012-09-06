
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Processor functions for images '''
import numpy as np

from .orientations import (io_orientation, inv_ornt_aff, flip_axis,
                           apply_orientation, OrientationError)
from .loadsave import load


def squeeze_image(img):
    ''' Return image, remove axes length 1 at end of image shape

    For example, an image may have shape (10,20,30,1,1).  In this case
    squeeze will result in an image with shape (10,20,30).  See doctests
    for further description of behavior.

    Parameters
    ----------
    img : ``SpatialImage``

    Returns
    -------
    squeezed_img : ``SpatialImage``
       Copy of img, such that data, and data shape have been squeezed,
       for dimensions > 3rd, and at the end of the shape list

    Examples
    --------
    >>> import nibabel as nf
    >>> shape = (10,20,30,1,1)
    >>> data = np.arange(np.prod(shape)).reshape(shape)
    >>> affine = np.eye(4)
    >>> img = nf.Nifti1Image(data, affine)
    >>> img.shape == (10, 20, 30, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 20, 30)
    True

    If the data are 3D then last dimensions of 1 are ignored

    >>> shape = (10,1,1)
    >>> data = np.arange(np.prod(shape)).reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (10, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 1, 1)
    True

    Only *final* dimensions of 1 are squeezed

    >>> shape = (1, 1, 5, 1, 2, 1, 1)
    >>> data = data.reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (1, 1, 5, 1, 2, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (1, 1, 5, 1, 2)
    True
    '''
    klass = img.__class__
    shape = img.shape
    slen = len(shape)
    if slen < 4:
        return klass.from_image(img)
    for bdim in shape[3::][::-1]:
        if bdim == 1:
            slen -= 1
        else:
            break
    if slen == len(shape):
        return klass.from_image(img)
    shape = shape[:slen]
    data = img.get_data()
    data = data.reshape(shape)
    return klass(data,
                 img.get_affine(),
                 img.get_header(),
                 img.extra)


def concat_images(images, check_affines=True):
    ''' Concatenate images in list to single image, along last dimension

    Parameters
    ----------
    images : sequence
       sequence of ``SpatialImage`` or of filenames\s
    check_affines : {True, False}, optional
       If True, then check that all the affines for `images` are nearly
       the same, raising a ``ValueError`` otherwise.  Default is True

    Returns
    -------
    concat_img : ``SpatialImage``
       New image resulting from concatenating `images` across last
       dimension
    '''
    n_imgs = len(images)
    img0 = images[0]
    is_filename = False
    if not hasattr(img0, 'get_data'):
        img0 = load(img0)
        is_filename = True
    i0shape = img0.shape
    affine = img0.get_affine()
    header = img0.get_header()
    out_shape = (n_imgs, ) + i0shape
    out_data = np.empty(out_shape)
    for i, img in enumerate(images):
        if is_filename:
            img = load(img)
        if check_affines:
            if not np.all(img.get_affine() == affine):
                raise ValueError('Affines do not match')
        out_data[i] = img.get_data()
    out_data = np.rollaxis(out_data, 0, len(i0shape)+1)
    klass = img0.__class__
    return klass(out_data, affine, header)


def four_to_three(img):
    ''' Create 3D images from 4D image by slicing over last axis

    Parameters
    ----------
    img :  image
       4D image instance of some class with methods ``get_data``,
       ``get_header`` and ``get_affine``, and a class constructor
       allowing Klass(data, affine, header)

    Returns
    -------
    imgs : list
       list of 3D images
    '''
    arr = img.get_data()
    header = img.get_header()
    affine = img.get_affine()
    image_maker = img.__class__
    if arr.ndim != 4:
        raise ValueError('Expecting four dimensions')
    imgs = []
    for i in range(arr.shape[3]):
        arr3d = arr[..., i]
        img3d = image_maker(arr3d, affine, header)
        imgs.append(img3d)
    return imgs


def as_closest_canonical(img, enforce_diag=False):
    ''' Return `img` with data reordered to be closest to canonical

    Canonical order is the ordering of the output axes.

    Parameters
    ----------
    img : ``spatialimage``
    enforce_diag : {False, True}, optional
       If True, before transforming image, check if the resulting image
       affine will be close to diagonal, and if not, raise an error

    Returns
    -------
    canonical_img : ``spatialimage``
       Version of `img` where the underlying array may have been
       reordered and / or flipped so that axes 0,1,2 are those axes in
       the input data that are, respectively, closest to the output axis
       orientation.  We modify the affine accordingly.  If `img` is
       already has the correct data ordering, we just return `img`
       unmodified.
    '''
    aff = img.get_affine()
    ornt = io_orientation(aff)
    if np.all(ornt == [[0, 1],
                       [1,1],
                       [2,1]]): # canonical already
        # however, the affine may not be diagonal
        if enforce_diag and not _aff_is_diag(aff):
            raise OrientationError('Transformed affine is not diagonal')
        return img
    shape = img.shape
    t_aff = inv_ornt_aff(ornt, shape)
    out_aff = np.dot(aff, t_aff)
    # check if we are going to end up with something diagonal
    if enforce_diag and not _aff_is_diag(aff):
        raise OrientationError('Transformed affine is not diagonal')
    # we need to transform the data
    arr = img.get_data()
    t_arr = apply_orientation(arr, ornt)
    return img.__class__(t_arr, out_aff, img.get_header())


def _aff_is_diag(aff):
    ''' Utility function returning True if affine is nearly diagonal '''
    rzs_aff = aff[:3, :3]
    return np.allclose(rzs_aff, np.diag(np.diag(rzs_aff)))


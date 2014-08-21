# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Routines to work with spaces

A space is defined by coordinate axes.

A voxel space can be expressed by a shape implying an array, where the axes are
the axes of the array.
"""

from itertools import product

import numpy as np

from .affines import apply_affine


def vox2out_vox(in_shape, in_affine, voxel_sizes=None):
    """ output-aligned shape, affine for input voxels implied by `in_shape`

    The input (voxel) space is given by `in_shape`

    The mapping between input space and output space is `in_affine`

    The output space is implied by the affine, we don't need to know what that
    is, we just return something with the same (implied) output space.

    Our job is to work out another voxel space where the voxel array axes and
    the output axes are aligned (top left 3 x 3 of affine is diagonal with all
    positive entries) and which contains all the voxels of the implied input
    image at their correct output space positions, once resampled into the
    output voxel space.

    Parameters
    ----------
    in_shape : sequence
        shape of implied input image voxel block. Up to length 3.
    in_affine : (4, 4) array-like
        affine mapping voxel coordinates in `in_shape` to output coordinates.
    voxel_sizes : None or sequence
        Gives the diagonal entries of `output_affine` (except the trailing 1
        for the homogenous coordinates) (``output_affine == np.diag(voxel_sizes
        + [1])``). If None, return identity `output_affine`.

    Returns
    -------
    output_shape : sequence
        Shape of output image that has voxel axes aligned to original image
        output space axes, and encloses all the voxel data from the original
        image implied by `in_shape`.
    output_affine : (4, 4) array
        Affine of output image that has voxel axes aligned to the output axes
        implied by `in_affine`. Top-left 3 x 3 part of affine is diagonal with
        all positive entries.  The entries come from `voxel_sizes` if
        specified, or are all 1.  If the image is < 3D, then the missing
        dimensions will have a 1 in the matching diagonal.
    """
    n_axes = len(in_shape)
    if n_axes > 3:
        raise ValueError('This function can only deal with 3D images')
    if n_axes < 3:
        in_shape += (1,) * (3 - n_axes)
    out_vox = np.ones((3,))
    if not voxel_sizes is None:
        if not len(voxel_sizes) == n_axes:
            raise ValueError('voxel sizes length should match shape')
        if not np.all(np.array(voxel_sizes) > 0):
            raise ValueError('voxel sizes should all be positive')
        out_vox[:n_axes] = voxel_sizes
    in_mn_mx = zip([0, 0, 0], np.array(in_shape) - 1)
    in_corners = list(product(*in_mn_mx))
    out_corners = apply_affine(in_affine, in_corners)
    out_mn = out_corners.min(axis=0)
    out_mx = out_corners.max(axis=0)
    out_shape = np.ceil((out_mx - out_mn) / out_vox) + 1
    out_affine = np.diag(list(out_vox) + [1])
    out_affine[:3, 3] = out_mn
    return tuple(int(i) for i in out_shape[:n_axes]), out_affine


def slice2volume(index, axis, shape=None):
    """ Affine expressing selection of a single slice from 3D volume

    Imagine we have taken a slice from an image data array, ``s = data[:, :,
    index]``.  This function returns the affine to map the array coordinates of
    ``s`` to the array coordinates of ``data``.

    This can be useful for resampling a single slice from a volume.  For
    example, to resample slice ``k`` in the space of ``img1`` from the matching
    spatial voxel values in ``img2``, you might do something like::

        slice_shape = img1.shape[:2]
        slice_aff = slice2volume(k, 2)
        whole_aff = np.linalg.inv(img2.affine).dot(img1.affine.dot(slice_aff))

    and then use ``whole_aff`` in ``scipy.ndimage.affine_transform``:

        rzs, trans = to_matvec(whole_aff)
        data = img2.get_data()
        new_slice = scipy.ndimage.affine_transform(data, rzs, trans, slice_shape)

    Parameters
    ----------
    index : int
        index of selected slice
    axis : {0, 1, 2}
        axis to which `index` applies

    Returns
    -------
    slice_aff : shape (4, 3) affine
        Affine relating input coordinates in a slice to output coordinates in
        the embedded volume
    """
    if index < 0:
        raise ValueError("Cannot handle negative index")
    if not 0 <= axis <= 2:
        raise ValueError("Axis should be between 0 and 2")
    axes = list(range(4))
    axes.remove(axis)
    slice_aff = np.eye(4)[:, axes]
    slice_aff[axis, -1] = index
    return slice_aff

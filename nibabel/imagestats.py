# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Functions for computing image statistics
"""

import numpy as np


def mask_volume(img, units='mm3'):
    """ Compute volume of mask image.
    Equivalent to "fslstats /path/file.nii -V"

    Parameters
    ----------
    img : ``SpatialImage``
        All voxels of the mask should be of value 1, background should have value 0.

    units : string {"mm3", "vox"}, optional
        Unit of the returned mask volume. Defaults to "mm3".

    Returns
    -------
    mask_volume_vx: float
        Volume of mask expressed in voxels.

    or

    mask_volume_mm3: float
        Volume of mask expressed in mm3.

    Examples
    --------
    >>> import nibabel as nf
    >>> path = 'path/to/nifti/mask.nii'
    >>> img = nf.load(path) # path is contains a path to an example nifti mask
    >>> mask_volume(img)
    50.3021
    """
    header = img.header
    _, vx, vy, vz, _, _, _, _ = header['pixdim']
    voxel_volume_mm3 = vx * vy * vz
    mask = img.get_fdata()
    mask_volume_vx = np.sum(mask)
    mask_volume_mm3 = mask_volume_vx * voxel_volume_mm3

    if units == 'vox':
        return mask_volume_vx
    elif units == 'mm3':
        return mask_volume_mm3
    else:
        raise ValueError(f'{units} is not a valid unit. Choose "mm3" or "vox".')

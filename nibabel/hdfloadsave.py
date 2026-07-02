# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import h5py

from .nifti1 import Nifti1Image, Nifti1Header

#from .spm2analyze import Spm2AnalyzeImage
#from .nifti1 import Nifti1Image, Nifti1Pair, Nifti1Header, header_dtype as ni1_hdr_dtype
#from .nifti2 import Nifti2Image, Nifti2Pair
#from .minc1 import Minc1Image
#from .minc2 import Minc2Image
#from .freesurfer import MGHImage

def nifti1img_to_hdf(fname, spatial_img, h5path='/img', append=True):
    """
    Saves a Nifti1Image into an HDF5 file.

    fname: string
    Output HDF5 file path

    spatial_img: nibabel SpatialImage
    Image to be saved

    h5path: string
    HDF5 group path where the image data will be saved.
    Datasets will be created inside the given group path:
    'data', 'extra', 'affine', the header information will
    be set as attributes of the 'data' dataset.

    append: bool
    True if you don't want to erase the content of the file
    if it already exists, False otherwise.

    @note:
    HDF5 open modes
    >>> 'r' Readonly, file must exist
    >>> 'r+' Read/write, file must exist
    >>> 'w' Create file, truncate if exists
    >>> 'w-' Create file, fail if exists
    >>> 'a' Read/write if exists, create otherwise (default)
    """
    mode = 'w'
    if append:
        mode = 'a'

    with h5py.File(fname, mode) as f:
        h5img = f.create_group(h5path)
        h5img['data']   = spatial_img.get_data()
        h5img['extra']  = spatial_img.get_extra()
        h5img['affine'] = spatial_img.get_affine()

        hdr = spatial_img.get_header()
        for k in hdr.keys():
            h5img['data'].attrs[k] = hdr[k]


def hdfgroup_to_nifti1image(fname, h5path):
    """
    Returns a nibabel Nifti1Image from an HDF5 group datasets

    @param fname: string
    HDF5 file path

    @param h5path:
    HDF5 group path in fname

    @return: nibabel Nifti1Image
    """
    with h5py.File(fname, 'r') as f:
        h5img  = f[h5path]
        data   = h5img['data'][()]
        extra  = h5img['extra'][()]
        affine = h5img['affine'][()]

        header = get_nifti1hdr_from_h5attrs(h5img['data'].attrs)

    img = Nifti1Image(data, affine, header=header, extra=extra)

    return img


def get_nifti1hdr_from_h5attrs(h5attrs):
    """
    Transforms an H5py Attributes set to a dict.
    Converts unicode string keys into standard strings
    and each value into a numpy array.

    @param h5attrs: H5py Attributes

    @return: dict
    """
    hdr = Nifti1Header()
    for k in h5attrs.keys():
        hdr[str(k)] = np.array(h5attrs[k])

    return hdr
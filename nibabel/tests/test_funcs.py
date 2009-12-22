''' Test for image funcs '''

from StringIO import StringIO

import numpy as np

import nibabel as nf

from nibabel.funcs import concat_images

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_equal, assert_raises

def test_concat():
    shape = (1,2,5)
    data0 = np.arange(10).reshape(shape)
    affine = np.eye(4)
    img0 = nf.Nifti1Image(data0, affine)
    data1 = data0 - 10
    img1 = nf.Nifti1Image(data1, affine)
    all_imgs = concat_images([img0, img1])
    all_data = np.concatenate(
        [data0[:,:,:,np.newaxis],data1[:,:,:,np.newaxis]],3)
    yield assert_array_equal, all_imgs.get_data(), all_data
    yield assert_array_equal, all_imgs.get_affine(), affine
    img2 = nf.Nifti1Image(data1, affine+1)
    yield assert_raises, ValueError, concat_images, [img0, img2]
    img2 = nf.Nifti1Image(data1.T, affine)
    yield assert_raises, ValueError, concat_images, [img0, img2]

    
    

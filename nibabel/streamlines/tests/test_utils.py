import os
import numpy as np
import nibabel as nib

from nibabel.testing import data_path
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_raises, assert_true

from ..utils import pop, get_affine_from_reference


def test_peek():
    gen = (i for i in range(3))
    assert_equal(pop(gen), 0)
    assert_equal(pop(gen), 1)
    assert_equal(pop(gen), 2)
    assert_true(pop(gen) is None)


def test_get_affine_from_reference():
    filename = os.path.join(data_path, 'example_nifti2.nii.gz')
    img = nib.load(filename)
    affine = img.affine

    # Get affine from an numpy array.
    assert_array_equal(get_affine_from_reference(affine), affine)
    wrong_ref = np.array([[1, 2, 3], [4, 5, 6]])
    assert_raises(ValueError, get_affine_from_reference, wrong_ref)

    # Get affine from a `SpatialImage`.
    assert_array_equal(get_affine_from_reference(img), affine)

    # Get affine from a `SpatialImage` using by its filename.
    assert_array_equal(get_affine_from_reference(filename), affine)

import os
import unittest
import tempfile
import numpy as np
import nibabel as nib

from nibabel.testing import data_path
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_raises, assert_true

from ..base_format import CompactList
from ..utils import pop, get_affine_from_reference
from ..utils import save_compact_list, load_compact_list


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


def test_save_and_load_compact_list():

    with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
        clist = CompactList()
        save_compact_list(f, clist)
        f.seek(0, os.SEEK_SET)
        loaded_clist = load_compact_list(f)
        assert_array_equal(loaded_clist._data, clist._data)
        assert_array_equal(loaded_clist._offsets, clist._offsets)
        assert_array_equal(loaded_clist._lengths, clist._lengths)

    with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
        rng = np.random.RandomState(42)
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        clist = CompactList(data)
        save_compact_list(f, clist)
        f.seek(0, os.SEEK_SET)
        loaded_clist = load_compact_list(f)
        assert_array_equal(loaded_clist._data, clist._data)
        assert_array_equal(loaded_clist._offsets, clist._offsets)
        assert_array_equal(loaded_clist._lengths, clist._lengths)

""" Tests for spaces module
"""

import numpy as np
import numpy.linalg as npl

from ..spaces import vox2out_vox, slice2volume
from ..affines import apply_affine
from ..eulerangles import euler2mat


from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)



def assert_all_in(in_shape, in_affine, out_shape, out_affine):
    slices = tuple(slice(N) for N in in_shape)
    n_axes = len(in_shape)
    in_grid = np.mgrid[slices]
    in_grid = np.rollaxis(in_grid, 0, n_axes + 1)
    v2v = npl.inv(out_affine).dot(in_affine)
    if n_axes < 3: # reduced dimensions case
        new_v2v = np.eye(n_axes + 1)
        new_v2v[:n_axes, :n_axes] = v2v[:n_axes, :n_axes]
        new_v2v[:n_axes, -1] = v2v[:n_axes, -1]
        v2v = new_v2v
    out_grid = apply_affine(v2v, in_grid)
    TINY = 1e-12
    assert_true(np.all(out_grid > -TINY))
    assert_true(np.all(out_grid < np.array(out_shape) + TINY))


def test_vox2out_vox():
    # Test world space bounding box
    shape, aff = vox2out_vox((2, 3, 4), np.eye(4))
    assert_array_equal(shape, (2, 3, 4))
    assert_true(isinstance(shape, tuple))
    assert_true(isinstance(shape[0], int))
    assert_array_equal(aff, np.eye(4))
    assert_all_in((2, 3, 4), np.eye(4), shape, aff)
    shape, aff = vox2out_vox((2, 3, 4), np.diag([-1, 1, 1, 1]))
    assert_array_equal(shape, (2, 3, 4))
    assert_array_equal(aff, [[1, 0, 0, -1], # axis reversed -> -ve offset
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    assert_all_in((2, 3, 4), np.diag([-1, 1, 1, 1]), shape, aff)
    # zooms for affine > 1 -> larger grid with default 1mm output voxels
    shape, aff = vox2out_vox((2, 3, 4), np.diag([4, 5, 6, 1]))
    assert_array_equal(shape, (5, 11, 19))
    assert_array_equal(aff, np.eye(4))
    assert_all_in((2, 3, 4), np.diag([4, 5, 6, 1]), shape, aff)
    # set output voxels to be same size as input. back to original shape
    shape, aff = vox2out_vox((2, 3, 4), np.diag([4, 5, 6, 1]), (4, 5, 6))
    assert_array_equal(shape, (2, 3, 4))
    assert_array_equal(aff, np.diag([4, 5, 6, 1]))
    assert_all_in((2, 3, 4), np.diag([4, 5, 6, 1]), shape, aff)
    # zero point preserved
    in_aff = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    shape, aff = vox2out_vox((2, 3, 4), in_aff)
    assert_array_equal(shape, (2, 3, 4))
    assert_array_equal(aff, in_aff)
    assert_all_in((2, 3, 4), in_aff, shape, aff)
    in_aff = [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]
    shape, aff = vox2out_vox((2, 3, 4), in_aff)
    assert_array_equal(shape, (2, 3, 4))
    assert_array_equal(aff, in_aff)
    assert_all_in((2, 3, 4), in_aff, shape, aff)
    # rotation around third axis
    in_aff = np.eye(4)
    in_aff[:3, :3] = euler2mat(np.pi / 4)
    shape, aff = vox2out_vox((2, 3, 4), in_aff)
    # x diff, y diff now 3 cos pi / 4 == 2.12, ceil to 3, add 1
    assert_array_equal(shape, (4, 4, 4))
    # most negative x now 2 cos pi / 4
    assert_almost_equal(aff, [[1, 0, 0, -2 * np.cos(np.pi / 4)],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    assert_all_in((2, 3, 4), in_aff, shape, aff)
    # Enforce number of axes
    assert_raises(ValueError, vox2out_vox, (2, 3, 4, 5), np.eye(4))
    assert_raises(ValueError, vox2out_vox, (2, 3, 4, 5, 6), np.eye(4))
    # Less than 3 is OK
    shape, aff = vox2out_vox((2, 3), np.eye(4))
    assert_array_equal(shape, (2, 3))
    assert_array_equal(aff, np.eye(4))
    assert_all_in((2, 3), np.eye(4), shape, aff)
    shape, aff = vox2out_vox((2,), np.eye(4))
    assert_array_equal(shape, (2,))
    assert_array_equal(aff, np.eye(4))
    assert_all_in((2,), np.eye(4), shape, aff)
    # Number of voxel sizes matches length
    shape, aff = vox2out_vox((2, 3), np.diag([4, 5, 6, 1]), (4, 5))
    assert_array_equal(shape, (2, 3))
    assert_array_equal(aff, np.diag([4, 5, 1, 1]))
    # Voxel sizes must be positive
    assert_raises(ValueError, vox2out_vox, (2, 3, 4), np.eye(4), [-1, 1, 1])
    assert_raises(ValueError, vox2out_vox, (2, 3, 4), np.eye(4), [1, 0, 1])


def test_slice2volume():
    # Get affine expressing selection of single slice from volume
    for axis, def_aff in zip((0, 1, 2), (
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])):
        for val in (0, 5, 10):
            exp_aff = np.array(def_aff)
            exp_aff[axis, -1] = val
            assert_array_equal(slice2volume(val, axis), exp_aff)
    assert_raises(ValueError, slice2volume, -1, 0)
    assert_raises(ValueError, slice2volume, 0, -1)
    assert_raises(ValueError, slice2volume, 0, 3)

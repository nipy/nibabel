# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

from ..affines import apply_affine, append_diag

from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_almost_equal, \
     assert_array_almost_equal


def validated_apply_affine(T, xyz):
    # This was the original apply_affine implementation that we've stashed here
    # to test against
    xyz = np.asarray(xyz)
    shape = xyz.shape[0:-1]
    XYZ = np.dot(np.reshape(xyz, (np.prod(shape), 3)), T[0:3,0:3].T)
    XYZ[:,0] += T[0,3]
    XYZ[:,1] += T[1,3]
    XYZ[:,2] += T[2,3]
    XYZ = np.reshape(XYZ, shape+(3,))
    return XYZ


def test_apply_affine():
    rng = np.random.RandomState(20110903)
    aff = np.diag([2, 3, 4, 1])
    pts = rng.uniform(size=(4,3))
    assert_array_equal(apply_affine(aff, pts),
                       pts * [[2, 3, 4]])
    aff[:3,3] = [10, 11, 12]
    assert_array_equal(apply_affine(aff, pts),
                       pts * [[2, 3, 4]] + [[10, 11, 12]])
    aff[:3,:] = rng.normal(size=(3,4))
    exp_res = np.concatenate((pts.T, np.ones((1,4))), axis=0)
    exp_res = np.dot(aff, exp_res)[:3,:].T
    assert_array_equal(apply_affine(aff, pts), exp_res)
    # Check we get the same result as the previous implementation
    assert_almost_equal(validated_apply_affine(aff, pts), apply_affine(aff, pts))
    # Check that lists work for inputs
    assert_array_equal(apply_affine(aff.tolist(), pts.tolist()), exp_res)
    # Check that it's the same as a banal implementation in the simple case
    aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
    pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
    exp_res = (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T
    assert_array_equal(apply_affine(aff, pts), exp_res)
    # That points can be reshaped and you'll get the same shape output
    pts = pts.reshape((2,2,3))
    exp_res = exp_res.reshape((2,2,3))
    assert_array_equal(apply_affine(aff, pts), exp_res)
    # That ND also works
    for N in range(2,6):
        aff = np.eye(N)
        nd = N-1
        aff[:nd,:nd] = rng.normal(size=(nd,nd))
        pts = rng.normal(size=(2,3,nd))
        res = apply_affine(aff, pts)
        # crude apply
        new_pts = np.ones((N,6))
        new_pts[:-1,:] = np.rollaxis(pts, -1).reshape((nd,6))
        exp_pts = np.dot(aff, new_pts)
        exp_pts = np.rollaxis(exp_pts[:-1,:], 0, 2)
        exp_res = exp_pts.reshape((2,3,nd))
        assert_array_almost_equal(res, exp_res)


def test_append_diag():
    # Routine for appending diagonal elements
    assert_array_equal(append_diag(np.diag([2,3,1]), [1]),
                       np.diag([2,3,1,1]))
    assert_array_equal(append_diag(np.diag([2,3,1]), [1,1]),
                       np.diag([2,3,1,1,1]))
    aff = np.array([[2,0,0],
                    [0,3,0],
                    [0,0,1],
                    [0,0,1]])
    assert_array_equal(append_diag(aff, [5], [9]),
                       [[2,0,0,0],
                        [0,3,0,0],
                        [0,0,0,1],
                        [0,0,5,9],
                        [0,0,0,1]])
    assert_array_equal(append_diag(aff, [5,6], [9,10]),
                       [[2,0,0,0,0],
                        [0,3,0,0,0],
                        [0,0,0,0,1],
                        [0,0,5,0,9],
                        [0,0,0,6,10],
                        [0,0,0,0,1]])
    aff = np.array([[2,0,0,0],
                    [0,3,0,0],
                    [0,0,0,1]])
    assert_array_equal(append_diag(aff, [5], [9]),
                       [[2,0,0,0,0],
                        [0,3,0,0,0],
                        [0,0,0,5,9],
                        [0,0,0,0,1]])
    # Length of starts has to match length of steps
    assert_raises(ValueError, append_diag, aff, [5,6], [9])

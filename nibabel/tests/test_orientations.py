''' Testing for orientations module '''

import numpy as np

from nose.tools import assert_true, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nibabel.orientations import (io_orientation, orientation_affine, flip_axis,
                             apply_orientation)

from nibabel.testing import parametric

IN_ARRS = (np.eye(4),
           [[0,0,1,0],
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1]],
           [[0,1,0,0],
            [0,0,1,0],
            [1,0,0,0],
            [0,0,0,1]],
           [[1,1,0,0],
            [1,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]
           )
OUT_ORNTS = ([[0,1],
              [1,1],
              [2,1]],
             [[2,1],
              [1,1],
              [0,1]],
             [[2,1],
              [0,1],
              [1,1]],
             [[0,1],
              [1,1],
              [2,1]]
             )
IN_ARRS = [np.array(arr) for arr in IN_ARRS]
OUT_ORNTS = [np.array(ornt) for ornt in OUT_ORNTS]


def same_transform(taff, ornt, shape):
    # Applying transformations implied by `ornt` to a made-up array
    # ``arr`` of shape `shape`, results in ``t_arr``. When the point
    # indices from ``arr`` are transformed by (the inverse of) `taff`,
    # and we index into ``t_arr`` with these transformed points, then we
    # should get the same values as we would from indexing into arr with
    # the untransformed points. 
    shape = np.array(shape)
    size = np.prod(shape)
    arr = np.arange(size).reshape(shape)
    # apply ornt transformations
    t_arr = apply_orientation(arr, ornt)
    # get all point indices in arr
    i,j,k = shape
    arr_pts = np.mgrid[:i,:j,:k].reshape((3,-1))
    # inverse of taff takes us from point index in arr to point index in
    # t_arr
    itaff = np.linalg.inv(taff)
    # apply itaff so that points indexed in t_arr should correspond 
    o2t_pts = np.dot(itaff[:3,:3], arr_pts) + itaff[:3,3][:,None]
    assert np.allclose(np.round(o2t_pts), o2t_pts)
    # fancy index out the t_arr values
    vals = t_arr[list(o2t_pts.astype('i'))]
    return np.all(vals == arr.ravel())


@parametric
def test_apply():
    # most tests are in ``same_transform`` above, via the
    # test_io_orientations. 
    a = np.arange(24).reshape((2,3,4))
    # Test 4D
    t_arr = apply_orientation(a[:,:,:,None], ornt)
    yield assert_equal, t_arr.ndim, 4


@parametric
def test_flip_axis():
    a = np.arange(24).reshape((2,3,4))
    yield assert_array_equal(
        flip_axis(a),
        np.flipud(a))
    yield assert_array_equal(
        flip_axis(a, axis=0),
        np.flipud(a))
    yield assert_array_equal(
        flip_axis(a, axis=1),
        np.fliplr(a))
    # check accepts array-like
    yield assert_array_equal(
        flip_axis(a.tolist(), axis=0),
        np.flipud(a))
    # third dimension
    b = a.transpose()
    b = np.flipud(b)
    b = b.transpose()
    yield assert_array_equal(flip_axis(a, axis=2), b)
    

@parametric
def test_io_orientation():
    shape = (2,3,4)
    for in_arr, out_ornt in zip(IN_ARRS, OUT_ORNTS):
        ornt = io_orientation(in_arr)
        yield assert_array_equal(ornt, out_ornt)
        taff = orientation_affine(ornt, shape)
        yield assert_true(same_transform(taff, ornt, shape))
        for axno in range(3):
            arr = in_arr.copy()
            ex_ornt = out_ornt.copy()
            # flip the input axis in affine
            arr[:,axno] *= -1
            # check that result shows flip
            ex_ornt[axno, 1] = -1
            ornt = io_orientation(arr)
            yield assert_array_equal(ornt, ex_ornt)
            taff = orientation_affine(ornt, shape)
            yield assert_true(same_transform(taff, ornt, shape))
           

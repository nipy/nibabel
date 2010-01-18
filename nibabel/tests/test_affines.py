''' Testing for affines module '''

import numpy as np

from nose.tools import assert_true, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nibabel.affines import (io_orientation, orientation_affine, flip_axis,
                             apply_orientation)

from nibabel.testing import parametric


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
    t_arr, taff2 = apply_orientation(arr, ornt)
    assert np.all(taff2 == taff)
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
    # most tests are in the ``same_transform`` function, where we
    # run a lot of orientation tests via test_io_orientation
    # Test 4D
    a = np.arange(24).reshape((2,3,4,1))
    ornt = [[0,1],
            [1,1],
            [2,1]]
    t_arr, t_aff = apply_orientation(a, ornt)
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
    in_arrs = (np.eye(4),
               np.array([[0,0,1,0],
                         [0,1,0,0],
                         [1,0,0,0],
                         [0,0,0,1]]),
               np.array([[0,1,0,0],
                         [0,0,1,0],
                         [1,0,0,0],
                         [0,0,0,1]]))
    out_ornts = (np.array([[0,1],
                           [1,1],
                           [2,1]]),
                 np.array([[2,1],
                           [1,1],
                           [0,1]]),
                 np.array([[2,1],
                           [0,1],
                           [1,1]]))
    for in_arr, out_ornt in zip(in_arrs, out_ornts):
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
    # tie breaks?
    arr = [[1,1,0,0],
           [1,1,0,0],
           [0,0,1,0],
           [0,0,0,1]]
    exo = [[0,1],
           [1,1],
           [2,1]]
    ornt = io_orientation(arr)
    yield assert_array_equal(ornt, exo)
    taff = orientation_affine(ornt, shape)
    yield assert_true(same_transform(taff, ornt, shape))
           

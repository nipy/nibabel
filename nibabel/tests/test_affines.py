''' Testing for affines module '''

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nibabel.affines import io_orientation

from nibabel.testing import parametric


@parametric
def test_io_orientation():
    in_arrs = (np.eye(4),
               np.array([[0,0,1,0],
                         [0,1,0,0],
                         [1,0,0,0],
                         [0,0,0,1]]))
    out_ornts = (np.array([[0,1],
                           [1,1],
                           [2,1]]),
                 np.array([[2,1],
                           [1,1],
                           [0,1]]))
    for in_arr, out_ornt in zip(in_arrs, out_ornts):
        ornt = io_orientation(in_arr)
        yield assert_array_equal(ornt, out_ornt)
        for axno in range(3):
            arr = in_arr.copy()
            arr[axno,:] *= -1
            ex_ornt = out_ornt.copy()
            ex_ornt[axno, 1] = -1
            ornt = io_orientation(arr)
            yield assert_array_equal(ornt, ex_ornt)
    arr = [[0,1,0,0],
           [0,0,-1,0],
           [1,0,0,0],
           [0,0,0,1]]
    ornt = io_orientation(arr)
    yield assert_array_equal(ornt, [[1,1],
                                    [2,-1],
                                    [0,1]])
    # tie breaks?
    


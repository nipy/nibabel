''' Utilities for affine transformation matrices '''

import numpy as np
import numpy.linalg as npl


def io_orientation(affine):
    ''' Return orientation of input axes in terms of output axes

    Parameters
    ----------
    affine : (4,4) ndarray-like

    Returns
    -------
    orientations : (3,2) ndarray
       one row per input axis, where the first value in each row is the
       closest corresponding output axis, and the second value is 1 if
       the input axis is in the same direction, and -1 if it is in the
       opposite direction.
    '''
    # make a copy of the array for the later axis selection
    affine = np.array(affine)
    # extract the underlying rotation matrix
    RZS = affine[:3,:3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    RS = RZS / zooms
    # Transform below is polar decomposition, returning the closest
    # orthogonal (rotation) matrix PR, to input R
    P, S, Qs = npl.svd(RS)
    R = np.dot(P, Qs)
    # inverse (transpose) gives the rotation of the unit x, y, z vectors
    # to input coordinates.
    iR = R.T
    # Therefore, the position of abs max of iR, for column N, is the
    # input axis changing most as output axis N changes.  In case there
    # are ties, we choose the axes iteratively, removing used axes from
    # consideration as we go
    ornt = np.ones((3,2), dtype=np.int8)
    for in_ax in range(3):
        col = iR[:,in_ax]
        out_ax = np.argmax(np.abs(col))
        ornt[in_ax,0] = out_ax
        assert col[out_ax] != 0
        if col[out_ax] < 0:
            ornt[in_ax,1] = -1
        # remove the identified axis from further consideration
        iR[out_ax,:] = 0 # hence the copy at the top
    return ornt

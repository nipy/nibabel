# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Utility routines for working with points and affine transforms
"""

import numpy as np


def apply_affine(aff, pts):
    """ Apply affine matrix `aff` to points `pts`

    Returns result of application of `aff` to the *right* of `pts`.  The
    coordinate dimension of `pts` should be the last.

    For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
    length 3 - maybe it will just be N by 3. The return value is the transformed
    points, in this case::

        res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
        transformed_pts = res.T

    Notice though, that this routine is more general, in that `aff` can have any
    shape (N,N), and `pts` can have any shape, as long as the last dimension is
    for the coordinates, and is therefore length N-1.

    Parameters
    ----------
    aff : (N, N) array-like
        Homogenous affine, for 3D points, will be 4 by 4. Contrary to first
        appearance, the affine will be applied on the left of `pts`.
    pts : (..., N-1) array-like
        Points, where the last dimension contains the coordinates of each point.
        For 3D, the last dimension will be length 3.

    Returns
    -------
    transformed_pts : (..., N-1) array
        transformed points

    Examples
    --------
    >>> aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
    >>> pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
    >>> apply_affine(aff, pts)
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]])

    Just to show that in the simple 3D case, it is equivalent to:

    >>> (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]])

    But `pts` can be a more complicated shape:

    >>> pts = pts.reshape((2,2,3))
    >>> apply_affine(aff, pts)
    array([[[14, 14, 24],
            [16, 17, 28]],
    <BLANKLINE>
           [[20, 23, 36],
            [24, 29, 44]]])
    """
    aff = np.asarray(aff)
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, shape[-1]))
    # rzs == rotations, zooms, shears
    rzs = aff[:-1,:-1]
    trans = aff[:-1,-1]
    res = np.dot(pts, rzs.T) + trans[None,:]
    return res.reshape(shape)


def to_matvec(transform):
    """Split a transform into its matrix and vector components.

    The tranformation must be represented in homogeneous coordinates and is
    split into its rotation matrix and translation vector components.

    Parameters
    ----------
    transform : array-like
        NxM transform matrix in homogeneous coordinates representing an affine
        transformation from an (N-1)-dimensional space to an (M-1)-dimensional
        space. An example is a 4x4 transform representing rotations and
        translations in 3 dimensions. A 4x3 matrix can represent a 2-dimensional
        plane embedded in 3 dimensional space.

    Returns
    -------
    matrix : (N-1, M-1) array
        Matrix component of `transform`
    vector : (M-1,) array
        Vector compoent of `transform`

    See Also
    --------
    from_matvec

    Examples
    --------
    >>> aff = np.diag([2, 3, 4, 1])
    >>> aff[:3,3] = [9, 10, 11]
    >>> to_matvec(aff)
    (array([[2, 0, 0],
           [0, 3, 0],
           [0, 0, 4]]), array([ 9, 10, 11]))
    """
    transform = np.asarray(transform)
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector


def from_matvec(matrix, vector=None):
    """ Combine a matrix and vector into an homogeneous affine

    Combine a rotation / scaling / shearing matrix and translation vector into a
    transform in homogeneous coordinates.

    Parameters
    ----------
    matrix : array-like
        An NxM array representing the the linear part of the transform.
        A transform from an M-dimensional space to an N-dimensional space.
    vector : None or array-like, optional
        None or an (N,) array representing the translation. None corresponds to
        an (N,) array of zeros.

    Returns
    -------
    xform : array
        An (N+1, M+1) homogenous transform matrix.

    See Also
    --------
    to_matvec

    Examples
    --------
    >>> from_matvec(np.diag([2, 3, 4]), [9, 10, 11])
    array([[ 2,  0,  0,  9],
           [ 0,  3,  0, 10],
           [ 0,  0,  4, 11],
           [ 0,  0,  0,  1]])

    The `vector` argument is optional:

    >>> from_matvec(np.diag([2, 3, 4]))
    array([[2, 0, 0, 0],
           [0, 3, 0, 0],
           [0, 0, 4, 0],
           [0, 0, 0, 1]])
    """
    matrix = np.asarray(matrix)
    nin, nout = matrix.shape
    t = np.zeros((nin+1,nout+1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin, nout] = 1.
    if not vector is None:
        t[0:nin, nout] = vector
    return t


def append_diag(aff, steps, starts=()):
    """ Add diagonal elements `steps` and translations `starts` to affine

    Typical use is in expanding 4x4 affines to larger dimensions.  Nipy is the
    main consumer because it uses NxM affines, whereas we generally only use 4x4
    affines; the routine is here for convenience.

    Parameters
    ----------
    aff : 2D array
        N by M affine matrix
    steps : scalar or sequence
        diagonal elements to append.
    starts : scalar or sequence
        elements to append to last column of `aff`, representing translations
        corresponding to the `steps`. If empty, expands to a vector of zeros
        of the same length as `steps`

    Returns
    -------
    aff_plus : 2D array
        Now P by Q where L = ``len(steps)`` and P == N+L, Q=N+L

    Examples
    --------
    >>> aff = np.eye(4)
    >>> aff[:3,:3] = np.arange(9).reshape((3,3))
    >>> append_diag(aff, [9, 10], [99,100])
    array([[   0.,    1.,    2.,    0.,    0.,    0.],
           [   3.,    4.,    5.,    0.,    0.,    0.],
           [   6.,    7.,    8.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    9.,    0.,   99.],
           [   0.,    0.,    0.,    0.,   10.,  100.],
           [   0.,    0.,    0.,    0.,    0.,    1.]])
    """
    aff = np.asarray(aff)
    steps = np.atleast_1d(steps)
    starts = np.atleast_1d(starts)
    n_steps = len(steps)
    if len(starts) == 0:
        starts = np.zeros(n_steps, dtype=steps.dtype)
    elif len(starts) != n_steps:
        raise ValueError('Steps should have same length as starts')
    old_n_out, old_n_in = aff.shape[0]-1, aff.shape[1]-1
    # make new affine
    aff_plus = np.zeros((old_n_out + n_steps + 1,
                         old_n_in + n_steps + 1), dtype=aff.dtype)
    # Get stuff from old affine
    aff_plus[:old_n_out,:old_n_in] = aff[:old_n_out, :old_n_in]
    aff_plus[:old_n_out,-1] = aff[:old_n_out,-1]
    # Add new diagonal elements
    for i, el in enumerate(steps):
        aff_plus[old_n_out+i, old_n_in+i] = el
    # Add translations for new affine, plus last 1
    aff_plus[old_n_out:,-1] = list(starts) + [1]
    return aff_plus

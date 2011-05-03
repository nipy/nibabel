# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Utilities for calculating and applying affine orientations '''

import numpy as np
import numpy.linalg as npl


class OrientationError(Exception):
    pass


def io_orientation(affine, tol=None):
    ''' Orientation of input axes in terms of output axes for `affine`

    Valid for an affine transformation from ``m`` dimensions to ``n``
    dimensions (``affine.shape == (n+1, m+1)``).

    The calculated orientations can be used to transform associated
    arrays to best match the output orientations. If ``n`` > ``m``, then
    some of the output axes should be considered dropped in this
    orientation.

    Parameters
    ----------
    affine : (n+1,m+1) ndarray-like
       Transformation affine from ``m`` inputs to ``n`` outputs.
       Usually this will be a shape (4,4) matrix, transforming 3 inputs
       to 3 outputs, but the code also handles the more general case
    tol : {None, float}, optional
       threshold below which SVD values of the affine are considered
       zero. If `tol` is None, and ``S`` is an array with singular
       values for `affine`, and ``eps`` is the epsilon value for
       datatype of ``S``, then `tol` set to ``S.max() * eps``.

    Returns
    -------
    orientations : (n,2) ndarray
       one row per input axis, where the first value in each row is the
       closest corresponding output axis, and the second value is 1 if
       the input axis is in the same direction as the corresponding
       output axis, , and -1 if it is in the opposite direction, to the
       corresponding output axis.  If a row is [np.nan, np.nan], which
       can happen when n > m, then this row should be considered
       dropped.
    '''
    affine = np.asarray(affine)
    n, m = affine.shape[0]-1, affine.shape[1]-1
    # extract the underlying rotation matrix
    RZS = affine[:n, :m]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    RS = RZS / zooms
    # Transform below is polar decomposition, returning the closest
    # shearless matrix R to RS
    P, S, Qs = npl.svd(RS)
    # Threshold the singular values to determine the rank.
    if tol is None:
        tol = S.max() * np.finfo(S.dtype).eps
    keep = (S > tol)
    R = np.dot(P[:, keep], Qs[keep])
    # the matrix R is such that np.dot(R,R.T) is projection onto the
    # columns of P[:,keep] and np.dot(R.T,R) is projection onto the rows
    # of Qs[keep].  R (== np.dot(R, np.eye(m))) gives rotation of the
    # unit input vectors to output coordinates.  Therefore, the row
    # index of abs max R[:,N], is the output axis changing most as input
    # axis N changes.  In case there are ties, we choose the axes
    # iteratively, removing used axes from consideration as we go
    ornt = np.ones((n, 2), dtype=np.int8) * np.nan
    for in_ax in range(m):
        col = R[:, in_ax]
        if not np.alltrue(np.equal(col, 0)):
            out_ax = np.argmax(np.abs(col))
            ornt[in_ax, 0] = out_ax
            assert col[out_ax] != 0
            if col[out_ax] < 0:
                ornt[in_ax, 1] = -1
            else:
                ornt[in_ax, 1] = 1
            # remove the identified axis from further consideration, by
            # zeroing out the corresponding row in R
            R[out_ax, :] = 0
    return ornt


def _ornt_to_affine(orientations):
    ''' Create affine transformation matrix determined by orientations.

    This transformation will simply flip, transpose, and possibly drop some
    coordinates.

    Parameters
    ----------
    orientations : (n,2) ndarray
       one row per input axis, where the first value in each row is the
       closest corresponding output axis, and the second value is 1 if
       the input axis is in the same direction as the corresponding
       output axis, and -1 if it is in the opposite direction, to the
       corresponding output axis.  If a row has first entry np.nan, then
       this axis is dropped from the output.

    Returns
    -------
    affine : (m+1,n+1) ndarray
       matrix representing flipping / dropping axes. m is equal to the
       number of rows of orientations[:,0] that are not np.nan
    '''
    ornt = np.asarray(orientations)
    n = ornt.shape[0]
    keep = ~np.isnan(ornt[:, 1])
    # These are the input coordinate axes that do have a matching output
    # column in the orientation.  That is, if the 2nd row is [np.nan,
    # np.nan] then the orientation indicates that no output axes of an
    # affine with this orientation matches the 2nd input coordinate
    # axis.  This would happen if the 2nd row of the affine that
    # generated ornt was [0,0,0,*]
    axes_kept = np.arange(n)[keep]
    m = keep.sum(0)
    # the matrix P represents the affine transform impled by ornt. If
    # all entries of ornt are not np.nan, then P is square otherwise it
    # has more columns than rows indicating some coordinates were
    # dropped
    P = np.zeros((m + 1, n + 1))
    P[-1, -1] = 1
    for idx in range(m):
        axs, flip = ornt[axes_kept[idx]]
        P[idx, axs] = flip
    return P


def apply_orientation(arr, ornt):
    ''' Apply transformations implied by `ornt` to the first
    n axes of the array `arr`

    Parameters
    ----------
    arr : array-like of data with ndim >= n
    ornt : (n,2) orientation array
       orientation transform. ``ornt[N,1]` is flip of axis N of the
       array implied by `shape`, where 1 means no flip and -1 means
       flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
       there's an array ``arr`` of shape `shape`, the flip would
       correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
       the transpose that needs to be done to the implied array, as in
       ``arr.transpose(ornt[:,0])``

    Returns
    -------
    t_arr : ndarray
       data array `arr` transformed according to ornt
    '''
    t_arr = np.asarray(arr)
    ornt = np.asarray(ornt)
    n = ornt.shape[0]
    if t_arr.ndim < n:
        raise OrientationError('Data array has fewer dimensions than '
                               'orientation')
    # no coordinates can be dropped for applying the orientations
    if np.any(np.isnan(ornt[:, 0])):
        raise OrientationError('Cannot drop coordinates when '
                               'applying orientation to data')
    shape = t_arr.shape
    # apply ornt transformations
    for ax, flip in enumerate(ornt[:, 1]):
        if flip == -1:
            t_arr = flip_axis(t_arr, axis=ax)
    full_transpose = np.arange(t_arr.ndim)
    # ornt indicates the transpose that has occurred - we reverse it
    full_transpose[:n] = np.argsort(ornt[:, 0])
    t_arr = t_arr.transpose(full_transpose)
    return t_arr


def orientation_affine(ornt, shape):
    ''' Affine transform resulting from transforms implied in `ornt`

    Imagine you have an array ``arr`` of shape `shape`, and you apply the
    transforms implied by `ornt` (more below), to get ``tarr``.
    ``tarr`` may have a different shape ``shape_prime``.  This routine
    returns the affine that will take a array coordinate for ``tarr``
    and give you the corresponding array coordinate in ``arr``.

    Parameters
    ----------
    ornt : (n,2) ndarray
       orientation transform. ``ornt[N,1]` is flip of axis N of the
       array implied by `shape`, where 1 means no flip and -1 means
       flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
       there's an array ``arr`` of shape `shape`, the flip would
       correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
       the transpose that needs to be done to the implied array, as in
       ``arr.transpose(ornt[:,0])``

    shape : length n sequence
       shape of array you may transform with `ornt`

    Returns
    -------
    transformed_affine : (n+1,n+1) ndarray
       An array ``arr`` (shape `shape`) might be transformed according
       to `ornt`, resulting in a transformed array ``tarr``.
       `transformed_affine` is the transform that takes you from array
       coordinates in ``tarr`` to array coordinates in ``arr``.
    '''
    ornt = np.asarray(ornt)
    n = ornt.shape[0]
    shape = np.array(shape)[:n]
    # ornt implies a flip, followed by a transpose.   We need the affine
    # that inverts these.  Thus we need the affine that first undoes the
    # effect of the transpose, then undoes the effects of the flip.
    # ornt indicates the transpose that has occurred to get the current
    # ordering, relative to canonical, so we just use that
    undo_reorder = np.eye(n + 1)[list(ornt[:, 0]) + [n], :]
    undo_flip = np.diag(list(ornt[:, 1]) + [1.0])
    center_trans = -(shape - 1) / 2.0
    undo_flip[:n, n] = (ornt[:, 1] * center_trans) - center_trans
    return np.dot(undo_flip, undo_reorder)


def flip_axis(arr, axis=0):
    ''' Flip contents of `axis` in array `arr`

    ``flip_axis`` is the same transform as ``np.flipud``, but for any
    axis.  For example ``flip_axis(arr, axis=0)`` is the same transform
    as ``np.flipud(arr)``, and ``flip_axis(arr, axis=1)`` is the same
    transform as ``np.fliplr(arr)``

    Parameters
    ----------
    arr : array-like
    axis : int, optional
       axis to flip.  Default `axis` == 0

    Returns
    -------
    farr : array
       Array with axis `axis` flipped

    Examples
    --------
    >>> a = np.arange(6).reshape((2,3))
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> flip_axis(a, axis=0)
    array([[3, 4, 5],
           [0, 1, 2]])
    >>> flip_axis(a, axis=1)
    array([[2, 1, 0],
           [5, 4, 3]])
    '''
    arr = np.asanyarray(arr)
    arr = arr.swapaxes(0, axis)
    arr = np.flipud(arr)
    return arr.swapaxes(axis, 0)


def ornt2axcodes(ornt, labels=None):
    """ Convert orientation `ornt` to labels for axis directions

    Parameters
    ----------
    ornt : (N,2) array-like
        orientation array - see io_orientation docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That is,
        if the first row in `ornt` is ``[1, 1]``, and the second (2,) sequence
        in `labels` is ('back', 'front') then the first returned axis code will
        be ``'front'``.  If the first row in `ornt` had been ``[1, -1]`` then
        the first returned value would have been ``'back'``.  If None,
        equivalent to ``(('L','R'),('P','A'),('I','S'))`` - that is - RAS axes.

    Returns
    -------
    axcodes : (N,) tuple
        labels for positive end of voxel axes.  Dropped axes get a label of
        None.

    Examples
    --------
    >>> ornt2axcodes([[1, 1],[0,-1],[2,1]], (('L','R'),('B','F'),('D','U')))
    ('F', 'L', 'U')
    """
    if labels is None:
        labels = zip('LPI', 'RAS')
    axcodes = []
    for axno, direction in np.asarray(ornt):
        if np.isnan(axno):
            axcodes.append(None)
            continue
        axint = int(np.round(axno))
        if axint != axno:
            raise ValueError('Non integer axis number %f' % axno)
        elif direction == 1:
            axcode = labels[axint][1]
        elif direction == -1:
            axcode = labels[axint][0]
        else:
            raise ValueError('Direction should be -1 or 1')
        axcodes.append(axcode)
    return tuple(axcodes)


def aff2axcodes(aff, labels=None, tol=None):
    """ axis direction codes for affine `aff`

    Parameters
    ----------
    aff : (N,M) array-like
        affine transformation matrix
    labels : optional, None or sequence of (2,) sequences
        Labels for negative and positive ends of output axes of `aff`.  See
        docstring for ``ornt2axcodes`` for more detail
    tol : None or float
        Tolerance for SVD of affine - see ``io_orientation`` for more detail.

    Returns
    -------
    axcodes : (N,) tuple
        labels for positive end of voxel axes.  Dropped axes get a label of
        None.

    Examples
    --------
    >>> aff = [[0,1,0,10],[-1,0,0,20],[0,0,1,30],[0,0,0,1]]
    >>> aff2axcodes(aff, (('L','R'),('B','F'),('D','U')))
    ('B', 'R', 'U')
    """
    ornt = io_orientation(aff, tol)
    return ornt2axcodes(ornt, labels)

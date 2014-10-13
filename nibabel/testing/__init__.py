# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Utilities for testing '''
from os.path import dirname, abspath, join as pjoin

import numpy as np
from warnings import catch_warnings, simplefilter

# set path to example data
data_path = abspath(pjoin(dirname(__file__), '..', 'tests', 'data'))

# Allow failed import of nose if not now running tests
try:
    import nose.tools as nt
except ImportError:
    pass
else:
    from nose.tools import (assert_equal, assert_not_equal,
                            assert_true, assert_false, assert_raises)


def assert_dt_equal(a, b):
    """ Assert two numpy dtype specifiers are equal

    Avoids failed comparison between int32 / int64 and intp
    """
    assert_equal(np.dtype(a).str, np.dtype(b).str)


def assert_allclose_safely(a, b, match_nans=True):
    """ Allclose in integers go all wrong for large integers
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if match_nans:
        nans = np.isnan(a)
        np.testing.assert_array_equal(nans, np.isnan(b))
        if np.any(nans):
            nans = np.logical_not(nans)
            a = a[nans]
            b = b[nans]
    if a.dtype.kind in 'ui':
        a = a.astype(float)
    if b.dtype.kind in 'ui':
        b = b.astype(float)
    assert_true(np.allclose(a, b))


class suppress_warnings(catch_warnings):
    """ Version of ``catch_warnings`` class that suppresses warnings
    """
    def __enter__(self):
        res = super(suppress_warnings, self).__enter__()
        simplefilter('ignore')
        return res

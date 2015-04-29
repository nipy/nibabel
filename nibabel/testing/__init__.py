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
from warnings import catch_warnings

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
    a = np.atleast_1d(a)  # 0d arrays cannot be indexed
    a, b = np.broadcast_arrays(a, b)
    if match_nans:
        nans = np.isnan(a)
        np.testing.assert_array_equal(nans, np.isnan(b))
        to_test = ~nans
    else:
        to_test = np.ones(a.shape, dtype=bool)
    # Deal with float128 inf comparisons (bug in numpy 1.9.2)
    # np.allclose(np.float128(np.inf), np.float128(np.inf)) == False
    to_test = to_test & (a != b)
    a = a[to_test]
    b = b[to_test]
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


class catch_warn_reset(catch_warnings):
    """ Version of ``catch_warnings`` class that resets warning registry
    """
    def __init__(self, *args, **kwargs):
        self.modules = kwargs.pop('modules', [])
        self._warnreg_copies = {}
        super(catch_warn_reset, self).__init__(*args, **kwargs)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(catch_warn_reset, self).__enter__()

    def __exit__(self, *exc_info):
        super(catch_warn_reset, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])

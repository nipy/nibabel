# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Utilities for testing '''

import re
import os
import sys
import warnings
from pkg_resources import resource_filename

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from .np_features import memmap_after_ufunc
from .helpers import bytesio_filemap, bytesio_round_trip, assert_data_similar

from itertools import zip_longest


def test_data(subdir=None, fname=None):
    if subdir is None:
        resource = os.path.join('tests', 'data')
    elif subdir in ('gifti', 'nicom', 'externals'):
        resource = os.path.join(subdir, 'tests', 'data')
    else:
        raise ValueError("Unknown test data directory: %s" % subdir)

    if fname is not None:
        resource = os.path.join(resource, fname)

    return resource_filename('nibabel', resource)


# set path to example data
data_path = test_data()


def assert_dt_equal(a, b):
    """ Assert two numpy dtype specifiers are equal

    Avoids failed comparison between int32 / int64 and intp
    """
    assert np.dtype(a).str == np.dtype(b).str


def assert_allclose_safely(a, b, match_nans=True, rtol=1e-5, atol=1e-8):
    """ Allclose in integers go all wrong for large integers
    """
    a = np.atleast_1d(a)  # 0d arrays cannot be indexed
    a, b = np.broadcast_arrays(a, b)
    if match_nans:
        nans = np.isnan(a)
        assert_array_equal(nans, np.isnan(b))
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
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def assert_arrays_equal(arrays1, arrays2):
    """ Check two iterables yield the same sequence of arrays. """
    for arr1, arr2 in zip_longest(arrays1, arrays2, fillvalue=None):
        assert (arr1 is not None and arr2 is not None)
        assert_array_equal(arr1, arr2)


def assert_re_in(regex, c, flags=0):
    """Assert that container (list, str, etc) contains entry matching the regex
    """
    if not isinstance(c, (list, tuple)):
        c = [c]
    for e in c:
        if re.match(regex, e, flags=flags):
            return
    raise AssertionError("Not a single entry matched %r in %r" % (regex, c))


def get_fresh_mod(mod_name=__name__):
    # Get this module, with warning registry empty
    my_mod = sys.modules[mod_name]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        pass
    return my_mod


class clear_and_catch_warnings(warnings.catch_warnings):
    """ Context manager that resets warning registry for catching warnings

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    For compatibility with Python 3.0, please consider all arguments to be
    keyword-only.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. The objects appended to the list are arguments whose
        attributes mirror the arguments to ``showwarning()``.

        NOTE: nibabel difference from numpy: default is True

    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit

    Examples
    --------
    >>> import warnings
    >>> with clear_and_catch_warnings(modules=[np.core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     # do something that raises a warning in np.core.fromnumeric
    """
    class_modules = ()

    def __init__(self, record=True, modules=()):
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super(clear_and_catch_warnings, self).__init__(record=record)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(clear_and_catch_warnings, self).__enter__()

    def __exit__(self, *exc_info):
        super(clear_and_catch_warnings, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


class error_warnings(clear_and_catch_warnings):
    """ Context manager to check for warnings as errors.  Usually used with
    ``assert_raises`` in the with block

    Examples
    --------
    >>> with error_warnings():
    ...     try:
    ...         warnings.warn('Message', UserWarning)
    ...     except UserWarning:
    ...         print('I consider myself warned')
    I consider myself warned
    """
    filter = 'error'

    def __enter__(self):
        mgr = super(error_warnings, self).__enter__()
        warnings.simplefilter(self.filter)
        return mgr


class suppress_warnings(error_warnings):
    """ Version of ``catch_warnings`` class that suppresses warnings
    """
    filter = 'ignore'


EXTRA_SET = os.environ.get('NIPY_EXTRA_TESTS', '').split(',')


def runif_extra_has(test_str):
    """Decorator checks to see if NIPY_EXTRA_TESTS env var contains test_str"""
    return unittest.skipUnless(test_str in EXTRA_SET, "Skip {0} tests.".format(test_str))


def assert_arr_dict_equal(dict1, dict2):
    """ Assert that two dicts are equal, where dicts contain arrays
    """
    assert set(dict1) == set(dict2)
    for key, value1 in dict1.items():
        value2 = dict2[key]
        assert_array_equal(value1, value2)


class BaseTestCase(unittest.TestCase):
    """ TestCase that does not attempt to run if prefixed with a ``_``

    This restores the nose-like behavior of skipping so-named test cases
    in test runners like pytest.
    """
    def setUp(self):
        if self.__class__.__name__.startswith('_'):
            raise unittest.SkipTest("Base test case - subclass to run")
        super().setUp()

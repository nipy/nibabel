""" Tests for warnings context managers
"""

from __future__ import with_statement

from warnings import warn, simplefilter, filters

from ..checkwarns import ErrorWarnings, IgnoreWarnings

from nose.tools import assert_true, assert_equal, assert_raises


def test_warn_error():
    # Check warning error context manager
    n_warns = len(filters)
    with ErrorWarnings():
        assert_raises(UserWarning, warn, 'A test')
    assert_equal(n_warns, len(filters))
    def f():
        with ErrorWarnings():
            simplefilter('ignore')
    try:
        assert_raises(RuntimeError, f)
        assert_equal(n_warns+2, len(filters))
    finally:
        filters.pop(0)
        filters.pop(0)
    # Check other errors are propagated
    def f():
        with ErrorWarnings():
            raise ValueError('An error')
    assert_raises(ValueError, f)


def test_warn_ignore():
    # Check warning ignore context manager
    n_warns = len(filters)
    with IgnoreWarnings():
        warn('Here is a warning, you will not see it')
        warn('Nor this one', DeprecationWarning)
    assert_equal(n_warns, len(filters))
    def f():
        with IgnoreWarnings():
            simplefilter('error')
    try:
        assert_raises(RuntimeError, f)
        assert_equal(n_warns+2, len(filters))
    finally:
        filters.pop(0)
        filters.pop(0)
    # Check other errors are propagated
    def f():
        with IgnoreWarnings():
            raise ValueError('An error')
    assert_raises(ValueError, f)

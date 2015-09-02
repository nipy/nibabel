""" Tests for warnings context managers
"""
from __future__ import division, print_function, absolute_import

from nose.tools import assert_true, assert_equal, assert_raises
from ..testing import (error_warnings, suppress_warnings,
                       clear_and_catch_warnings)


def test_ignore_and_error_warnings():
    with clear_and_catch_warnings() as w:
        from ..checkwarns import ErrorWarnings, IgnoreWarnings
        assert_equal(len(w), 1)
        assert_equal(w[0].category, FutureWarning)

    with clear_and_catch_warnings() as w:
        with IgnoreWarnings():
            pass
        assert_equal(len(w), 1)
        assert_equal(w[0].category, FutureWarning)

    with clear_and_catch_warnings() as w:
        with ErrorWarnings():
            pass
        assert_equal(len(w), 1)
        assert_equal(w[0].category, FutureWarning)

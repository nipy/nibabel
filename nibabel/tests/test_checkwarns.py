""" Tests for warnings context managers
"""
from __future__ import division, print_function, absolute_import

from nose.tools import assert_equal
from ..testing import clear_and_catch_warnings, suppress_warnings


def test_ignore_and_error_warnings():
    with suppress_warnings():
        from .. import checkwarns

    with clear_and_catch_warnings() as w:
        checkwarns.IgnoreWarnings()
        assert_equal(len(w), 1)
        assert_equal(w[0].category, FutureWarning)

    with clear_and_catch_warnings() as w:
        checkwarns.ErrorWarnings()
        assert_equal(len(w), 1)
        assert_equal(w[0].category, FutureWarning)

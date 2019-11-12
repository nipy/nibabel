""" Tests for warnings context managers
"""
from ..testing_pytest import assert_warns, suppress_warnings


def test_ignore_and_error_warnings():
    with suppress_warnings():
        from .. import checkwarns

    assert_warns(DeprecationWarning, checkwarns.IgnoreWarnings)
    assert_warns(DeprecationWarning, checkwarns.ErrorWarnings)

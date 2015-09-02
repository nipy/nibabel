""" Tests for warnings context managers
"""
from __future__ import division, print_function, absolute_import

from ..checkwarns import ErrorWarnings, IgnoreWarnings

from nose.tools import assert_true, assert_equal, assert_raises
from ..testing import (error_warnings, suppress_warnings,
                       clear_and_catch_warnings)

""" Tests for nisexts.sexts module
"""

import sys
import imp

from ..sexts import package_check

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

FAKE_NAME = 'nisext_improbable'
assert FAKE_NAME not in sys.modules
FAKE_MODULE = imp.new_module('nisext_fake')


def test_package_check():
    # Try to use a required package - raise error
    assert_raises(RuntimeError, package_check, FAKE_NAME)
    # Optional, log.warn
    package_check(FAKE_NAME, optional=True)
    # Make a package
    sys.modules[FAKE_NAME] = FAKE_MODULE
    # Now it passes if we don't check the version
    package_check(FAKE_NAME)
    # A fake version
    FAKE_MODULE.__version__ = '0.2'
    package_check(FAKE_NAME, version='0.2')
    # fails when version not good enough
    assert_raises(RuntimeError, package_check, FAKE_NAME, '0.3')
    # Unless optional in which case log.warns
    package_check(FAKE_NAME, version='0.3', optional=True)
    # Might do custom version check
    package_check(FAKE_NAME, version='0.2', version_getter=lambda x: '0.2')

""" Tests for nisexts.sexts module
"""

import sys
import types

from ..sexts import package_check

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

FAKE_NAME = 'nisext_improbable'
assert FAKE_NAME not in sys.modules
FAKE_MODULE = types.ModuleType('nisext_fake')


def test_package_check():
    # Try to use a required package - raise error
    assert_raises(RuntimeError, package_check, FAKE_NAME)
    # Optional, log.warn
    package_check(FAKE_NAME, optional=True)
    # Can also pass a string
    package_check(FAKE_NAME, optional='some-package')
    try:
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
    finally:
        del sys.modules[FAKE_NAME]


def test_package_check_setuptools():
    # If setuptools arg not None, missing package just adds it to arg
    assert_raises(RuntimeError, package_check, FAKE_NAME, setuptools_args=None)
    def pkg_chk_sta(*args, **kwargs):
        st_args = {}
        package_check(*args, setuptools_args=st_args, **kwargs)
        return st_args
    assert_equal(pkg_chk_sta(FAKE_NAME),
                 {'install_requires': ['nisext_improbable']})
    # Check that this gets appended to existing value
    old_sta = {'install_requires': ['something']}
    package_check(FAKE_NAME, setuptools_args=old_sta)
    assert_equal(old_sta,
                 {'install_requires': ['something', 'nisext_improbable']})
    # That existing value as string gets converted to a list
    old_sta = {'install_requires': 'something'}
    package_check(FAKE_NAME, setuptools_args=old_sta)
    assert_equal(old_sta,
                 {'install_requires': ['something', 'nisext_improbable']})
    # Optional, add to extras_require
    assert_equal(pkg_chk_sta(FAKE_NAME, optional='something'),
                 {'extras_require': {'something': ['nisext_improbable']}})
    # Check that this gets appended to existing value
    old_sta = {'extras_require': {'something': ['amodule']}}
    package_check(FAKE_NAME, optional='something', setuptools_args=old_sta)
    assert_equal(old_sta,
                 {'extras_require':
                  {'something': ['amodule', 'nisext_improbable']}})
    # That string gets converted to a list here too
    old_sta = {'extras_require': {'something': 'amodule'}}
    package_check(FAKE_NAME, optional='something', setuptools_args=old_sta)
    assert_equal(old_sta,
                 {'extras_require':
                  {'something': ['amodule', 'nisext_improbable']}})
    # But optional has to be a string if not empty and setuptools_args defined
    assert_raises(RuntimeError,
                  package_check, FAKE_NAME, optional=True, setuptools_args={})
    try:
        # Make a package
        sys.modules[FAKE_NAME] = FAKE_MODULE
        # No install_requires because we already have it
        assert_equal(pkg_chk_sta(FAKE_NAME), {})
        # A fake version still works
        FAKE_MODULE.__version__ = '0.2'
        assert_equal(pkg_chk_sta(FAKE_NAME, version='0.2'), {})
        # goes into install requires when version not good enough
        exp_spec = [FAKE_NAME + '>=0.3']
        assert_equal(pkg_chk_sta(FAKE_NAME, version='0.3'),
                     {'install_requires': exp_spec})
        # Unless optional in which case goes into extras_require
        package_check(FAKE_NAME, version='0.2', version_getter=lambda x: '0.2')
        assert_equal(
            pkg_chk_sta(FAKE_NAME, version='0.3', optional='afeature'),
            {'extras_require': {'afeature': exp_spec}})
        # Might do custom version check
        assert_equal(
            pkg_chk_sta(FAKE_NAME,
                        version='0.2',
                        version_getter=lambda x: '0.2'),
            {})
        # If the version check fails, put into requires
        bad_getter = lambda x: x.not_an_attribute
        exp_spec = [FAKE_NAME + '>=0.2']
        assert_equal(
            pkg_chk_sta(FAKE_NAME,
                        version='0.2',
                        version_getter=bad_getter),
            {'install_requires': exp_spec})
        # Likewise for optional dependency
        assert_equal(
            pkg_chk_sta(FAKE_NAME,
                        version='0.2',
                        optional='afeature',
                        version_getter=bad_getter),
            {'extras_require': {'afeature': [FAKE_NAME + '>=0.2']}})
    finally:
        del sys.modules[FAKE_NAME]

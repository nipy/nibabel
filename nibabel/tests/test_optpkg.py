""" Testing optpkg module
"""

import types
import sys
from distutils.version import LooseVersion

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal)


from nibabel.optpkg import optional_package
from nibabel.tripwire import TripWire, TripWireError


def assert_good(pkg_name, min_version=None):
    pkg, have_pkg, setup = optional_package(pkg_name, min_version=min_version)
    assert_true(have_pkg)
    assert_equal(sys.modules[pkg_name], pkg)
    assert_equal(setup(), None)


def assert_bad(pkg_name, min_version=None):
    pkg, have_pkg, setup = optional_package(pkg_name, min_version=min_version)
    assert_false(have_pkg)
    assert_true(isinstance(pkg, TripWire))
    assert_raises(TripWireError, getattr, pkg, 'a_method')
    assert_raises(SkipTest, setup)


def test_basic():
    # We always have os
    assert_good('os')
    # Subpackage
    assert_good('os.path')
    # We never have package _not_a_package
    assert_bad('_not_a_package')


def test_versions():
    fake_name = '_a_fake_package'
    fake_pkg = types.ModuleType(fake_name)
    assert_false('fake_pkg' in sys.modules)
    # Not inserted yet
    assert_bad(fake_name)
    try:
        sys.modules[fake_name] = fake_pkg
        # No __version__ yet
        assert_good(fake_name)  # With no version check
        assert_bad(fake_name, '1.0')
        # We can make an arbitrary callable to check version
        assert_good(fake_name, lambda pkg: True)
        # Now add a version
        fake_pkg.__version__ = '2.0'
        # We have fake_pkg > 1.0
        for min_ver in (None, '1.0', LooseVersion('1.0'), lambda pkg: True):
            assert_good(fake_name, min_ver)
        # We never have fake_pkg > 100.0
        for min_ver in ('100.0', LooseVersion('100.0'), lambda pkg: False):
            assert_bad(fake_name, min_ver)
        # Check error string for bad version
        pkg, _, _ = optional_package(fake_name, min_version='3.0')
        try:
            pkg.some_method
        except TripWireError as err:
            assert_equal(str(err),
                         'These functions need _a_fake_package version >= 3.0')
    finally:
        del sys.modules[fake_name]

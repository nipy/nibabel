""" Testing package info
"""

import nibabel as nib
from nibabel.pkg_info import cmp_pkg_version
from ..info import VERSION

from nose.tools import (assert_raises, assert_equal)


def test_pkg_info():
    """Smoke test nibabel.get_info()
    
    Hits:
        - nibabel.get_info
        - nibabel.pkg_info.get_pkg_info
        - nibabel.pkg_info.pkg_commit_hash
    """
    info = nib.get_info()


def test_version():
    # Test info version is the same as our own version
    assert_equal(nib.pkg_info.__version__, nib.__version__)


def test_fallback_version():
    """Test fallback version is up-to-date

    This should only fail if we fail to bump nibabel.info.VERSION immediately
    after release
    """
    assert (
        # dev version should be larger than tag+commit-githash
        cmp_pkg_version(VERSION) >= 0 or
        # Allow VERSION bump to lag releases by one commit
        VERSION == nib.__version__ + 'dev'), \
        "nibabel.info.VERSION does not match current tag information"


def test_cmp_pkg_version():
    # Test version comparator
    assert_equal(cmp_pkg_version(nib.__version__), 0)
    assert_equal(cmp_pkg_version('0.0'), -1)
    assert_equal(cmp_pkg_version('1000.1000.1'), 1)
    assert_equal(cmp_pkg_version(nib.__version__, nib.__version__), 0)
    for test_ver, pkg_ver, exp_out in (('1.0', '1.0', 0),
                                       ('1.0.0', '1.0', 0),
                                       ('1.0', '1.0.0', 0),
                                       ('1.1', '1.1', 0),
                                       ('1.2', '1.1', 1),
                                       ('1.1', '1.2', -1),
                                       ('1.1.1', '1.1.1', 0),
                                       ('1.1.2', '1.1.1', 1),
                                       ('1.1.1', '1.1.2', -1),
                                       ('1.1', '1.1dev', 1),
                                       ('1.1dev', '1.1', -1),
                                       ('1.2.1', '1.2.1rc1', 1),
                                       ('1.2.1rc1', '1.2.1', -1),
                                       ('1.2.1rc1', '1.2.1rc', 1),
                                       ('1.2.1rc', '1.2.1rc1', -1),
                                       ('1.2.1rc1', '1.2.1rc', 1),
                                       ('1.2.1rc', '1.2.1rc1', -1),
                                       ('1.2.1b', '1.2.1a', 1),
                                       ('1.2.1a', '1.2.1b', -1),
                                       ('1.2.0+1', '1.2', 1),
                                       ('1.2', '1.2.0+1', -1),
                                       ('1.2.1+1', '1.2.1', 1),
                                       ('1.2.1', '1.2.1+1', -1),
                                       ('1.2.1rc1+1', '1.2.1', -1),
                                       ('1.2.1', '1.2.1rc1+1', 1),
                                       ('1.2.1rc1+1', '1.2.1+1', -1),
                                       ('1.2.1+1', '1.2.1rc1+1', 1),
                                      ):
        assert_equal(cmp_pkg_version(test_ver, pkg_ver), exp_out)
    assert_raises(ValueError, cmp_pkg_version, 'foo.2')
    assert_raises(ValueError, cmp_pkg_version, 'foo.2', '1.0')
    assert_raises(ValueError, cmp_pkg_version, '1.0', 'foo.2')
    assert_raises(ValueError, cmp_pkg_version, '1')
    assert_raises(ValueError, cmp_pkg_version, 'foo')


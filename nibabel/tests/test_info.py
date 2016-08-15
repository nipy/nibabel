""" Testing info module
"""

import nibabel as nib
from nibabel import info
from nibabel.info import cmp_pkg_version

from nose.tools import (assert_raises, assert_equal)


def test_version():
    # Test info version is the same as our own version
    assert_equal(info.__version__, nib.__version__)


def test_cmp_pkg_version():
    # Test version comparator
    assert_equal(cmp_pkg_version(info.__version__), 0)
    assert_equal(cmp_pkg_version('0.0'), -1)
    assert_equal(cmp_pkg_version('1000.1000.1'), 1)
    assert_equal(cmp_pkg_version(info.__version__, info.__version__), 0)
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
                                      ):
        assert_equal(cmp_pkg_version(test_ver, pkg_ver), exp_out)
    assert_raises(ValueError, cmp_pkg_version, 'foo.2')
    assert_raises(ValueError, cmp_pkg_version, 'foo.2', '1.0')
    assert_raises(ValueError, cmp_pkg_version, '1.0', 'foo.2')
    assert_raises(ValueError, cmp_pkg_version, '1')
    assert_raises(ValueError, cmp_pkg_version, 'foo')

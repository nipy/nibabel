""" Testing package info
"""

import nibabel as nib
from nibabel.pkg_info import cmp_pkg_version
from ..info import VERSION

import pytest


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
    assert nib.pkg_info.__version__ == nib.__version__


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


def test_cmp_pkg_version_0():
    # Test version comparator
    assert cmp_pkg_version(nib.__version__) == 0
    assert cmp_pkg_version('0.0') == -1
    assert cmp_pkg_version('1000.1000.1') == 1
    assert cmp_pkg_version(nib.__version__, nib.__version__) == 0


@pytest.mark.parametrize("test_ver, pkg_ver, exp_out",
                         [
                             ('1.0', '1.0', 0),
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
                         ])
def test_cmp_pkg_version_1(test_ver, pkg_ver, exp_out):
    # Test version comparator
    assert cmp_pkg_version(test_ver, pkg_ver) == exp_out


@pytest.mark.parametrize("args", [['foo.2'], ['foo.2', '1.0'], ['1.0', 'foo.2'],
                         ['1'], ['foo']])
def test_cmp_pkg_version_error(args):
    with pytest.raises(ValueError):
        cmp_pkg_version(*args)


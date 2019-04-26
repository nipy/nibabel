from ..info import cmp_pkg_version
from ..testing import assert_raises


def test_minc_removed():
    if cmp_pkg_version('3.0.0dev') < 1:
        with assert_raises(ImportError):
            import nibabel.minc

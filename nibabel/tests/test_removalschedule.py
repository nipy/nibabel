from importlib import import_module
from ..info import cmp_pkg_version
from ..testing import assert_raises

SCHEDULE = [
    ('3.0.0', ('nibabel.minc', 'nibabel.checkwarns')),
    ]


def test_removals():
    for version, to_remove in SCHEDULE:
        if cmp_pkg_version(version) < 1:
            for module in to_remove:
                with assert_raises(ImportError, msg="Time to remove " + module):
                    import_module(module)

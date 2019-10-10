from ..pkg_info import cmp_pkg_version
from ..testing import assert_raises, assert_false

MODULE_SCHEDULE = [
    ('4.0.0', ['nibabel.trackvis']),
    ('3.0.0', ['nibabel.minc', 'nibabel.checkwarns']),
    # Verify that the test will be quiet if the schedule outlives the modules
    ('1.0.0', ['nibabel.neverexisted']),
    ]

OBJECT_SCHEDULE = [
    ('3.0.0', [('nibabel.testing', 'catch_warn_reset')]),
    # Verify that the test will be quiet if the schedule outlives the modules
    ('1.0.0', [('nibabel', 'neverexisted')]),
    ]

ATTRIBUTE_SCHEDULE = [
    ('5.0.0', [('nibabel.dataobj_images', 'DataobjImage', 'get_data')]),
    # Verify that the test will be quiet if the schedule outlives the modules
    ('1.0.0', [('nibabel', 'Nifti1Image', 'neverexisted')]),
    ]


def test_module_removal():
    for version, to_remove in MODULE_SCHEDULE:
        if cmp_pkg_version(version) < 1:
            for module in to_remove:
                with assert_raises(ImportError, msg="Time to remove " + module):
                    __import__(module)


def test_object_removal():
    for version, to_remove in OBJECT_SCHEDULE:
        if cmp_pkg_version(version) < 1:
            for module_name, obj in to_remove:
                try:
                    module = __import__(module_name)
                except ImportError:
                    continue
                assert_false(hasattr(module, obj), msg="Time to remove %s.%s" % (module_name, obj))


def test_attribute_removal():
    for version, to_remove in ATTRIBUTE_SCHEDULE:
        if cmp_pkg_version(version) < 1:
            for module_name, cls, attr in to_remove:
                try:
                    module = __import__(module_name)
                except ImportError:
                    continue
                try:
                    klass = getattr(module, cls)
                except AttributeError:
                    continue
                assert_false(hasattr(klass, attr),
                             msg="Time to remove %s.%s.%s" % (module_name, cls, attr))

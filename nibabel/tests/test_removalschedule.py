from ..pkg_info import cmp_pkg_version
import unittest
from unittest import mock
import pytest

MODULE_SCHEDULE = [
    ("5.0.0", ["nibabel.keywordonly"]),
    ("4.0.0", ["nibabel.trackvis"]),
    ("3.0.0", ["nibabel.minc", "nibabel.checkwarns"]),
    # Verify that the test will be quiet if the schedule outlives the modules
    ("1.0.0", ["nibabel.nosuchmod"]),
]

OBJECT_SCHEDULE = [
    ("5.0.0", [("nibabel.pydicom_compat", "dicom_test"),
               ("nibabel.onetime", "setattr_on_read")]),
    ("3.0.0", [("nibabel.testing", "catch_warn_reset")]),
    # Verify that the test will be quiet if the schedule outlives the modules
    ("1.0.0", [("nibabel.nosuchmod", "anyobj"), ("nibabel.nifti1", "nosuchobj")]),
]

ATTRIBUTE_SCHEDULE = [
    ("5.0.0", [("nibabel.dataobj_images", "DataobjImage", "get_data")]),
    # Verify that the test will be quiet if the schedule outlives the modules
    ("1.0.0", [("nibabel.nosuchmod", "anyobj", "anyattr"),
               ("nibabel.nifti1", "nosuchobj", "anyattr"),
               ("nibabel.nifti1", "Nifti1Image", "nosuchattr")]),
]


def _filter(schedule):
    return [entry for ver, entries in schedule if cmp_pkg_version(ver) < 1 for entry in entries]


def test_module_removal():
    for module in _filter(MODULE_SCHEDULE):
        with pytest.raises(ImportError):
            __import__(module)
            assert False, f"Time to remove {module}"


def test_object_removal():
    for module_name, obj in _filter(OBJECT_SCHEDULE):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        assert not hasattr(module, obj), f"Time to remove {module_name}.{obj}"


def test_attribute_removal():
    for module_name, cls, attr in _filter(ATTRIBUTE_SCHEDULE):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        try:
            klass = getattr(module, cls)
        except AttributeError:
            continue
        assert not hasattr(klass, attr), f"Time to remove {module_name}.{cls}.{attr}"


#
# Test the tests, making sure that we will get errors when the time comes
#

_sched = "nibabel.tests.test_removalschedule.{}_SCHEDULE".format


@mock.patch(_sched("MODULE"), [("3.0.0", ["nibabel.nifti1"])])
def test_unremoved_module():
    with pytest.raises(AssertionError):
        test_module_removal()


@mock.patch(_sched("OBJECT"), [("3.0.0", [("nibabel.nifti1", "Nifti1Image")])])
def test_unremoved_object():
    with pytest.raises(AssertionError):
        test_object_removal()


@mock.patch(_sched("ATTRIBUTE"), [("3.0.0", [("nibabel.nifti1", "Nifti1Image", "affine")])])
def test_unremoved_attr():
    with pytest.raises(AssertionError):
        test_attribute_removal()

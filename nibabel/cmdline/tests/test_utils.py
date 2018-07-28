# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""

from nose.tools import assert_equal
from numpy.testing import assert_raises

import nibabel as nib
import numpy as np
from nibabel.cmdline.utils import *
from nibabel.cmdline.diff import get_headers_diff, display_diff, main, get_data_diff
from os.path import (join as pjoin)
from nibabel.testing import data_path
from collections import OrderedDict
from six import StringIO


def test_table2string():
    assert_equal(table2string([["A", "B", "C", "D"], ["E", "F", "G", "H"]]), "A B C D\nE F G H\n")
    assert_equal(table2string([["Let's", "Make", "Tests", "And"], ["Have", "Lots", "Of", "Fun"],
                               ["With", "Python", "Guys", "!"]]), "Let's  Make  Tests And\n Have  Lots    Of  Fun"+
                                                                  "\n With Python  Guys  !\n")


def test_ap():
    assert_equal(ap([1, 2], "%2d"), " 1,  2")
    assert_equal(ap([1, 2], "%3d"), "  1,   2")
    assert_equal(ap([1, 2], "%-2d"), "1 , 2 ")
    assert_equal(ap([1, 2], "%d", "+"), "1+2")
    assert_equal(ap([1, 2, 3], "%d", "-"), "1-2-3")


def test_safe_get():
    class TestObject:
        def __init__(self, test=None):
            self.test = test

        def get_test(self):
            return self.test

    test = TestObject()
    test.test = 2

    assert_equal(safe_get(test, "test"), 2)
    assert_equal(safe_get(test, "failtest"), "-")


def test_get_headers_diff():
    fnames = [pjoin(data_path, f)
              for f in ('standard.nii.gz', 'example4d.nii.gz')]
    actual_difference = get_headers_diff([nib.load(f).header for f in fnames])
    expected_difference = OrderedDict([
        ("regular", [np.asarray("".encode("utf-8")), np.asarray("r".encode("utf-8"))]),
        ("dim_info", [np.asarray(0).astype(dtype="uint8"), np.asarray(57).astype(dtype="uint8")]),
        ("dim", [np.array([3, 4, 5, 7, 1, 1, 1, 1]).astype(dtype="int16"),
         np.array([  4, 128,  96,  24,   2,   1,   1,   1]).astype(dtype="int16")]),
        ("datatype", [np.array(2).astype(dtype="uint8"), np.array(4).astype(dtype="uint8")]),
        ("bitpix", [np.array(8).astype(dtype="uint8"), np.array(16).astype(dtype="uint8")]),
        ("pixdim", [np.array([ 1.,  1.,  3.,  2.,  1.,  1.,  1.,  1.]).astype(dtype="float32"), np.array(
            [ -1.00000000e+00,   2.00000000e+00, 2.00000000e+00,   2.19999909e+00, 2.00000000e+03,   1.00000000e+00,
            1.00000000e+00,   1.00000000e+00]).astype(dtype="float32")]),
        ("slice_end", [np.array(0).astype(dtype="uint8"), np.array(23).astype(dtype="uint8")]),
        ("xyzt_units", [np.array(0).astype(dtype="uint8"), np.array(10).astype(dtype="uint8")]),
        ("cal_max", [np.array(0.0).astype(dtype="float32"), np.asarray(1162.0).astype(dtype="float32")]),
        ("descrip", [np.array("".encode("utf-8")).astype(dtype="S80"),
                     np.array("FSL3.3\x00 v2.25 NIfTI-1 Single file format".encode("utf-8")).astype(dtype="S80")]),
        ("qform_code", [np.array(0).astype(dtype="int16"), np.array(1).astype(dtype="int16")]),
        ("sform_code", [np.array(2).astype(dtype="int16"), np.array(1).astype(dtype="int16")]),
        ("quatern_b", [np.array(0.0).astype(dtype="float32"),
                       np.array(-1.9451068140294884e-26).astype(dtype="float32")]),
        ("quatern_c", [np.array(0.0).astype(dtype="float32"), np.array(-0.9967085123062134).astype(dtype="float32")]),
        ("quatern_d", [np.array(0.0).astype(dtype="float32"), np.array(-0.0810687392950058).astype(dtype="float32")]),
        ("qoffset_x", [np.array(0.0).astype(dtype="float32"), np.array(117.8551025390625).astype(dtype="float32")]),
        ("qoffset_y", [np.array(0.0).astype(dtype="float32"), np.array(-35.72294235229492).astype(dtype="float32")]),
        ("qoffset_z", [np.array(0.0).astype(dtype="float32"), np.array(-7.248798370361328).astype(dtype="float32")]),
        ("srow_x", [np.array([ 1.,  0.,  0.,  0.]).astype(dtype="float32"),
                    np.array([ -2.00000000e+00,   6.71471565e-19,   9.08102451e-18,
                               1.17855103e+02]).astype(dtype="float32")]),
        ("srow_y", [np.array([ 0.,  3.,  0.,  0.]).astype(dtype="float32"),
         np.array([ -6.71471565e-19,   1.97371149e+00,  -3.55528235e-01, -3.57229424e+01]).astype(dtype="float32")]),
        ("srow_z", [np.array([ 0.,  0.,  2.,  0.]).astype(dtype="float32"),
                              np.array([  8.25548089e-18,   3.23207617e-01,   2.17108178e+00,
                                                     -7.24879837e+00]).astype(dtype="float32")])])

    np.testing.assert_equal(actual_difference, expected_difference)


def test_display_diff():
    bogus_names = ["hellokitty.nii.gz", "privettovarish.nii.gz"]

    dict_values = OrderedDict([
        ("datatype", [np.array(2).astype(dtype="uint8"), np.array(4).astype(dtype="uint8")]),
        ("bitpix", [np.array(8).astype(dtype="uint8"), np.array(16).astype(dtype="uint8")])
    ])

    expected_output = "These files are different.\n" + "Field          hellokitty.nii.gz" \
                                                       "                                      " \
                                                       "privettovarish.nii.gz                                  \n" \
                                                       "datatype       " \
                                                       "2                                                      " \
                                                       "4                                                      \n" \
                                                       "bitpix         " \
                                                       "8                                                      16" \
                                                       "                                                     " \
                                                       "\n"

    assert_equal(display_diff(bogus_names, dict_values), expected_output)


def test_get_data_diff():
    #  testing for identical files specifically as md5 may vary by computer
    test_names = [pjoin(data_path, f)
                  for f in ('standard.nii.gz', 'standard.nii.gz')]
    assert_equal(get_data_diff(test_names), [])


def test_main():
    test_names = [pjoin(data_path, f)
                  for f in ('standard.nii.gz', 'example4d.nii.gz')]
    expected_difference = OrderedDict([
        ("regular", [np.asarray("".encode("utf-8")), np.asarray("r".encode("utf-8"))]),
        ("dim_info", [np.asarray(0).astype(dtype="uint8"), np.asarray(57).astype(dtype="uint8")]),
        ("dim", [np.array([3, 4, 5, 7, 1, 1, 1, 1]).astype(dtype="int16"),
                 np.array([4, 128, 96, 24, 2, 1, 1, 1]).astype(dtype="int16")]),
        ("datatype", [np.array(2).astype(dtype="uint8"), np.array(4).astype(dtype="uint8")]),
        ("bitpix", [np.array(8).astype(dtype="uint8"), np.array(16).astype(dtype="uint8")]),
        ("pixdim", [np.array([1., 1., 3., 2., 1., 1., 1., 1.]).astype(dtype="float32"), np.array(
            [-1.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.19999909e+00, 2.00000000e+03, 1.00000000e+00,
             1.00000000e+00, 1.00000000e+00]).astype(dtype="float32")]),
        ("slice_end", [np.array(0).astype(dtype="uint8"), np.array(23).astype(dtype="uint8")]),
        ("xyzt_units", [np.array(0).astype(dtype="uint8"), np.array(10).astype(dtype="uint8")]),
        ("cal_max", [np.array(0.0).astype(dtype="float32"), np.asarray(1162.0).astype(dtype="float32")]),
        ("descrip", [np.array("".encode("utf-8")).astype(dtype="S80"),
                     np.array("FSL3.3\x00 v2.25 NIfTI-1 Single file format".encode("utf-8")).astype(dtype="S80")]),
        ("qform_code", [np.array(0).astype(dtype="int16"), np.array(1).astype(dtype="int16")]),
        ("sform_code", [np.array(2).astype(dtype="int16"), np.array(1).astype(dtype="int16")]),
        ("quatern_b", [np.array(0.0).astype(dtype="float32"),
                       np.array(-1.9451068140294884e-26).astype(dtype="float32")]),
        ("quatern_c", [np.array(0.0).astype(dtype="float32"), np.array(-0.9967085123062134).astype(dtype="float32")]),
        ("quatern_d", [np.array(0.0).astype(dtype="float32"), np.array(-0.0810687392950058).astype(dtype="float32")]),
        ("qoffset_x", [np.array(0.0).astype(dtype="float32"), np.array(117.8551025390625).astype(dtype="float32")]),
        ("qoffset_y", [np.array(0.0).astype(dtype="float32"), np.array(-35.72294235229492).astype(dtype="float32")]),
        ("qoffset_z", [np.array(0.0).astype(dtype="float32"), np.array(-7.248798370361328).astype(dtype="float32")]),
        ("srow_x", [np.array([1., 0., 0., 0.]).astype(dtype="float32"),
                    np.array([-2.00000000e+00, 6.71471565e-19, 9.08102451e-18,
                              1.17855103e+02]).astype(dtype="float32")]),
        ("srow_y", [np.array([0., 3., 0., 0.]).astype(dtype="float32"),
                    np.array([-6.71471565e-19, 1.97371149e+00, -3.55528235e-01, -3.57229424e+01]).astype(
                        dtype="float32")]),
        ("srow_z", [np.array([0., 0., 2., 0.]).astype(dtype="float32"),
                    np.array([8.25548089e-18, 3.23207617e-01, 2.17108178e+00,
                              -7.24879837e+00]).astype(dtype="float32")]),
        ('DATA(md5)', ['0a2576dd6badbb25bfb3b12076df986b', 'b0abbc492b4fd533b2c80d82570062cf'])])

    with assert_raises(SystemExit):
        np.testing.assert_equal(main(test_names, StringIO()), expected_difference)

    test_names_2 = [pjoin(data_path, f) for f in ('standard.nii.gz', 'standard.nii.gz')]

    with assert_raises(SystemExit):
        assert_equal(main(test_names_2, StringIO()), "These files are identical.")

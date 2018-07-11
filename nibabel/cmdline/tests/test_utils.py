# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""

from nose.tools import assert_equal

import nibabel as nib
import numpy as np
from nibabel.cmdline.utils import *
from nibabel.cmdline.diff import get_headers_diff
from os.path import (dirname, join as pjoin, abspath)


DATA_PATH = abspath(pjoin(dirname(__file__), '../../tests/data'))


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
    fnames = [pjoin(DATA_PATH, f)
              for f in ('standard.nii.gz', 'example4d.nii.gz')]
    actual_difference = get_headers_diff([nib.load(f).header for f in fnames])
    expected_difference = {
        "regular": ["".encode("utf-8"), "r".encode("utf-8")],
        "dim_info": [0, 57],
        "dim": np.array([[3, 4, 5, 7, 1, 1, 1, 1], [  4, 128,  96,  24,   2,   1,   1,   1]], "int16"),
        "datatype": [2, 4],
        "bitpix": [8, 16],
        "pixdim": np.array([[ 1.,  1.,  3.,  2.,  1.,  1.,  1.,  1.], [ -1.00000000e+00,   2.00000000e+00,
                                                                        2.00000000e+00,   2.19999909e+00,
                                                                        2.00000000e+03,   1.00000000e+00,
                                                                        1.00000000e+00,   1.00000000e+00]], "float32"),
        "slice_end": [0, 23],
        "xyzt_units": [0, 10],
        "cal_max": [0.0, 1162.0],
        "descrip": ["".encode("utf-8"), "FSL3.3\x00 v2.25 NIfTI-1 Single file format".encode("utf-8")],
        "qform_code": [0, 1],
        "sform_code": [2, 1],
        "quatern_b": [0.0, -1.9451068140294884e-26],
        "quatern_c": [0.0, -0.9967085123062134],
        "quatern_d": [0.0, -0.0810687392950058],
        "qoffset_x": [0.0, 117.8551025390625],
        "qoffset_y": [0.0, -35.72294235229492],
        "qoffset_z": [0.0, -7.248798370361328],
        "srow_x": np.array([[ 1.,  0.,  0.,  0.], [ -2.00000000e+00,   6.71471565e-19,   9.08102451e-18,
                                                    1.17855103e+02]], "float32"),
        "srow_y": np.array([[ 0.,  3.,  0.,  0.], [ -6.71471565e-19,   1.97371149e+00,  -3.55528235e-01,
                                                    -3.57229424e+01]], "float32"),
        "srow_z": np.array([[ 0.,  0.,  2.,  0.], [  8.25548089e-18,   3.23207617e-01,   2.17108178e+00,
                                                     -7.24879837e+00]], "float32")
    }

    assert_equal(actual_difference["regular"], expected_difference["regular"])
    assert_equal(actual_difference["dim_info"], expected_difference["dim_info"])
    np.testing.assert_array_equal(actual_difference["dim"], expected_difference["dim"])
    assert_equal(actual_difference["datatype"], expected_difference["datatype"])
    assert_equal(actual_difference["bitpix"], expected_difference["bitpix"])
    np.testing.assert_array_equal(actual_difference["pixdim"], expected_difference["pixdim"])
    assert_equal(actual_difference["slice_end"], expected_difference["slice_end"])
    assert_equal(actual_difference["xyzt_units"], expected_difference["xyzt_units"])
    assert_equal(actual_difference["cal_max"], expected_difference["cal_max"])
    assert_equal(actual_difference["descrip"], expected_difference["descrip"])
    assert_equal(actual_difference["qform_code"], expected_difference["qform_code"])
    assert_equal(actual_difference["sform_code"], expected_difference["sform_code"])
    assert_equal(actual_difference["quatern_b"], expected_difference["quatern_b"])
    assert_equal(actual_difference["quatern_c"], expected_difference["quatern_c"])
    assert_equal(actual_difference["quatern_d"], expected_difference["quatern_d"])
    assert_equal(actual_difference["qoffset_x"], expected_difference["qoffset_x"])
    assert_equal(actual_difference["qoffset_y"], expected_difference["qoffset_y"])
    assert_equal(actual_difference["qoffset_z"], expected_difference["qoffset_z"])
    np.testing.assert_array_equal(actual_difference["srow_x"], expected_difference["srow_x"])
    np.testing.assert_array_equal(actual_difference["srow_y"], expected_difference["srow_y"])
    np.testing.assert_array_equal(actual_difference["srow_z"], expected_difference["srow_z"])

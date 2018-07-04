# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

import nibabel as nib
from nibabel.cmdline.utils import *
from nibabel.cmdline.diff import diff_header_fields, diff_headers
from os.path import (dirname, join as pjoin, abspath, splitext, basename,
                     exists)


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


def test_diff_headers():
    fnames = [pjoin(DATA_PATH, f)
              for f in ('standard.nii.gz', 'example4d.nii.gz')]
    file_headers = [nib.load(f).header for f in fnames]
    headers = ['sizeof_hdr', 'data_type', 'db_name', 'extents', 'session_error', 'regular', 'dim_info', 'dim', 'intent_p1',
     'intent_p2', 'intent_p3', 'intent_code', 'datatype', 'bitpix', 'slice_start', 'pixdim', 'vox_offset', 'scl_slope',
     'scl_inter', 'slice_end', 'slice_code', 'xyzt_units', 'cal_max', 'cal_min', 'slice_duration', 'toffset', 'glmax',
     'glmin', 'descrip', 'aux_file', 'qform_code', 'sform_code', 'quatern_b', 'quatern_c', 'quatern_d', 'qoffset_x',
     'qoffset_y', 'qoffset_z', 'srow_x', 'srow_y', 'srow_z', 'intent_name', 'magic']

    assert_equal(diff_headers(file_headers, headers), ['regular', 'dim_info', 'dim', 'datatype', 'bitpix', 'pixdim',
                                                 'slice_end', 'xyzt_units', 'cal_max', 'descrip', 'qform_code',
                                                 'sform_code', 'quatern_b', 'quatern_c', 'quatern_d', 'qoffset_x',
                                                 'qoffset_y', 'qoffset_z', 'srow_x', 'srow_y', 'srow_z'])


def test_diff_header_fields():
    fnames = [pjoin(DATA_PATH, f)
              for f in ('standard.nii.gz', 'example4d.nii.gz')]
    file_headers = [nib.load(f).header for f in fnames]
    assert_equal(diff_header_fields("dim_info", file_headers), ['0', '57'])

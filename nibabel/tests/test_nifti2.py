# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for nifti2 reading package '''
from __future__ import division, print_function, absolute_import
import os

import numpy as np

from .. import nifti2
from ..nifti1 import (Nifti1Header, Nifti1PairHeader, Nifti1Extension,
                      Nifti1Extensions)
from ..nifti2 import (Nifti2Header, Nifti2PairHeader, Nifti2Image, Nifti2Pair)

from .test_nifti1 import (TestNifti1PairHeader, TestNifti1SingleHeader,
                          TestNifti1Pair, TestNifti1Image, TestNifti1General)

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from ..testing import data_path

header_file = os.path.join(data_path, 'nifti2.hdr')
image_file = os.path.join(data_path, 'example_nifti2.nii.gz')


class _Nifti2Mixin(object):
    example_file = header_file
    sizeof_hdr = Nifti2Header.sizeof_hdr
    quat_dtype = np.float64

    def test_freesurfer_hack(self):
        # Disable this check
        pass

    def test_eol_check(self):
        # Check checking of EOL check field
        HC = self.header_class
        hdr = HC()
        good_eol = (13, 10, 26, 10)
        assert_array_equal(hdr['eol_check'], good_eol)
        hdr['eol_check'] = 0
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_array_equal(fhdr['eol_check'], good_eol)
        assert_equal(message,
                     'EOL check all 0; '
                     'setting EOL check to 13, 10, 26, 10')
        hdr['eol_check'] = (13, 10, 0, 10)
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert_array_equal(fhdr['eol_check'], good_eol)
        assert_equal(message,
                     'EOL check not 0 or 13, 10, 26, 10; '
                     'data may be corrupted by EOL conversion; '
                     'setting EOL check to 13, 10, 26, 10')


class TestNifti2PairHeader(_Nifti2Mixin, TestNifti1PairHeader):
    header_class = Nifti2PairHeader
    example_file = header_file


class TestNifti2SingleHeader(_Nifti2Mixin, TestNifti1SingleHeader):
    header_class = Nifti2Header
    example_file = header_file


class TestNifti2Image(TestNifti1Image):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti2Image


class TestNifti2Image(TestNifti1Image):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti2Image


class TestNifti2Pair(TestNifti1Pair):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti2Pair


class TestNifti2General(TestNifti1General):
    """ Test class to test nifti2 in general

    Tests here which mix the pair and the single type, and that should only be
    run once (not for each type) because they are slow
    """
    single_class = Nifti2Image
    pair_class = Nifti2Pair
    module = nifti2
    example_file = image_file


def test_nifti12_conversion():
    shape = (2, 3, 4)
    dtype_type = np.int64
    ext1 = Nifti1Extension(6, b'My comment')
    ext2 = Nifti1Extension(6, b'Fresh comment')
    for in_type, out_type in ((Nifti1Header, Nifti2Header),
                              (Nifti1PairHeader, Nifti2Header),
                              (Nifti1PairHeader, Nifti2PairHeader),
                              (Nifti2Header, Nifti1Header),
                              (Nifti2PairHeader, Nifti1Header),
                              (Nifti2PairHeader, Nifti1PairHeader)):
        in_hdr = in_type()
        in_hdr.set_data_shape(shape)
        in_hdr.set_data_dtype(dtype_type)
        in_hdr.extensions[:] = [ext1, ext2]
        out_hdr = out_type.from_header(in_hdr)
        assert_equal(out_hdr.get_data_shape(), shape)
        assert_equal(out_hdr.get_data_dtype(), dtype_type)
        assert_equal(in_hdr.extensions, out_hdr.extensions)

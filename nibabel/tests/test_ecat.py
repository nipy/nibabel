# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

import os
import warnings

import numpy as np

from ..openers import Opener
from ..ecat import (EcatHeader, EcatSubHeader, EcatImage, read_mlist,
                    get_frame_order, get_series_framenumbers)

from unittest import TestCase
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import data_path, suppress_warnings, clear_and_catch_warnings
from ..tmpdirs import InTemporaryDirectory

from .test_wrapstruct import _TestWrapStructBase
from .test_fileslice import slicer_samples

ecat_file = os.path.join(data_path, 'tinypet.v')


class TestEcatHeader(_TestWrapStructBase):
    header_class = EcatHeader
    example_file = ecat_file

    def test_header_size(self):
        assert_equal(self.header_class.template_dtype.itemsize, 512)

    def test_empty(self):
        hdr = self.header_class()
        assert_true(len(hdr.binaryblock) == 512)
        assert_true(hdr['magic_number'] == b'MATRIX72')
        assert_true(hdr['sw_version'] == 74)
        assert_true(hdr['num_frames'] == 0)
        assert_true(hdr['file_type'] == 0)
        assert_true(hdr['ecat_calibration_factor'] == 1.0)

    def _set_something_into_hdr(self, hdr):
        # Called from test_bytes test method.  Specific to the header data type
        hdr['scan_start_time'] = 42

    def test_dtype(self):
        # dtype not specified in header, only in subheaders
        hdr = self.header_class()
        assert_raises(NotImplementedError, hdr.get_data_dtype)

    def test_header_codes(self):
        fid = open(ecat_file, 'rb')
        hdr = self.header_class()
        newhdr = hdr.from_fileobj(fid)
        fid.close()
        assert_true(newhdr.get_filetype() == 'ECAT7_VOLUME16')
        assert_equal(newhdr.get_patient_orient(),
                     'ECAT7_Unknown_Orientation')

    def test_update(self):
        hdr = self.header_class()
        assert_true(hdr['num_frames'] == 0)
        hdr['num_frames'] = 2
        assert_true(hdr['num_frames'] == 2)

    def test_from_eg_file(self):
        # Example header is big-endian
        with Opener(self.example_file) as fileobj:
            hdr = self.header_class.from_fileobj(fileobj, check=False)
        assert_equal(hdr.endianness, '>')


class TestEcatMlist(TestCase):
    header_class = EcatHeader
    example_file = ecat_file

    def test_mlist(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fid)
        mlist = read_mlist(fid, hdr.endianness)
        fid.seek(0)
        fid.seek(512)
        dat = fid.read(128 * 32)
        dt = np.dtype([('matlist', np.int32)])
        dt = dt.newbyteorder('>')
        mats = np.recarray(shape=(32, 4), dtype=dt, buf=dat)
        fid.close()
        # tests
        assert_true(mats['matlist'][0, 0] + mats['matlist'][0, 3] == 31)
        assert_true(get_frame_order(mlist)[0][0] == 0)
        assert_true(get_frame_order(mlist)[0][1] == 16842758.0)
        # test badly ordered mlist
        badordermlist = np.array([[1.68427540e+07, 3.00000000e+00,
                                   1.20350000e+04, 1.00000000e+00],
                                  [1.68427530e+07, 1.20360000e+04,
                                   2.40680000e+04, 1.00000000e+00],
                                  [1.68427550e+07, 2.40690000e+04,
                                   3.61010000e+04, 1.00000000e+00],
                                  [1.68427560e+07, 3.61020000e+04,
                                   4.81340000e+04, 1.00000000e+00],
                                  [1.68427570e+07, 4.81350000e+04,
                                   6.01670000e+04, 1.00000000e+00],
                                  [1.68427580e+07, 6.01680000e+04,
                                   7.22000000e+04, 1.00000000e+00]])
        with suppress_warnings():  # STORED order
            assert_true(get_frame_order(badordermlist)[0][0] == 1)

    def test_mlist_errors(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fid)
        hdr['num_frames'] = 6
        mlist = read_mlist(fid, hdr.endianness)
        mlist = np.array([[1.68427540e+07, 3.00000000e+00,
                           1.20350000e+04, 1.00000000e+00],
                          [1.68427530e+07, 1.20360000e+04,
                           2.40680000e+04, 1.00000000e+00],
                          [1.68427550e+07, 2.40690000e+04,
                           3.61010000e+04, 1.00000000e+00],
                          [1.68427560e+07, 3.61020000e+04,
                           4.81340000e+04, 1.00000000e+00],
                          [1.68427570e+07, 4.81350000e+04,
                           6.01670000e+04, 1.00000000e+00],
                          [1.68427580e+07, 6.01680000e+04,
                           7.22000000e+04, 1.00000000e+00]])
        with suppress_warnings():  # STORED order
            series_framenumbers = get_series_framenumbers(mlist)
        # first frame stored was actually 2nd frame acquired
        assert_true(series_framenumbers[0] == 2)
        order = [series_framenumbers[x] for x in sorted(series_framenumbers)]
        # true series order is [2,1,3,4,5,6], note counting starts at 1
        assert_true(order == [2, 1, 3, 4, 5, 6])
        mlist[0, 0] = 0
        with suppress_warnings():
            frames_order = get_frame_order(mlist)
        neworder = [frames_order[x][0] for x in sorted(frames_order)]
        assert_true(neworder == [1, 2, 3, 4, 5])
        with suppress_warnings():
            assert_raises(IOError, get_series_framenumbers, mlist)


class TestEcatSubHeader(TestCase):
    header_class = EcatHeader
    subhdr_class = EcatSubHeader
    example_file = ecat_file
    fid = open(example_file, 'rb')
    hdr = header_class.from_fileobj(fid)
    mlist = read_mlist(fid, hdr.endianness)
    subhdr = subhdr_class(hdr, mlist, fid)

    def test_subheader_size(self):
        assert_equal(self.subhdr_class._subhdrdtype.itemsize, 510)

    def test_subheader(self):
        assert_equal(self.subhdr.get_shape(), (10, 10, 3))
        assert_equal(self.subhdr.get_nframes(), 1)
        assert_equal(self.subhdr.get_nframes(),
                     len(self.subhdr.subheaders))
        assert_equal(self.subhdr._check_affines(), True)
        assert_array_almost_equal(np.diag(self.subhdr.get_frame_affine()),
                                  np.array([2.20241979, 2.20241979, 3.125, 1.]))
        assert_equal(self.subhdr.get_zooms()[0], 2.20241978764534)
        assert_equal(self.subhdr.get_zooms()[2], 3.125)
        assert_equal(self.subhdr._get_data_dtype(0), np.int16)
        #assert_equal(self.subhdr._get_frame_offset(), 1024)
        assert_equal(self.subhdr._get_frame_offset(), 1536)
        dat = self.subhdr.raw_data_from_fileobj()
        assert_equal(dat.shape, self.subhdr.get_shape())
        assert_equal(self.subhdr.subheaders[0]['scale_factor'].item(), 1.0)
        ecat_calib_factor = self.hdr['ecat_calibration_factor']
        assert_equal(ecat_calib_factor, 25007614.0)


class TestEcatImage(TestCase):
    image_class = EcatImage
    example_file = ecat_file
    img = image_class.load(example_file)

    def test_file(self):
        assert_equal(self.img.file_map['header'].filename,
                     self.example_file)
        assert_equal(self.img.file_map['image'].filename,
                     self.example_file)

    def test_save(self):
        tmp_file = 'tinypet_tmp.v'
        with InTemporaryDirectory():
            self.img.to_filename(tmp_file)
            other = self.image_class.load(tmp_file)
            assert_equal(self.img.get_data().all(), other.get_data().all())
            # Delete object holding reference to temporary file to make Windows
            # happier.
            del other

    def test_data(self):
        dat = self.img.get_data()
        assert_equal(dat.shape, self.img.shape)
        frame = self.img.get_frame(0)
        assert_array_equal(frame, dat[:, :, :, 0])

    def test_array_proxy(self):
        # Get the cached data copy
        dat = self.img.get_data()
        # Make a new one to test arrayproxy
        img = self.image_class.load(self.example_file)
        data_prox = img.dataobj
        data2 = np.array(data_prox)
        assert_array_equal(data2, dat)
        # Check it rereads
        data3 = np.array(data_prox)
        assert_array_equal(data3, dat)

    def test_array_proxy_slicing(self):
        # Test slicing of array proxy
        arr = self.img.get_data()
        prox = self.img.dataobj
        assert_true(prox.is_proxy)
        for sliceobj in slicer_samples(self.img.shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])

    def test_isolation(self):
        # Test image isolated from external changes to affine
        img_klass = self.image_class
        arr, aff, hdr, sub_hdr, mlist = (self.img.get_data(),
                                         self.img.affine,
                                         self.img.header,
                                         self.img.get_subheaders(),
                                         self.img.get_mlist())
        img = img_klass(arr, aff, hdr, sub_hdr, mlist)
        assert_array_equal(img.affine, aff)
        aff[0, 0] = 99
        assert_false(np.all(img.affine == aff))

    def test_float_affine(self):
        # Check affines get converted to float
        img_klass = self.image_class
        arr, aff, hdr, sub_hdr, mlist = (self.img.get_data(),
                                         self.img.affine,
                                         self.img.header,
                                         self.img.get_subheaders(),
                                         self.img.get_mlist())
        img = img_klass(arr, aff.astype(np.float32), hdr, sub_hdr, mlist)
        assert_equal(img.get_affine().dtype, np.dtype(np.float64))
        img = img_klass(arr, aff.astype(np.int16), hdr, sub_hdr, mlist)
        assert_equal(img.get_affine().dtype, np.dtype(np.float64))

    def test_data_regression(self):
        # Test whether data read has changed since 1.3.0
        # These values came from reading the example image using nibabel 1.3.0
        vals = dict(max=248750736458.0,
                    min=1125342630.0,
                    mean=117907565661.46666)
        data = self.img.get_data()
        assert_equal(data.max(), vals['max'])
        assert_equal(data.min(), vals['min'])
        assert_array_almost_equal(data.mean(), vals['mean'])

    def test_mlist_regression(self):
        # Test mlist is as same as for nibabel 1.3.0
        assert_array_equal(self.img.get_mlist(),
                           [[16842758, 3, 3011, 1]])


def test_from_filespec_deprecation():
    # Check from_filespec raises Deprecation
    with clear_and_catch_warnings() as w:
        warnings.simplefilter('always', DeprecationWarning)
        # No warning for standard load
        img_loaded = EcatImage.load(ecat_file)
        assert_equal(len(w), 0)
        # Warning for from_filespec
        img_speced = EcatImage.from_filespec(ecat_file)
        assert_equal(len(w), 1)
        assert_array_equal(img_loaded.get_data(), img_speced.get_data())

# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import with_statement

import os

import numpy as np

from ..py3k import asbytes

from ..volumeutils import native_code, swapped_code
from ..ecat import EcatHeader, EcatMlist, EcatSubHeader, EcatImage

from unittest import TestCase

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import data_path
from ..tmpdirs import InTemporaryDirectory

ecat_file = os.path.join(data_path, 'tinypet.v')

class TestEcatHeader(TestCase):
    header_class = EcatHeader
    example_file = ecat_file

    def test_header_size(self):
        assert_equal(self.header_class._dtype.itemsize, 512)

    def test_empty(self):
        hdr = self.header_class()
        assert_true(len(hdr.binaryblock) == 512)
        assert_true(hdr['magic_number'] == asbytes('MATRIX72'))
        assert_true(hdr['sw_version'] == 74)
        assert_true(hdr['num_frames'] == 0)
        assert_true(hdr['file_type'] == 0)
        assert_true(hdr['ecat_calibration_factor'] == 1.0)

    def test_dtype(self):
        #dtype not specified in header, only in subheaders
        hdr = self.header_class()
        assert_raises(NotImplementedError,
                            hdr.get_data_dtype)

    def test_header_codes(self):
        fid = open(ecat_file, 'rb')
        hdr = self.header_class()
        newhdr = hdr.from_fileobj(fid)
        fid.close()
        assert_true(newhdr.get_filetype() == 'ECAT7_VOLUME16')
        assert_equal(newhdr.get_patient_orient(),
                           'ECAT7_Unknown_Orientation')

    def test_copy(self):
        hdr = self.header_class()
        hdr2 = hdr.copy()
        assert_true(hdr == hdr2)
        assert_true(not hdr.binaryblock == hdr2._header_data.byteswap().tostring())
        assert_true(hdr.keys() == hdr2.keys())

    def test_update(self):
        hdr = self.header_class()
        assert_true(hdr['num_frames'] == 0)
        hdr['num_frames'] = 2
        assert_true(hdr['num_frames'] == 2)

    def test_endianness(self):
        # Default constructed header should be native
        native_hdr = self.header_class()
        assert_true(native_hdr.endianness == native_code)
        # Swapped constructed header should be swapped
        swapped_hdr = self.header_class(endianness=swapped_code)
        assert_true(swapped_hdr.endianness == swapped_code)
        # Example header is big-endian
        fid = open(ecat_file, 'rb')
        file_hdr = native_hdr.from_fileobj(fid)
        fid.close()
        assert_true(file_hdr.endianness == '>')


class TestEcatMlist(TestCase):
    header_class = EcatHeader
    mlist_class = EcatMlist
    example_file = ecat_file

    def test_mlist(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fid)
        mlist =  self.mlist_class(fid, hdr)
        fid.seek(0)
        fid.seek(512)
        dat=fid.read(128*32)
        dt = np.dtype([('matlist',np.int32)])
        dt = dt.newbyteorder('>')
        mats = np.recarray(shape=(32,4), dtype=dt,  buf=dat)
        fid.close()
        #tests
        assert_true(mats['matlist'][0,0] +  mats['matlist'][0,3] == 31)
        assert_true(mlist.get_frame_order()[0][0] == 0)
        assert_true(mlist.get_frame_order()[0][1] == 16842758.0)
        # test badly ordered mlist
        badordermlist = mlist
        badordermlist._mlist = np.array([[  1.68427540e+07,   3.00000000e+00,
                                            1.20350000e+04,   1.00000000e+00],
                                         [  1.68427530e+07,   1.20360000e+04,
                                            2.40680000e+04,   1.00000000e+00],
                                         [  1.68427550e+07,   2.40690000e+04,
                                            3.61010000e+04,   1.00000000e+00],
                                         [  1.68427560e+07,   3.61020000e+04,
                                            4.81340000e+04,   1.00000000e+00],
                                         [  1.68427570e+07,   4.81350000e+04,
                                            6.01670000e+04,   1.00000000e+00],
                                         [  1.68427580e+07,   6.01680000e+04,
                                            7.22000000e+04,   1.00000000e+00]])
        assert_true(badordermlist.get_frame_order()[0][0] == 1)

    def test_mlist_errors(self):
        fid = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fid)
        hdr['num_frames'] = 6
        mlist =  self.mlist_class(fid, hdr)    
        mlist._mlist = np.array([[  1.68427540e+07,   3.00000000e+00,
                                    1.20350000e+04,   1.00000000e+00],
                                 [  1.68427530e+07,   1.20360000e+04,
                                    2.40680000e+04,   1.00000000e+00],
                                 [  1.68427550e+07,   2.40690000e+04,
                                    3.61010000e+04,   1.00000000e+00],
                                 [  1.68427560e+07,   3.61020000e+04,
                                    4.81340000e+04,   1.00000000e+00],
                                 [  1.68427570e+07,   4.81350000e+04,
                                    6.01670000e+04,   1.00000000e+00],
                                 [  1.68427580e+07,   6.01680000e+04,
                                    7.22000000e+04,   1.00000000e+00]])        
        series_framenumbers = mlist.get_series_framenumbers()
        # first frame stored was actually 2nd frame acquired
        assert_true(series_framenumbers[0] == 2)
        order = [series_framenumbers[x] for x in sorted(series_framenumbers)]
        # true series order is [2,1,3,4,5,6], note counting starts at 1
        assert_true(order == [2, 1, 3, 4, 5, 6])
        mlist._mlist[0,0] = 0
        frames_order = mlist.get_frame_order()
        neworder =[frames_order[x][0] for x in sorted(frames_order)] 
        assert_true(neworder == [1, 2, 3, 4, 5])
        assert_raises(IOError,
                      mlist.get_series_framenumbers)
        
        

class TestEcatSubHeader(TestCase):
    header_class = EcatHeader
    mlist_class = EcatMlist
    subhdr_class = EcatSubHeader
    example_file = ecat_file
    fid = open(example_file, 'rb')
    hdr = header_class.from_fileobj(fid)
    mlist =  mlist_class(fid, hdr)
    subhdr = subhdr_class(hdr, mlist, fid)

    def test_subheader_size(self):
        assert_equal(self.subhdr_class._subhdrdtype.itemsize, 510)

    def test_subheader(self):
        assert_equal(self.subhdr.get_shape() , (10,10,3))
        assert_equal(self.subhdr.get_nframes() , 1)
        assert_equal(self.subhdr.get_nframes(),
                     len(self.subhdr.subheaders))
        assert_equal(self.subhdr._check_affines(), True)
        assert_array_almost_equal(np.diag(self.subhdr.get_frame_affine()),
                                  np.array([ 2.20241979, 2.20241979, 3.125,  1.]))
        assert_equal(self.subhdr.get_zooms()[0], 2.20241978764534)
        assert_equal(self.subhdr.get_zooms()[2], 3.125)
        assert_equal(self.subhdr._get_data_dtype(0),np.uint16)
        #assert_equal(self.subhdr._get_frame_offset(), 1024)
        assert_equal(self.subhdr._get_frame_offset(), 1536)
        dat = self.subhdr.raw_data_from_fileobj()
        assert_equal(dat.shape, self.subhdr.get_shape())
        scale_factor = self.subhdr.subheaders[0]['scale_factor']
        assert_equal(self.subhdr.subheaders[0]['scale_factor'].item(),1.0)
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
        assert_array_equal(frame, dat[:,:,:,0])

    def test_array_proxy(self):
        # Get the cached data copy
        dat = self.img.get_data()
        # Make a new one to test arrayproxy
        img = self.image_class.load(self.example_file)
        # Maybe we will promote _data to public, but I know this looks bad
        secret_data = img._data
        data2 = np.array(secret_data)
        assert_array_equal(data2, dat)
        # Check it rereads
        data3 = np.array(secret_data)
        assert_array_equal(data3, dat)

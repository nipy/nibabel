# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import os
from StringIO import StringIO
import numpy as np

from ..testing import assert_equal, assert_not_equal, \
    assert_true, assert_false, assert_raises

from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal)


from ..ecat import EcatHeader, EcatMlist, EcatSubHeader, EcatImage
import test_binary as tb
from ..testing import parametric, data_path, ParametricTestCase

ecat_file = os.path.join(data_path, 'tinypet.v')

class TestEcatHeader(ParametricTestCase):
    header_class = EcatHeader
    example_file = ecat_file

    def test_header_size(self):
        yield assert_equal(self.header_class._dtype.itemsize, 502)

    def test_empty(self):
        hdr = self.header_class()
        yield assert_true(len(hdr.binaryblock) == 502)
        yield assert_true(hdr['magic_number'] == 'MATRIX72')
        yield assert_true(hdr['sw_version'] == 74)
        yield assert_true(hdr['num_frames'] == 0)
        yield assert_true(hdr['file_type'] == 0)
        yield assert_true(hdr['ecat_calibration_factor'] == 1.0)

    def test_dtype(self):
        hdr = self.header_class()
        yield assert_raises(NotImplementedError,
                            hdr.get_data_dtype)

    def test_copy(self):
        hdr = self.header_class()
        hdr2 = hdr.copy()
        yield assert_true(hdr == hdr2)
        yield assert_true(not hdr.binaryblock == hdr2._header_data.byteswap().tostring())
        yield assert_true(hdr.keys() == hdr2.keys())

    def test_update(self):
        hdr = self.header_class()
        yield assert_true(hdr['num_frames'] == 0)
        hdr['num_frames'] = 2
        yield assert_true(hdr['num_frames'] == 2)
    
    def test_endianness(self):
        fid = open(ecat_file)
        hdr = self.header_class()
        newhdr = hdr.from_fileobj(fid)
        fid.close()
        yield assert_true(hdr.endianness == '<')
        yield assert_true(newhdr.endianness == '>')

class TestEcatMlist(ParametricTestCase):
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
        yield assert_true(mats['matlist'][0,0] +  mats['matlist'][0,3] == 31)
        yield assert_true(mlist.get_frame_order()[0][0] == 0)
        yield assert_true(mlist.get_frame_order()[0][1] == 16842758.0)

class TestEcatSubHeader(ParametricTestCase):
    header_class = EcatHeader
    mlist_class = EcatMlist
    subhdr_class = EcatSubHeader
    example_file = ecat_file    
    fid = open(example_file, 'rb')
    hdr = header_class.from_fileobj(fid)
    mlist =  mlist_class(fid, hdr)        
    subhdr = subhdr_class(hdr, mlist, fid)
    

    def test_subheader_size(self):
        yield assert_equal(self.subhdr_class._subhdrdtype.itemsize, 242)

    def test_subheader(self):
        yield assert_equal(self.subhdr.get_shape() , (10,10,3))
        yield assert_equal(self.subhdr.get_nframes() , 1)
        yield assert_equal(self.subhdr.get_nframes(),
                           len(self.subhdr.subheaders))
        yield assert_equal(self.subhdr._check_affines(), True)
        yield assert_array_almost_equal(np.diag(self.subhdr.get_frame_affine()),
                                        np.array([ 2.20241979, 2.20241979, 3.125,  1.]))
        yield assert_equal(self.subhdr.get_zooms()[0], 2.20241978764534)
        yield assert_equal(self.subhdr.get_zooms()[2], 3.125)
        yield assert_equal(self.subhdr._get_data_dtype(0),np.dtype('ushort'))
        yield assert_equal(self.subhdr._get_frame_offset(), 1536)
        dat = self.subhdr.raw_data_from_fileobj()
        yield assert_equal(dat.shape, self.subhdr.get_shape())
        scale_factor = self.subhdr.subheaders[0]['scale_factor']
        ecat_calib_factor = self.hdr['ecat_calibration_factor']
        scaled_dat = self.subhdr.data_from_fileobj()
        yield assert_array_equal(dat * scale_factor * ecat_calib_factor,
                                 scaled_dat)


class TestEcatImage(ParametricTestCase):
    image_class = EcatImage
    example_file = ecat_file
    img = image_class.load(example_file)

    def test_file(self):
        yield assert_equal(self.img.file_map['header'].filename,
                           self.example_file)
        yield assert_equal(self.img.file_map['image'].filename,
                           self.example_file)
        yield assert_raises(NotImplementedError,
                            self.img.to_filename,
                            'tmp.v')

    def test_data(self):
        dat = self.img.get_data()
        yield assert_equal(dat.shape, self.img.get_shape())
        frame = self.img.get_frame(0)
        yield assert_array_equal(frame, dat[:,:,:,0])
        
        

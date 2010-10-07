# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test binary header objects like Analyze, Nifti...

This is a root testing class, used in the Analyze and other tests as a
framework for all the tests common to the Analyze types

'''

from StringIO import StringIO

import numpy as np

from ..volumeutils import swapped_code, native_code, array_to_file
from ..spatialimages import HeaderDataError

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import (assert_equal, assert_true, assert_false,
                       assert_raises, assert_not_equal, parametric,
                       ParametricTestCase)


def _write_data(hdr, data, fileobj):
    # auxilary function to write data
    out_dtype = hdr.get_data_dtype()
    offset = hdr.get_data_offset()
    array_to_file(data, fileobj, out_dtype, offset)


class _TestBinaryHeader(ParametricTestCase):
    ''' Class implements tests for binary headers

    It serves as a base class for other binary header tests

    The underscore in the name prevents it from being run as a test
    directly and allows it to be used in sub-classes.
    '''
    header_class = None # overwrite with sub-classes

    def test_general_init(self):
        hdr = self.header_class()
        # binaryblock has length given by header data dtype
        binblock = hdr.binaryblock
        yield assert_equal(len(binblock), hdr.structarr.dtype.itemsize)
        # an empty header has shape (0,) - like an empty array
        # (np.array([]))
        yield assert_equal(hdr.get_data_shape(), (0,))
        # The affine is always homogenous 3D regardless of shape. The
        # default affine will have -1 as the X zoom iff default_x_flip
        # is True (which it is by default). We have to be careful of the
        # translations though - these arise from SPM's use of the origin
        # field, and the center of the image.
        yield assert_array_equal(np.diag(hdr.get_base_affine()),
                                 [-1,1,1,1])
        # But zooms only go with number of dimensions
        yield assert_equal(hdr.get_zooms(), (1.0,))
        # Endianness will be native by default for empty header
        yield assert_equal(hdr.endianness, native_code)
        # But you can change this if you want
        hdr = self.header_class(endianness='swapped')
        yield assert_equal(hdr.endianness, swapped_code)
        # Trying to read data from an empty header gives no data
        yield assert_equal(len(hdr.data_from_fileobj(StringIO())), 0)
        # Setting no data into an empty header results in - no data
        sfobj = StringIO()
        hdr.data_to_fileobj([], sfobj)
        yield assert_equal(sfobj.getvalue(), '')
        # Setting more data then there should be gives an error
        yield assert_raises(HeaderDataError,
                            hdr.data_to_fileobj,
                            np.zeros(3),
                            sfobj)
        # You can also pass in a check flag, without data this has no
        # effect
        hdr = self.header_class(check=False)
        
    def test_mappingness(self):
        hdr = self.header_class()
        yield assert_raises(ValueError,
                            hdr.__setitem__,
                            'nonexistent key',
                            0.1)
        hdr_dt = hdr.structarr.dtype
        keys = hdr.keys()
        yield assert_equal(keys, list(hdr))
        vals = hdr.values()
        items = hdr.items()
        yield assert_equal(keys, list(hdr_dt.names))
        for key, val in hdr.items():
            yield assert_array_equal(hdr[key], val)

    def test_str(self):
        pass

    def test_endianness(self):
        # endianness is a read only property
        ''' Its use in initialization tested in the init tests.
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization (or occasionally byteswapping the
        data) - but this is done via via the as_byteswapped method
        '''
        hdr = self.header_class()
        endianness = hdr.endianness
        yield assert_raises(AttributeError,
                            hdr.__setattr__,
                            'endianness',
                            '<')

    def test_endian_guess(self):
        # Check guesses of endian
        eh = self.header_class()
        yield assert_equal(eh.endianness, native_code)
        hdr_data = eh.structarr.copy()
        hdr_data = hdr_data.byteswap(swapped_code)
        eh_swapped = self.header_class(hdr_data.tostring())
        yield assert_equal(eh_swapped.endianness, swapped_code)

    def test_from_to_fileobj(self):
        hdr = self.header_class()
        str_io = StringIO()
        hdr.write_to(str_io)
        str_io.seek(0)
        hdr2 = self.header_class.from_fileobj(str_io)
        yield assert_equal(hdr2.endianness, native_code)
        yield assert_equal(hdr2.binaryblock, hdr.binaryblock)

    def test_binblock_is_file(self):
        # Checks that the binary string respresentation is the whole of the
        # header file.  This is true for Analyze types, but not true Nifti
        # single file headers, for example, because they will have extension
        # strings following.  More generally, there may be other perhaps
        # optional data after the binary block, in which case you will need to
        # override this test
        hdr = self.header_class()
        str_io = StringIO()
        hdr.write_to(str_io)
        assert_equal(str_io.getvalue(), hdr.binaryblock)


    def test_structarr(self):
        # structarr attribute also read only
        hdr = self.header_class()
        hdr_fs = hdr.structarr
        yield assert_raises(AttributeError,
                            hdr.__setattr__,
                            'structarr',
                            0)

    def test_binaryblock(self):
        # Test get of binaryblock
        hdr1 = self.header_class()
        bb = hdr1.binaryblock
        hdr2 = self.header_class(hdr1.binaryblock)
        yield assert_equal(hdr1, hdr2)
        yield assert_equal(hdr1.binaryblock, hdr2.binaryblock)
        # Do a set into the header, and try again
        hdr1.set_data_shape((1, 2, 3))
        hdr2 = self.header_class(hdr1.binaryblock)
        yield assert_equal(hdr1, hdr2)
        yield assert_equal(hdr1.binaryblock, hdr2.binaryblock)
        # Short and long binaryblocks give errors
        # (here set through init)
        bblen = len(hdr1.binaryblock)
        yield assert_raises(HeaderDataError,
                            self.header_class,
                            bb[:-1])
        yield assert_raises(HeaderDataError,
                            self.header_class,
                            bb + chr(0))
        # Checking set to true by default, and prevents nonsense being
        # set into the header. Completely zeros binary block always
        # (fairly) bad
        bb_bad = chr(0) * len(bb)
        yield assert_raises(HeaderDataError,
                            self.header_class,
                            bb_bad)
        # now slips past without check
        hdr = self.header_class(bb_bad, check=False)

    def test_data_shape_zooms_affine(self):
        hdr = self.header_class()
        for shape in ((1,2,3),(0,),(1,),(1,2),(1,2,3,4)):
            L = len(shape)
            hdr.set_data_shape(shape)
            if L:
                yield assert_equal(hdr.get_data_shape(), shape)
            else:
                yield assert_equal(hdr.get_data_shape(), (0,))
            # Default zoom - for 3D - is 1(())
            yield assert_equal(hdr.get_zooms(), (1,) * L)
            # errors if zooms do not match shape
            if len(shape):
                yield assert_raises(HeaderDataError, 
                                    hdr.set_zooms,
                                    (1,) * (L-1))
                # Errors for negative zooms
                yield assert_raises(HeaderDataError,
                                    hdr.set_zooms,
                                    (-1,) + (1,)*(L-1))
            yield assert_raises(HeaderDataError,
                                hdr.set_zooms,
                                (1,) * (L+1))
            # Errors for negative zooms
            yield assert_raises(HeaderDataError,
                                hdr.set_zooms,
                                (-1,) * L)
        # reducing the dimensionality of the array and then increasing
        # it again reverts the previously set zoom values to 1.0
        hdr = self.header_class()
        hdr.set_data_shape((1,2,3))
        hdr.set_zooms((4,5,6))
        yield assert_array_equal(hdr.get_zooms(), (4,5,6))
        hdr.set_data_shape((1,2))
        yield assert_array_equal(hdr.get_zooms(), (4,5))
        hdr.set_data_shape((1,2,3))
        yield assert_array_equal(hdr.get_zooms(), (4,5,1))
        # Setting affine changes zooms
        hdr.set_data_shape((1,2,3))
        hdr.set_zooms((1,1,1))
        # abs to allow for neurological / radiological flips
        yield assert_array_equal(np.diag(hdr.get_base_affine()),
                                 [-1,1,1,1])
        zooms = (4, 5, 6)
        affine = np.diag(zooms + (1,))
        # Setting zooms changes affine
        hdr.set_zooms((1,1,1))
        yield assert_array_equal(np.diag(hdr.get_base_affine()),
                                 [-1,1,1,1])

    def test_default_x_flip(self):
        hdr = self.header_class()
        hdr.default_x_flip = True
        hdr.set_data_shape((1,2,3))
        hdr.set_zooms((1,1,1))
        yield assert_array_equal(np.diag(hdr.get_base_affine()),
                                 [-1,1,1,1])
        hdr.default_x_flip = False
        # Check avoids translations
        yield assert_array_equal(np.diag(hdr.get_base_affine()),
                                 [1,1,1,1])

    def test_as_byteswapped(self):
        # Check byte swapping
        hdr = self.header_class()
        yield assert_equal(hdr.endianness, native_code)
        # same code just returns a copy
        hdr2 = hdr.as_byteswapped(native_code)
        yield assert_false(hdr is hdr2)
        # Different code gives byteswapped copy
        hdr_bs = hdr.as_byteswapped(swapped_code)
        yield assert_equal(hdr_bs.endianness, swapped_code)
        yield assert_not_equal(hdr.binaryblock, hdr_bs.binaryblock)
        # Note that contents is not rechecked on swap / copy
        class DC(self.header_class):
            def check_fix(self, *args, **kwargs):
                raise Exception
        yield assert_raises(Exception, DC, hdr.binaryblock)
        hdr = DC(hdr.binaryblock, check=False)
        hdr2 = hdr.as_byteswapped(native_code)
        hdr_bs = hdr.as_byteswapped(swapped_code)

    def test_data_dtype(self):
        # check getting and setting of data type
        # codes / types supported by all binary headers
        supported_types = ((2, np.uint8),
                           (4, np.int16),
                           (8, np.int32),
                           (16, np.float32),
                           (32, np.complex64),
                           (64, np.float64))
        # and unsupported - here using some labels instead
        unsupported_types = (np.void, 'none', 'all', 0)
        hdr = self.header_class()
        for code, npt in supported_types:
            # Can set with code value, or numpy dtype, both return the
            # dtype as output on get
            hdr.set_data_dtype(npt)
            yield assert_equal(hdr.get_data_dtype(), npt)
        for inp in unsupported_types:
            yield assert_raises(HeaderDataError,
                                hdr.set_data_dtype,
                                inp)

    def test_read_write_data(self):
        # Check reading and writing of data
        hdr = self.header_class()
        hdr.set_data_shape((1,2,3))
        hdr.set_data_dtype(np.float32)
        S = StringIO()
        data = np.arange(6, dtype=np.float64)
        # data have to be the right shape
        yield assert_raises(HeaderDataError,
                            hdr.data_to_fileobj, data, S)
        data = data.reshape((1,2,3))
        # and size
        yield assert_raises(HeaderDataError,
                            hdr.data_to_fileobj,
                            data[:,:,:-1], S)
        yield assert_raises(HeaderDataError,
                            hdr.data_to_fileobj,
                            data[:,:-1,:], S)
        # OK if so
        hdr.data_to_fileobj(data, S)
        # Read it back
        data_back = hdr.data_from_fileobj(S)
        # Should be about the same
        yield assert_array_almost_equal(data, data_back)
        # but with the header dtype, not the data dtype
        yield assert_equal(hdr.get_data_dtype(), data_back.dtype)
        # this is with native endian, not so for swapped
        S2 = StringIO()
        hdr2 = hdr.as_byteswapped()
        hdr2.set_data_dtype(np.float32)
        hdr2.set_data_shape((1,2,3))
        hdr2.data_to_fileobj(data, S)
        data_back2 = hdr2.data_from_fileobj(S)
        # Compares the same
        yield assert_array_almost_equal(data_back, data_back2)
        # Same dtype names
        yield assert_equal(data_back.dtype.name,
                           data_back2.dtype.name)
        # But not the same endianness
        yield assert_not_equal(data.dtype.byteorder,
                               data_back2.dtype.byteorder)
        # Try scaling down to integer
        hdr.set_data_dtype(np.uint8)
        S3 = StringIO()
        # Analyze header cannot do scaling, but, if not scaling,
        # AnalyzeHeader is OK
        _write_data(hdr, data, S3)
        data_back = hdr.data_from_fileobj(S3)
        yield assert_array_almost_equal(data, data_back)
        # But, the data won't always be same as input if not scaling
        data = np.arange(6, dtype=np.float64).reshape((1,2,3)) + 0.5
        _write_data(hdr, data, S3)
        data_back = hdr.data_from_fileobj(S3)
        yield assert_false(np.allclose(data, data_back))

    def test_empty_check(self):
        # Empty header should be error free
        hdr = self.header_class()
        hdr.check_fix(error_level=0)

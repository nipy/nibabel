# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test analyze headers

See test_binary.py for general binary header tests

This - basic - analyze header cannot encode full affines (only
diagonal affines), and cannot do integer scaling.

The inability to do affines raises the problem of whether the image is
neurological (left is left), or radiological (left is right).  In
general this is the problem of whether the affine should consider
proceeding within the data down an X line as being from left to right,
or right to left.

To solve this, we have a ``default_x_flip`` flag that can be True or
False.  True means assume radiological.

If the image is 3D, and the X, Y and Z zooms are x, y, and z, then::

    If default_x_flip is True::
        affine = np.diag((-x,y,z,1))
    else:
        affine = np.diag((x,y,z,1))

In our implementation, there is no way of saving this assumed flip
into the header.  One way of doing this, that we have not used, is to
allow negative zooms, in particular, negative X zooms.  We did not do
this because the image can be loaded with and without a default flip,
so the saved zoom will not constrain the affine.
'''

import os
import sys
import re
import logging
from StringIO import StringIO

import numpy as np

from ..py3k import BytesIO, asbytes
from ..spatialimages import (HeaderDataError, HeaderTypeError)
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..nifti1 import Nifti1Header
from ..loadsave import read_img_data
from .. import imageglobals

from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal)

from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path)

from . import test_binary as tb
from .test_binary import _write_data
from . import test_spatialimages as tsi

header_file = os.path.join(data_path, 'analyze.hdr')

PIXDIM0_MSG = 'pixdim[1,2,3] should be non-zero; setting 0 dims to 1'

def _log_chk(hdr, level):
    # utility function to check header checking / logging
    # If level == 0, this header should always be OK
    str_io = StringIO()
    logger = logging.getLogger('test.logger')
    handler = logging.StreamHandler(str_io)
    logger.addHandler(handler)
    str_io.truncate(0)
    hdrc = hdr.copy()
    if level == 0: # Should never log or raise error
        logger.setLevel(0)
        hdrc.check_fix(logger=logger, error_level=0)
        assert_equal(str_io.getvalue(), '')
        logger.removeHandler(handler)
        return hdrc, '', ()
    # Non zero level, test above and below threshold
    # Logging level above threshold, no log
    logger.setLevel(level+1)
    e_lev = level+1
    hdrc.check_fix(logger=logger, error_level=e_lev)
    assert_equal(str_io.getvalue(), '')
    # Logging level below threshold, log appears
    logger.setLevel(level+1)
    logger.setLevel(level-1)
    hdrc = hdr.copy()
    hdrc.check_fix(logger=logger, error_level=e_lev)
    assert_true(str_io.getvalue() != '')
    message = str_io.getvalue().strip()
    logger.removeHandler(handler)
    hdrc2 = hdr.copy()
    raiser = (HeaderDataError,
              hdrc2.check_fix,
              logger,
              level)
    return hdrc, message, raiser


class TestAnalyzeHeader(tb._TestBinaryHeader):
    header_class = AnalyzeHeader
    example_file = header_file

    def test_header_size(self):
        assert_equal(self.header_class._dtype.itemsize, 348)

    def test_empty(self):
        hdr = self.header_class()
        assert_true(len(hdr.binaryblock) == 348)
        assert_true(hdr['sizeof_hdr'] == 348)
        assert_true(np.all(hdr['dim'][1:] == 1))
        assert_true(hdr['dim'][0] == 0        )
        assert_true(np.all(hdr['pixdim'] == 1))
        assert_true(hdr['datatype'] == 16) # float32
        assert_true(hdr['bitpix'] == 32)

    def test_checks(self):
        # Test header checks
        hdr_t = self.header_class()
        def dxer(hdr):
            binblock = hdr.binaryblock
            return self.header_class.diagnose_binaryblock(binblock)
        assert_equal(dxer(hdr_t), '')
        hdr = hdr_t.copy()
        hdr['sizeof_hdr'] = 1
        assert_equal(dxer(hdr), 'sizeof_hdr should be 348')
        hdr = hdr_t.copy()
        hdr['datatype'] = 0
        assert_equal(dxer(hdr),
                           'data code 0 not supported\nbitpix '
                           'does not match datatype')
        hdr = hdr_t.copy()
        hdr['bitpix'] = 0
        assert_equal(dxer(hdr),
                           'bitpix does not match datatype')
        for i in (1,2,3):
            hdr = hdr_t.copy()
            hdr['pixdim'][i] = -1
            assert_equal(dxer(hdr),
                               'pixdim[1,2,3] should be positive')

    def test_log_checks(self):
        # Test logging, fixing, errors for header checking
        HC = self.header_class
        # magic
        hdr = HC()
        hdr['sizeof_hdr'] = 350 # severity 30
        fhdr, message, raiser = _log_chk(hdr, 30)
        assert_equal(fhdr['sizeof_hdr'], 348)
        assert_equal(message, 'sizeof_hdr should be 348; '
                           'set sizeof_hdr to 348')
        assert_raises(*raiser)
        # datatype not recognized
        hdr = HC()
        hdr['datatype'] = -1 # severity 40
        fhdr, message, raiser = _log_chk(hdr, 40)
        assert_equal(message, 'data code -1 not recognized; '
                           'not attempting fix')
        assert_raises(*raiser)
        # datatype not supported
        hdr['datatype'] = 255 # severity 40
        fhdr, message, raiser = _log_chk(hdr, 40)
        assert_equal(message, 'data code 255 not supported; '
                           'not attempting fix')
        assert_raises(*raiser)
        # bitpix
        hdr = HC()
        hdr['datatype'] = 16 # float32
        hdr['bitpix'] = 16 # severity 10
        fhdr, message, raiser = _log_chk(hdr, 10)
        assert_equal(fhdr['bitpix'], 32)
        assert_equal(message, 'bitpix does not match datatype; '
                           'setting bitpix to match datatype')
        assert_raises(*raiser)
        # pixdim positive
        hdr = HC()
        hdr['pixdim'][1] = -2 # severity 35
        fhdr, message, raiser = _log_chk(hdr, 35)
        assert_equal(fhdr['pixdim'][1], 2)
        assert_equal(message, 'pixdim[1,2,3] should be positive; '
                           'setting to abs of pixdim values')
        assert_raises(*raiser)
        hdr = HC()
        hdr['pixdim'][1] = 0 # severity 30
        fhdr, message, raiser = _log_chk(hdr, 30)
        assert_equal(fhdr['pixdim'][1], 1)
        assert_equal(message, PIXDIM0_MSG)
        assert_raises(*raiser)
        # both
        hdr = HC()
        hdr['pixdim'][1] = 0 # severity 30
        hdr['pixdim'][2] = -2 # severity 35
        fhdr, message, raiser = _log_chk(hdr, 35)
        assert_equal(fhdr['pixdim'][1], 1)
        assert_equal(fhdr['pixdim'][2], 2)
        assert_equal(message, 'pixdim[1,2,3] should be '
                           'non-zero and pixdim[1,2,3] should '
                           'be positive; setting 0 dims to 1 '
                           'and setting to abs of pixdim values')
        assert_raises(*raiser)

    def test_logger_error(self):
        # Check that we can reset the logger and error level
        HC = self.header_class
        hdr = HC()
        # Make a new logger
        str_io = StringIO()
        logger = logging.getLogger('test.logger')
        logger.setLevel(30) # defaultish level
        logger.addHandler(logging.StreamHandler(str_io))
        # Prepare an error
        hdr['pixdim'][1] = 0 # severity 30
        log_cache = imageglobals.logger, imageglobals.error_level
        try:
            # Check log message appears in new logger
            imageglobals.logger = logger
            hdr.copy().check_fix()
            assert_equal(str_io.getvalue(), PIXDIM0_MSG + '\n')
            # Check that error_level in fact causes error to be raised
            imageglobals.error_level = 30
            assert_raises(HeaderDataError, hdr.copy().check_fix)
        finally:
            imageglobals.logger, imageglobals.error_level = log_cache

    def test_datatype(self):
        ehdr = self.header_class()
        codes = self.header_class._data_type_codes
        for code in codes.value_set():
            npt = codes.type[code]
            if npt is np.void:
                assert_raises(
                       HeaderDataError,
                       ehdr.set_data_dtype,
                       code)
                continue
            dt = codes.dtype[code]
            ehdr.set_data_dtype(npt)
            assert_true(ehdr['datatype'] == code)
            assert_true(ehdr['bitpix'] == dt.itemsize*8)
            ehdr.set_data_dtype(code)
            assert_true(ehdr['datatype'] == code)
            ehdr.set_data_dtype(dt)
            assert_true(ehdr['datatype'] == code)

    def test_from_eg_file(self):
        fileobj = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fileobj, check=False)
        assert_equal(hdr.endianness, '>')
        assert_equal(hdr['sizeof_hdr'], 348)

    def test_orientation(self):
        # Test flips
        hdr = self.header_class()
        assert_true(hdr.default_x_flip)
        hdr.set_data_shape((3,5,7))
        hdr.set_zooms((4,5,6))
        aff = np.diag((-4,5,6,1))
        aff[:3,3] = np.array([1,2,3]) * np.array([-4,5,6]) * -1
        assert_array_equal(hdr.get_base_affine(), aff)
        hdr.default_x_flip = False
        assert_false(hdr.default_x_flip)
        aff[0]*=-1
        assert_array_equal(hdr.get_base_affine(), aff)

    def test_str(self):
        hdr = self.header_class()
        # Check something returns from str
        S = hdr.__str__()
        assert_true(len(S)>0)
        # check the datacode recoding
        rexp = re.compile('^datatype +: float32', re.MULTILINE)
        assert_true(rexp.search(S) is not None)

    def test_from_header(self):
        # check from header class method.
        klass = self.header_class
        empty = klass.from_header()
        assert_equal(klass(), empty)
        empty = klass.from_header(None)
        assert_equal(klass(), empty)
        hdr = klass()
        hdr.set_data_dtype(np.float64)
        hdr.set_data_shape((1,2,3))
        hdr.set_zooms((3.0, 2.0, 1.0))
        copy = klass.from_header(hdr)
        assert_equal(hdr, copy)
        assert_false(hdr is copy)
        class C(object):
            def get_data_dtype(self): return np.dtype('i2')
            def get_data_shape(self): return (5,4,3)
            def get_zooms(self): return (10.0, 9.0, 8.0)
        converted = klass.from_header(C())
        assert_true(isinstance(converted, klass))
        assert_equal(converted.get_data_dtype(), np.dtype('i2'))
        assert_equal(converted.get_data_shape(), (5,4,3))
        assert_equal(converted.get_zooms(), (10.0,9.0,8.0))

    def test_base_affine(self):
        klass = self.header_class
        hdr = klass()
        hdr.set_data_shape((3, 5, 7))
        hdr.set_zooms((3, 2, 1))
        assert_true(hdr.default_x_flip)
        assert_array_almost_equal(
            hdr.get_base_affine(),
            [[-3.,  0.,  0.,  3.],
             [ 0.,  2.,  0., -4.],
             [ 0.,  0.,  1., -3.],
             [ 0.,  0.,  0.,  1.]])
        hdr.set_data_shape((3, 5))
        assert_array_almost_equal(
            hdr.get_base_affine(),
            [[-3.,  0.,  0.,  3.],
             [ 0.,  2.,  0., -4.],
             [ 0.,  0.,  1., -0.],
             [ 0.,  0.,  0.,  1.]])
        hdr.set_data_shape((3, 5, 7))
        assert_array_almost_equal(
            hdr.get_base_affine(),
            [[-3.,  0.,  0.,  3.],
             [ 0.,  2.,  0., -4.],
             [ 0.,  0.,  1., -3.],
             [ 0.,  0.,  0.,  1.]])


def test_best_affine():
    hdr = AnalyzeHeader()
    hdr.set_data_shape((3,5,7))
    hdr.set_zooms((4,5,6))
    assert_array_equal(hdr.get_base_affine(), hdr.get_best_affine())


def test_scaling():
    # Test integer scaling from float
    # Analyze headers cannot do float-integer scaling '''
    hdr = AnalyzeHeader()
    assert_true(hdr.default_x_flip)
    shape = (1,2,3)
    hdr.set_data_shape(shape)
    hdr.set_data_dtype(np.float32)
    data = np.ones(shape, dtype=np.float64)
    S = BytesIO()
    # Writing to float datatype doesn't need scaling
    hdr.data_to_fileobj(data, S)
    rdata = hdr.data_from_fileobj(S)
    assert_true(np.allclose(data, rdata))
    # Writing to integer datatype does, and raises an error
    hdr.set_data_dtype(np.int32)
    assert_raises(HeaderTypeError,
                        hdr.data_to_fileobj,
                        data, BytesIO())
    # unless we aren't scaling, in which case we convert the floats to
    # integers and write
    _write_data(hdr, data, S)
    rdata = hdr.data_from_fileobj(S)
    assert_true(np.allclose(data, rdata))
    # This won't work for floats that aren't close to integers
    data_p5 = data + 0.5
    _write_data(hdr, data_p5, S)
    rdata = hdr.data_from_fileobj(S)
    assert_false(np.allclose(data_p5, rdata))


def test_slope_inter():
    hdr = AnalyzeHeader()
    assert_equal(hdr.get_slope_inter(), (None, None))
    for slinter in ((None,),
                    (None, None),
                    (1.0,),
                    (1.0, None),
                    (None, 0),
                    (1.0, 0)):
        hdr.set_slope_inter(*slinter)
        assert_equal(hdr.get_slope_inter(), (None, None))
    assert_raises(HeaderTypeError, hdr.set_slope_inter, 1.1)
    assert_raises(HeaderTypeError, hdr.set_slope_inter, 1.0, 0.1)


def test_data_code_error():
    # test analyze raising error for unsupported codes
    hdr = Nifti1Header()
    hdr['datatype'] = 256
    assert_raises(HeaderDataError, AnalyzeHeader.from_header, hdr)


class TestAnalyzeImage(tsi.TestSpatialImage):
    image_class = AnalyzeImage

    def test_data_hdr_cache(self):
        # test the API for loaded images, such that the data returned
        # from img.get_data() is not affected by subsequent changes to
        # the header.
        IC = self.image_class
        # save an image to a file map
        fm = IC.make_file_map()
        for key, value in fm.items():
            fm[key].fileobj = BytesIO()
        shape = (2,3,4)
        data = np.arange(24, dtype=np.int8).reshape(shape)
        affine = np.eye(4)
        hdr = IC.header_class()
        hdr.set_data_dtype(np.int16)
        img = IC(data, affine, hdr)
        img.to_file_map(fm)
        img2 = IC.from_file_map(fm)
        assert_equal(img2.shape, shape)
        assert_equal(img2.get_data_dtype().type, np.int16)
        hdr = img2.get_header()
        hdr.set_data_shape((3,2,2))
        assert_equal(hdr.get_data_shape(), (3,2,2))
        hdr.set_data_dtype(np.uint8)
        assert_equal(hdr.get_data_dtype(), np.dtype(np.uint8))
        assert_array_equal(img2.get_data(), data)
        # now check read_img_data function - here we do see the changed
        # header
        sc_data = read_img_data(img2)
        assert_equal(sc_data.shape, (3,2,2))
        us_data = read_img_data(img2, prefer='unscaled')
        assert_equal(us_data.shape, (3,2,2))

    def test_affine_44(self):
        IC = self.image_class
        shape = (2,3,4)
        data = np.arange(24, dtype=np.int16).reshape(shape)
        affine = np.diag([2, 3, 4, 1])
        # OK - affine correct shape
        img = IC(data, affine)
        assert_array_equal(affine, img.get_affine())
        # OK - affine can be array-like
        img = IC(data, affine.tolist())
        assert_array_equal(affine, img.get_affine())
        # Not OK - affine wrong shape
        assert_raises(ValueError, IC, data, np.diag([2, 3, 4]))


def test_unsupported():
    # analyze does not support uint32
    img_klass = AnalyzeImage
    data = np.arange(24, dtype=np.int32).reshape((2,3,4))
    affine = np.eye(4)
    data = np.arange(24, dtype=np.uint32).reshape((2,3,4))
    assert_raises(HeaderDataError,
                  AnalyzeImage,
                  data,
                  affine)





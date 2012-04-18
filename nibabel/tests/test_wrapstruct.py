# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test binary header objects

This is a root testing class, used in the Analyze and other tests as a
framework for all the tests common to the Analyze types

Refactoring TODO maybe
----------------------

binaryblock
diagnose_binaryblock

-> bytes, diagnose_bytes

With deprecation warnings

_field_recoders -> field_recoders
'''
import logging

import numpy as np

from ..wrapstruct import WrapStructError, WrapStruct
from ..batteryrunners import Report

from ..py3k import BytesIO, StringIO, asbytes, ZEROB
from ..volumeutils import swapped_code, native_code, Recoder
from ..spatialimages import HeaderDataError
from .. import imageglobals

from unittest import TestCase

from numpy.testing import assert_array_equal

from ..testing import (assert_equal, assert_true, assert_false,
                       assert_raises, assert_not_equal)


class MyWrapStruct(WrapStruct):
    """ An example wrapped struct class """
    _field_recoders = {} # for recoding values for str
    template_dtype = np.dtype([('an_integer', 'i2'), ('a_str', 'S10')])

    @classmethod
    def guessed_endian(klass, hdr):
        if hdr['an_integer'] < 256:
            return native_code
        return swapped_code

    @classmethod
    def default_structarr(klass, endianness=None):
        structarr = super(MyWrapStruct, klass).default_structarr(endianness)
        structarr['an_integer'] = 1
        structarr['a_str'] = asbytes('a string')
        return structarr

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return (klass._chk_integer,
                klass._chk_string)

    def get_value_label(self, fieldname):
        if not fieldname in self._field_recoders:
            raise ValueError('%s not a coded field' % fieldname)
        code = int(self._structarr[fieldname])
        return self._field_recoders[fieldname].label[code]

    ''' Check functions in format expected by BatteryRunner class '''
    @staticmethod
    def _chk_integer(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['an_integer'] == 1:
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'an_integer should be 1'
        if fix:
            hdr['an_integer'] = 1
            rep.fix_msg = 'set an_integer to 1'
        return hdr, rep

    @staticmethod
    def _chk_string(hdr, fix=False):
        rep = Report(HeaderDataError)
        hdr_str = str(hdr['a_str'])
        if hdr_str.lower() == hdr_str:
            return hdr, rep
        rep.problem_level = 20
        rep.problem_msg = 'a_str should be lower case'
        if fix:
            hdr['a_str'] = hdr_str.lower()
            rep.fix_msg = 'set a_str to lower case'
        return hdr, rep


class _TestWrapStructBase(TestCase):
    ''' Class implements base tests for binary headers

    It serves as a base class for other binary header tests
    '''
    header_class = None

    def test_general_init(self):
        hdr = self.header_class()
        # binaryblock has length given by header data dtype
        binblock = hdr.binaryblock
        assert_equal(len(binblock), hdr.structarr.dtype.itemsize)
        # Endianness will be native by default for empty header
        assert_equal(hdr.endianness, native_code)
        # But you can change this if you want
        hdr = self.header_class(endianness='swapped')
        assert_equal(hdr.endianness, swapped_code)
        # You can also pass in a check flag, without data this has no
        # effect
        hdr = self.header_class(check=False)

    def _set_something_into_hdr(self, hdr):
        # Called from test_bytes test method.  Specific to the header data type
        raise NotImplementedError('Not in base type')

    def test__eq__(self):
        # Test equal and not equal
        hdr1 = self.header_class()
        hdr2 = self.header_class()
        assert_equal(hdr1, hdr2)
        self._set_something_into_hdr(hdr1)
        assert_not_equal(hdr1, hdr2)
        self._set_something_into_hdr(hdr2)
        assert_equal(hdr1, hdr2)
        # Check byteswapping maintains equality
        hdr3 = hdr2.as_byteswapped()
        assert_equal(hdr2, hdr3)
        # Check comparing to funny thing says no
        assert_not_equal(hdr1, None)
        assert_not_equal(hdr1, 1)

    def test_to_from_fileobj(self):
        # Successful write using write_to
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        str_io.seek(0)
        hdr2 = self.header_class.from_fileobj(str_io)
        assert_equal(hdr2.endianness, native_code)
        assert_equal(hdr2.binaryblock, hdr.binaryblock)

    def test_mappingness(self):
        hdr = self.header_class()
        assert_raises(ValueError,
                    hdr.__setitem__,
                    'nonexistent key',
                    0.1)
        hdr_dt = hdr.structarr.dtype
        keys = hdr.keys()
        assert_equal(keys, list(hdr))
        vals = hdr.values()
        assert_equal(len(vals), len(keys))
        assert_equal(keys, list(hdr_dt.names))
        for key, val in hdr.items():
            assert_array_equal(hdr[key], val)
        # verify that .get operates as destined
        assert_equal(hdr.get('nonexistent key'), None)
        assert_equal(hdr.get('nonexistent key', 'default'), 'default')
        assert_equal(hdr.get(keys[0]), vals[0])
        assert_equal(hdr.get(keys[0], 'default'), vals[0])

    def test_endianness_ro(self):
        # endianness is a read only property
        ''' Its use in initialization tested in the init tests.
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization (or occasionally byteswapping the
        data) - but this is done via via the as_byteswapped method
        '''
        hdr = self.header_class()
        assert_raises(AttributeError, hdr.__setattr__, 'endianness', '<')

    def test_endian_guess(self):
        # Check guesses of endian
        eh = self.header_class()
        assert_equal(eh.endianness, native_code)
        hdr_data = eh.structarr.copy()
        hdr_data = hdr_data.byteswap(swapped_code)
        eh_swapped = self.header_class(hdr_data.tostring())
        assert_equal(eh_swapped.endianness, swapped_code)

    def test_binblock_is_file(self):
        # Checks that the binary string respresentation is the whole of the
        # header file.  This is true for Analyze types, but not true Nifti
        # single file headers, for example, because they will have extension
        # strings following.  More generally, there may be other perhaps
        # optional data after the binary block, in which case you will need to
        # override this test
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert_equal(str_io.getvalue(), hdr.binaryblock)

    def test_structarr(self):
        # structarr attribute also read only
        hdr = self.header_class()
        # Just check we can get structarr
        _ = hdr.structarr
        # That it's read only
        assert_raises(AttributeError, hdr.__setattr__, 'structarr', 0)

    def log_chk(self, hdr, level):
        # utility method to check header checking / logging
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

    def test_bytes(self):
        # Test get of bytes
        hdr1 = self.header_class()
        bb = hdr1.binaryblock
        hdr2 = self.header_class(hdr1.binaryblock)
        assert_equal(hdr1, hdr2)
        assert_equal(hdr1.binaryblock, hdr2.binaryblock)
        # Do a set into the header, and try again.  The specifics of 'setting
        # something' will depend on the nature of the bytes object
        self._set_something_into_hdr(hdr1)
        hdr2 = self.header_class(hdr1.binaryblock)
        assert_equal(hdr1, hdr2)
        assert_equal(hdr1.binaryblock, hdr2.binaryblock)
        # Short and long binaryblocks give errors
        # (here set through init)
        assert_raises(WrapStructError,
                      self.header_class,
                      bb[:-1])
        assert_raises(WrapStructError,
                      self.header_class,
                      bb + ZEROB)
        # Checking set to true by default, and prevents nonsense being
        # set into the header. Completely zeros binary block always
        # (fairly) bad
        bb_bad = ZEROB * len(bb)
        assert_raises(HeaderDataError, self.header_class, bb_bad)
        # now slips past without check
        _ = self.header_class(bb_bad, check=False)

    def test_as_byteswapped(self):
        # Check byte swapping
        hdr = self.header_class()
        assert_equal(hdr.endianness, native_code)
        # same code just returns a copy
        hdr2 = hdr.as_byteswapped(native_code)
        assert_false(hdr is hdr2)
        # Different code gives byteswapped copy
        hdr_bs = hdr.as_byteswapped(swapped_code)
        assert_equal(hdr_bs.endianness, swapped_code)
        assert_not_equal(hdr.binaryblock, hdr_bs.binaryblock)
        # Note that contents is not rechecked on swap / copy
        class DC(self.header_class):
            def check_fix(self, *args, **kwargs):
                raise Exception
        assert_raises(Exception, DC, hdr.binaryblock)
        hdr = DC(hdr.binaryblock, check=False)
        hdr2 = hdr.as_byteswapped(native_code)
        hdr_bs = hdr.as_byteswapped(swapped_code)

    def test_empty_check(self):
        # Empty header should be error free
        hdr = self.header_class()
        hdr.check_fix(error_level=0)

    def _dxer(self, hdr):
        # Return diagnostics on bytes in `hdr`
        binblock = hdr.binaryblock
        return self.header_class.diagnose_binaryblock(binblock)

    def test_get_value_label(self):
        hdr = self.header_class()
        original_recoders = hdr._field_recoders
        # Key not existing raises error
        assert_true('improbable' not in original_recoders)
        assert_raises(ValueError, hdr.get_value_label, 'improbable')
        new_recoders = {}
        hdr._field_recoders = new_recoders
        # Even if there is a recoder
        assert_true('improbable' not in hdr.keys())
        rec = Recoder([[0, 'fullness of heart']], ('code', 'label'))
        hdr._field_recoders['improbable'] = rec
        assert_raises(ValueError, hdr.get_value_label, 'improbable')
        # If the key exists in the structure, and is intable, then we can recode
        for key, value in hdr.items():
            # No recoder at first
            assert_raises(ValueError, hdr.get_value_label, 0)
            try:
                code = int(value)
            except (ValueError, TypeError):
                pass
            else: # codeable
                rec = Recoder([[code, 'fullness of heart']], ('code', 'label'))
                hdr._field_recoders[key] = rec
                assert_equal(hdr.get_value_label(key), 'fullness of heart')

    def test_str(self):
        hdr = self.header_class()
        # Check something returns from str
        s1 = str(hdr)
        assert_true(len(s1) > 0)


class TestWrapStruct(_TestWrapStructBase):
    """ Test fake binary header defined at top of module """
    header_class = MyWrapStruct

    def _set_something_into_hdr(self, hdr):
        # Called from test_bytes test method.  Specific to the header data type
        hdr['a_str'] = 'reggie'

    def test_empty(self):
        # Test contents of default header
        hdr = self.header_class()
        assert_equal(hdr['an_integer'], 1)
        assert_equal(hdr['a_str'], asbytes('a string'))

    def test_str(self):
        hdr = self.header_class()
        s1 = str(hdr)
        assert_true(len(s1) > 0)
        assert_true('fullness of heart' not in s1)
        rec = Recoder([[1, 'fullness of heart']], ('code', 'label'))
        hdr._field_recoders['an_integer'] = rec
        s2 = str(hdr)
        assert_true('fullness of heart' in s2)

    def test_checks(self):
        # Test header checks
        hdr_t = self.header_class()
        # _dxer just returns the diagnostics as a string
        # Default hdr is OK
        assert_equal(self._dxer(hdr_t), '')
        # An integer should be 1
        hdr = hdr_t.copy()
        hdr['an_integer'] = 2
        assert_equal(self._dxer(hdr), 'an_integer should be 1')
        # String should be lower case
        hdr = hdr_t.copy()
        hdr['a_str'] = 'My Name'
        assert_equal(self._dxer(hdr), 'a_str should be lower case')

    def test_log_checks(self):
        # Test logging, fixing, errors for header checking
        # This is specific to the particular header type. Here we use the
        # pretent header defined at the top of this file
        HC = self.header_class
        hdr = HC()
        hdr['an_integer'] = 2 # severity 40
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert_equal(fhdr['an_integer'], 1)
        assert_equal(message, 'an_integer should be 1; '
                           'set an_integer to 1')
        assert_raises(*raiser)
        # lower case string
        hdr = HC()
        hdr['a_str'] = 'Hello' # severity = 20
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_equal(message, 'a_str should be lower case; '
                           'set a_str to lower case')
        assert_raises(*raiser)

    def test_logger_error(self):
        # Check that we can reset the logger and error level
        # This is again specific to this pretend header
        HC = self.header_class
        hdr = HC()
        # Make a new logger
        str_io = StringIO()
        logger = logging.getLogger('test.logger')
        logger.setLevel(20)
        logger.addHandler(logging.StreamHandler(str_io))
        # Prepare something that needs fixing
        hdr['a_str'] = 'Fullness' # severity 20
        log_cache = imageglobals.logger, imageglobals.error_level
        try:
            # Check log message appears in new logger
            imageglobals.logger = logger
            hdr.copy().check_fix()
            assert_equal(str_io.getvalue(),
                         'a_str should be lower case; '
                         'set a_str to lower case\n')
            # Check that error_level in fact causes error to be raised
            imageglobals.error_level = 20
            assert_raises(HeaderDataError, hdr.copy().check_fix)
        finally:
            imageglobals.logger, imageglobals.error_level = log_cache

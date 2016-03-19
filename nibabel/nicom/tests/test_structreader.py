""" Testing Siemens CSA header reader
"""
import sys
import struct

from ..structreader import Unpacker

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)



def test_unpacker():
    s = b'1234\x00\x01'
    le_int, = struct.unpack('<h', b'\x00\x01')
    be_int, = struct.unpack('>h', b'\x00\x01')
    if sys.byteorder == 'little':
        native_int = le_int
        swapped_int = be_int
        swapped_code = '>'
    else:
        native_int = be_int
        swapped_int = le_int
        swapped_code = '<'
    up_str = Unpacker(s, endian='<')
    assert_equal(up_str.read(4), b'1234')
    up_str.ptr = 0
    assert_equal(up_str.unpack('4s'), (b'1234',))
    assert_equal(up_str.unpack('h'), (le_int,))
    up_str = Unpacker(s, endian='>')
    assert_equal(up_str.unpack('4s'), (b'1234',))
    assert_equal(up_str.unpack('h'), (be_int,))
    # now test conflict of endian
    up_str = Unpacker(s, ptr=4, endian='>')
    assert_equal(up_str.unpack('<h'), (le_int,))
    up_str = Unpacker(s, ptr=4, endian=swapped_code)
    assert_equal(up_str.unpack('h'), (swapped_int,))
    up_str.ptr = 4
    assert_equal(up_str.unpack('<h'), (le_int,))
    up_str.ptr = 4
    assert_equal(up_str.unpack('>h'), (be_int,))
    up_str.ptr = 4
    assert_equal(up_str.unpack('@h'), (native_int,))
    # test -1 for read
    up_str.ptr = 2
    assert_equal(up_str.read(), b'34\x00\x01')
    # past end
    assert_equal(up_str.read(), b'')
    # with n_bytes
    up_str.ptr = 2
    assert_equal(up_str.read(2), b'34')
    assert_equal(up_str.read(2), b'\x00\x01')

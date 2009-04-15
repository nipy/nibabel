''' Tests for endiancodes module '''

import sys

import numpy as np

from nose.tools import assert_raises, assert_true

from volumeimages.volumeutils import endian_codes, \
     native_code, swapped_code

def test_native_swapped():
    native_is_le = sys.byteorder == 'little'
    if native_is_le:
        assert (native_code, swapped_code) == ('<', '>')
    else:
        assert (native_code, swapped_code) == ('>', '<')
    
def test_to_numpy():
    if sys.byteorder == 'little':
        yield assert_true, endian_codes['native'] == '<'
        yield assert_true, endian_codes['swapped'] == '>'
    else:
        yield assert_true, endian_codes['native'] == '>' 
        yield assert_true, endian_codes['swapped'] == '<'
    yield assert_true, endian_codes['native'] == endian_codes['=']
    yield assert_true, endian_codes['big'] == '>'
    for code in ('little', '<', 'l', 'L', 'le'):
        yield assert_true, endian_codes[code] == '<'
    for code in ('big', '>', 'b', 'B', 'be'):
        yield assert_true, endian_codes[code] == '>'
   


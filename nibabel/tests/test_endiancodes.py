# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for endiancodes module '''

import sys


from nose.tools import assert_equal
from nose.tools import assert_true

from ..volumeutils import (endian_codes, native_code, swapped_code)


def test_native_swapped():
    native_is_le = sys.byteorder == 'little'
    if native_is_le:
        assert_equal((native_code, swapped_code), ('<', '>'))
    else:
        assert_equal((native_code, swapped_code), ('>', '<'))


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

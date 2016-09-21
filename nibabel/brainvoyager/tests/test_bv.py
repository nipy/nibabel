# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test main BV module."""

import os
import numpy as np
from ...loadsave import load
from ...tmpdirs import InTemporaryDirectory
from ..bv import (read_c_string, parse_BV_header, pack_BV_header, BvFileHeader,
                  calc_BV_header_size, _proto2default, update_BV_header,
                  parse_st, combine_st, BvError)
from ..bv_vtc import VTC_HDR_DICT_PROTO, BvVtcHeader
from ..bv_vmr import BvVmrImage
from ...testing import (assert_equal, assert_array_equal, data_path,
                        assert_true, assert_raises)
from . import BV_EXAMPLE_IMAGES, BV_EXAMPLE_HDRS
from ...externals import OrderedDict


vtc_file = os.path.join(data_path, 'test.vtc')
vmp_file = os.path.join(data_path, 'test.vmp')
vmr_file = os.path.join(data_path, 'test.vmr')

TEST_PROTO = (
    ('some_signed_char', 'b', 1),
    ('some_unsigned_char', 'B', 255),
    ('some_signed_short_integer', 'h', 6),
    ('some_signed_integer', 'i', 1),
    ('another_signed_integer', 'i', 3),
    ('some_counter_integer', 'i', 4),
    ('another_counter_integer', 'i', 0),
    ('some_unsigned_long_integer', 'I', 2712847316),
    ('some_float', 'f', 1.0),
    ('some_zero_terminated_string', 'z', b'HelloWorld!'),
    ('some_conditional_integer', 'i', (0, 'some_signed_integer', 1)),
    ('another_conditional_integer', 'i', (23, 'another_signed_integer', 1)),
    ('some_nested_field', (
        ('a_number', 'i', 1),
        ('a_float', 'f', 1.6500),
        ('a_string', 'z', b'test.txt'),
        ('nested_counter_integer', 'i', 2),
        ('fdr_table_info', (
            ('another_float', 'f', 0.0),
            ('another_string', 'z', b'sample'),
        ), 'nested_counter_integer'),
    ), 'some_counter_integer'),
    ('another_nested_field', (
        ('b_float', 'f', 1.234),
    ), 'another_counter_integer'),
)

TEST_HDR = OrderedDict([
     ('some_signed_char', 1),
     ('some_unsigned_char', 255),
     ('some_signed_short_integer', 6),
     ('some_signed_integer', 1),
     ('another_signed_integer', 3),
     ('some_counter_integer', 4),
     ('another_counter_integer', 0),
     ('some_unsigned_long_integer', 2712847316),
     ('some_float', 1.0),
     ('some_zero_terminated_string', b'HelloWorld!'),
     ('some_conditional_integer', 0),
     ('another_conditional_integer', 23),
     ('some_nested_field',
      [OrderedDict([('a_number', 1),
                    ('a_float', 1.65),
                    ('a_string', b'test.txt'),
                    ('nested_counter_integer', 2),
                    ('fdr_table_info',
                     [OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')]),
                      OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')])])]),
       OrderedDict([('a_number', 1),
                    ('a_float', 1.65),
                    ('a_string', b'test.txt'),
                    ('nested_counter_integer', 2),
                    ('fdr_table_info',
                     [OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')]),
                      OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')])])]),
       OrderedDict([('a_number', 1),
                    ('a_float', 1.65),
                    ('a_string', b'test.txt'),
                    ('nested_counter_integer', 2),
                    ('fdr_table_info',
                     [OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')]),
                      OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')])])]),
       OrderedDict([('a_number', 1),
                    ('a_float', 1.65),
                    ('a_string', b'test.txt'),
                    ('nested_counter_integer', 2),
                    ('fdr_table_info',
                     [OrderedDict([('another_float', 0.0),
                                   ('another_string', b'sample')]),
                      OrderedDict([('another_float', 0.0),
                                   ('another_string',
                                    b'sample')])])])]),
     ('another_nested_field', [])])

TEST_HDR_PACKED = \
    b''.join([b'\x01\xff\x06\x00\x01\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00',
              b'\x00\x00\x00\x00\x00\xd4\xc3\xb2\xa1\x00\x00\x80?HelloWorld!',
              b'\x00\x00\x00\x00\x00\x01\x00\x00\x0033\xd3?test.txt\x00\x02',
              b'\x00\x00\x00\x00\x00\x00\x00sample\x00\x00\x00\x00\x00sample',
              b'\x00\x01\x00\x00\x0033\xd3?test.txt\x00\x02\x00\x00\x00\x00',
              b'\x00\x00\x00sample\x00\x00\x00\x00\x00sample\x00\x01\x00\x00',
              b'\x0033\xd3?test.txt\x00\x02\x00\x00\x00\x00\x00\x00\x00sample',
              b'\x00\x00\x00\x00\x00sample\x00\x01\x00\x00\x0033\xd3?test.txt',
              b'\x00\x02\x00\x00\x00\x00\x00\x00\x00sample\x00\x00\x00\x00',
              b'\x00sample\x00'])


def test_read_c_string():
    # sample binary block
    binary = b'test.fmr\x00test.prt\x00'
    with InTemporaryDirectory():
        # create a tempfile
        path = 'test.header'
        fwrite = open(path, 'wb')

        # write the binary block to it
        fwrite.write(binary)
        fwrite.close()

        # open it again
        fread = open(path, 'rb')

        # test readout of one string
        assert_equal([s for s in read_c_string(fread)], [b'test.fmr'])

        # test new file position
        assert_equal(fread.tell(), 9)

        # manually rewind
        fread.seek(0)

        # test readout of two strings
        assert_equal([s for s in read_c_string(fread, 2, rewind=True)],
                     [b'test.fmr', b'test.prt'])

        # test automatic rewind
        assert_equal(fread.tell(), 0)

        # test readout of two strings with trailing zeros
        assert_equal([s for s in read_c_string(fread, 2, strip=False)],
                     [b'test.fmr\x00', b'test.prt\x00'])

        # test new file position
        assert_equal(fread.tell(), 18)

        # test readout of one string from given position
        fread.seek(0)
        assert_equal([s for s in read_c_string(fread, start_pos=9)],
                     [b'test.prt'])
        fread.close()


def test_combine_st():
    vmr = BvVmrImage.from_filename(vmr_file)
    STarray = []
    for st in range(vmr.header._hdr_dict['nr_of_past_spatial_trans']):
        STarray.append(parse_st(
            vmr.header._hdr_dict['past_spatial_trans'][st]))
    STarray = np.array(STarray)
    combinedST = combine_st(STarray, inv=True)
    correctCombinedST = [[1., 0., 0., 0.],
                         [0., 1., 0., -1.],
                         [0., 0., 1., 1.],
                         [0., 0., 0., 1.]]
    assert_array_equal(combinedST, correctCombinedST)
    combinedST = combine_st(STarray, inv=False)
    correctCombinedST = [[1., 0., 0., 0.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., -1.],
                         [0., 0., 0., 1.]]
    assert_array_equal(combinedST, correctCombinedST)


def test_parse_st():
    vmr = BvVmrImage.from_filename(vmr_file)
    ST = parse_st(vmr.header._hdr_dict['past_spatial_trans'][0])
    correctST = [[1., 0., 0., -1.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., -1.],
                 [0., 0., 0., 1.]]
    assert_array_equal(ST, correctST)

    # parse_st will only handle 4x4 matrices
    vmr.header._hdr_dict['past_spatial_trans'][0]['nr_of_trans_val'] = 10
    assert_raises(BvError, parse_st,
                  vmr.header._hdr_dict['past_spatial_trans'][0])


def compare_header_values(header, expected_header):
    '''recursively compare every value in header with expected_header'''

    for key in header:
        if (type(header[key]) is list):
            for i in range(len(expected_header[key])):
                compare_header_values(header[key][i], expected_header[key][i])
        assert_equal(header[key], expected_header[key])


def test_BvFileHeader_parse_BV_header():
    bv = _proto2default(TEST_PROTO)
    compare_header_values(bv, TEST_HDR)


def test_BvFileHeader_pack_BV_header():
    bv = _proto2default(TEST_PROTO)
    packed_bv = pack_BV_header(TEST_PROTO, bv)
    assert_equal(packed_bv, TEST_HDR_PACKED)

    # open vtc test file
    fileobj = open(vtc_file, 'rb')
    hdr_dict = parse_BV_header(VTC_HDR_DICT_PROTO, fileobj)
    binaryblock = pack_BV_header(VTC_HDR_DICT_PROTO, hdr_dict)
    assert_equal(binaryblock, b''.join([
        b'\x03\x00test.fmr\x00\x01\x00test.prt\x00\x00\x00\x02\x00\x05\x00',
        b'\x03\x00x\x00\x96\x00x\x00\x96\x00x\x00\x96\x00\x01\x01\x00\x00\xfaD'
    ]))


def test_BvFileHeader_calc_BV_header_size():
    bv = _proto2default(TEST_PROTO)
    assert_equal(calc_BV_header_size(TEST_PROTO, bv), 216)

    # change a header field
    bv['some_zero_terminated_string'] = 'AnotherString'
    assert_equal(calc_BV_header_size(TEST_PROTO, bv), 218)

    # open vtc test file
    fileobj = open(vtc_file, 'rb')
    hdr_dict = parse_BV_header(VTC_HDR_DICT_PROTO, fileobj)
    hdrSize = calc_BV_header_size(VTC_HDR_DICT_PROTO, hdr_dict)
    assert_equal(hdrSize, 48)


def test_BvFileHeader_update_BV_header():
    # increase a nested field counter
    bv = _proto2default(TEST_PROTO)
    bv_new = bv.copy()
    bv_new['some_counter_integer'] = 5
    bv_updated = update_BV_header(TEST_PROTO, bv, bv_new)
    assert_equal(len(bv_updated['some_nested_field']), 5)

    # decrease a nested field counter
    bv = _proto2default(TEST_PROTO)
    bv_new = bv.copy()
    bv_new['some_counter_integer'] = 3
    bv_updated = update_BV_header(TEST_PROTO, bv, bv_new)
    assert_equal(len(bv_updated['some_nested_field']), 3)


def test_BvFileHeader_xflip():
    bv = BvFileHeader()
    assert_true(bv.get_xflip())

    # should only return
    bv.set_xflip(True)

    # cannot flip most BV images
    assert_raises(BvError, bv.set_xflip, False)


def test_BvFileHeader_endianness():
    assert_raises(BvError, BvFileHeader, endianness='>')


def test_BvFileHeader_not_implemented():
    bv = BvFileHeader()
    assert_raises(NotImplementedError, bv.get_data_shape)
    assert_raises(NotImplementedError, bv.set_data_shape, (1, 2, 3))


def test_BvVtcHeader_from_header():
    vtc = load(vtc_file)
    vtc_data = vtc.get_data()

    # try the same load through the header
    fread = open(vtc_file, 'rb')
    header = BvVtcHeader.from_fileobj(fread)
    image = header.data_from_fileobj(fread)
    assert_array_equal(vtc_data, image)
    fread.close()


def test_BvVtcHeader_data_from_fileobj():
    vtc = load(vtc_file)
    vtc_data = vtc.get_data()

    # try the same load through the header
    fread = open(vtc_file, 'rb')
    header = BvVtcHeader.from_fileobj(fread)
    image = header.data_from_fileobj(fread)
    assert_array_equal(vtc_data, image)
    fread.close()


def test_parse_all_BV_headers():
    for images, headers in zip(BV_EXAMPLE_IMAGES, BV_EXAMPLE_HDRS):
        for i in range(len(images)):
            image = load(images[i]['fname'])
            compare_header_values(image.header._hdr_dict, headers[i])

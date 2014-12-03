""" Testing Siemens CSA header reader
"""

from .. import csareader as csa
from .. import xprotocol as xprotocol

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from .test_dicomwrappers import dicom_test, DATA


@dicom_test
def test_xprotocol_parse():
    # One place we find the XProtocol format is in the CSA series header
    csa_series_hdr = csa.get_csa_header(DATA, 'series')
    csa_series_dict = csa.header_to_key_val_mapping(csa_series_hdr)
    # It is stored in the element 'MrPhoenixProtocol'
    outer_xproto_str = csa_series_dict['MrPhoenixProtocol']
    outer_xprotos, remainder = xprotocol.read_protos(outer_xproto_str)
    assert_equal(''.join(remainder), '')
    assert_equal(len(outer_xprotos), 1)
    outer_xproto = outer_xprotos[0]
    # The format has a root containter 'XProtocol', which in turn (always?)
    # contains a anonymous container
    assert_equal(outer_xproto.keys(), ['XProtocol'])
    assert_equal(outer_xproto['XProtocol'].keys(), [''])
    # Any element can have meta data associated with it. In this case the
    # root 'XProtocol' element three pieces of meta data
    assert_equal(outer_xproto.get_elem('XProtocol').meta['Name'],
                 '"PhoenixMetaProtocol"')
    assert_equal(outer_xproto.get_elem('XProtocol').meta['ID'], '1000002')
    assert_equal(outer_xproto.get_elem('XProtocol').meta['Userversion'], '2.0')
    # In this case we also have three items in the anonymous container
    anon_dict = outer_xproto['XProtocol']['']
    assert_true('IsInlineComposed' in anon_dict)
    assert_true('Count' in anon_dict)
    assert_true('Protocol0' in anon_dict)
    assert_equal(anon_dict['IsInlineComposed'], None)
    assert_equal(anon_dict['Count'], 1)
    # The 'Protocol0' item is a string containing three separate definitions
    # all concatenated together: two XProtocol definitions followed by an
    # "ASCCONV" section
    inner_xproto_str = anon_dict['Protocol0']
    inner_xprotos, remainder = xprotocol.read_protos(inner_xproto_str)
    remainder = ''.join(remainder)
    assert_true(remainder.strip().startswith("### ASCCONV BEGIN ###"))
    assert_equal(len(inner_xprotos), 2)
    assert_equal(inner_xprotos[0].keys(), ['XProtocol'])
    assert_equal(inner_xprotos[1].keys(), ['XProtocol'])
    assert_equal(len(inner_xprotos[0]['XProtocol'].keys()), 21)
    assert_equal(len(inner_xprotos[0]['XProtocol'][''].keys()), 6)
    assert_equal(len(inner_xprotos[1]['XProtocol'].keys()), 1)
    assert_equal(len(inner_xprotos[1]['XProtocol'][''].keys()), 1)
    assert_equal(len(inner_xprotos[1]['XProtocol']['']['EVA'].keys()), 9)

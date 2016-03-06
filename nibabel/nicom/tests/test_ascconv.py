""" Testing Siemens "ASCCONV" parser
"""

from os.path import join as pjoin, dirname

import numpy as np

from .. import ascconv
from ...externals import OrderedDict

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal

DATA_PATH = pjoin(dirname(__file__), 'data')
ASCCONV_INPUT = pjoin(DATA_PATH, 'ascconv_sample.txt')


def test_ascconv_parse():
    with open(ASCCONV_INPUT, 'rt') as fobj:
        contents = fobj.read()
    ascconv_dict, attrs = ascconv.parse_ascconv(contents, str_delim='""')
    assert_equal(attrs, OrderedDict())
    assert_equal(len(ascconv_dict), 72)
    assert_equal(ascconv_dict['tProtocolName'], 'CBU+AF8-DTI+AF8-64D+AF8-1A')
    assert_equal(ascconv_dict['ucScanRegionPosValid'], 1)
    assert_array_almost_equal(ascconv_dict['sProtConsistencyInfo']['flNominalB0'],
                              2.89362)
    assert_equal(ascconv_dict['sProtConsistencyInfo']['flGMax'], 26)
    assert_equal(ascconv_dict['sSliceArray'].keys(),
                 ['asSlice', 'anAsc', 'anPos', 'lSize', 'lConc', 'ucMode',
                  'sTSat'])
    slice_arr = ascconv_dict['sSliceArray']
    as_slice = slice_arr['asSlice']
    assert_array_equal([e['dPhaseFOV'] for e in as_slice], 230)
    assert_array_equal([e['dReadoutFOV'] for e in as_slice], 230)
    assert_array_equal([e['dThickness'] for e in as_slice], 2.5)
    # Some lists defined starting at 1, so have None as first element
    assert_equal(slice_arr['anAsc'], [None] + list(range(1, 48)))
    assert_equal(slice_arr['anPos'], [None] + list(range(1, 48)))
    # A top level list
    assert_equal(len(ascconv_dict['asCoilSelectMeas']), 1)
    as_list = ascconv_dict['asCoilSelectMeas'][0]['asList']
    # This lower-level list does start indexing at 0
    assert_equal(len(as_list), 12)
    for i, el in enumerate(as_list):
        assert_equal(
            el.keys(),
            ['sCoilElementID', 'lElementSelected', 'lRxChannelConnected'])
        assert_equal(el['lElementSelected'], 1)
        assert_equal(el['lRxChannelConnected'], i + 1)


def test_ascconv_w_attrs():
    in_str = ("### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData "
              "version=41340006 "
              "converter=%MEASCONST%/ConverterList/Prot_Converter.txt ###\n"
              "test = \"hello\"\n"
              "### ASCCONV END ###")
    ascconv_dict, attrs = ascconv.parse_ascconv(in_str, '""')
    assert_equal(attrs['object'], 'MrProtDataImpl@MrProtocolData')
    assert_equal(attrs['version'], '41340006')
    assert_equal(attrs['converter'],
                 '%MEASCONST%/ConverterList/Prot_Converter.txt')
    assert_equal(ascconv_dict['test'], 'hello')

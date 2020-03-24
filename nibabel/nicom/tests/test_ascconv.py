""" Testing Siemens "ASCCONV" parser
"""

from os.path import join as pjoin, dirname

from .. import ascconv
from ...externals import OrderedDict

from numpy.testing import assert_array_equal, assert_array_almost_equal

DATA_PATH = pjoin(dirname(__file__), 'data')
ASCCONV_INPUT = pjoin(DATA_PATH, 'ascconv_sample.txt')


def test_ascconv_parse():
    with open(ASCCONV_INPUT, 'rt') as fobj:
        contents = fobj.read()
    ascconv_dict, attrs = ascconv.parse_ascconv('MrPhoenixProtocol', contents)
    assert attrs == OrderedDict()
    assert len(ascconv_dict) == 917
    assert ascconv_dict['tProtocolName'] == 'CBU+AF8-DTI+AF8-64D+AF8-1A'
    assert ascconv_dict['ucScanRegionPosValid'] == 1
    assert_array_almost_equal(ascconv_dict['sProtConsistencyInfo.flNominalB0'],
                              2.89362)
    assert ascconv_dict['sProtConsistencyInfo.flGMax'] == 26


def test_ascconv_w_attrs():
    in_str = ("### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData "
              "version=41340006 "
              "converter=%MEASCONST%/ConverterList/Prot_Converter.txt ###\n"
              "test = \"hello\"\n"
              "### ASCCONV END ###")
    ascconv_dict, attrs = ascconv.parse_ascconv('MrPhoenixProtocol', in_str)
    assert attrs['object'] == 'MrProtDataImpl@MrProtocolData'
    assert attrs['version'] == '41340006'
    assert attrs['converter'] == '%MEASCONST%/ConverterList/Prot_Converter.txt'
    assert ascconv_dict['test'] == 'hello'

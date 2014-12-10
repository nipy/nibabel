""" Testing Siemens "ASCCONV" parser
"""

from .. import csareader as csa
from .. import ascconv

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .test_dicomwrappers import dicom_test, DATA


@dicom_test
def test_ascconv_parse():
    # We grab an input string from the CSA header in a DICOM
    csa_series_hdr = csa.get_csa_header(DATA, 'series')
    csa_series_dict = csa.header_to_key_val_mapping(csa_series_hdr)
    input_str = csa_series_dict['MrPhoenixProtocol']
    ascconv_dict = ascconv.parse_ascconv('MrPhoenixProtocol', input_str)
    assert_equal(len(ascconv_dict), 917)
    assert_equal(ascconv_dict['tProtocolName'], 'CBU+AF8-DTI+AF8-64D+AF8-1A')
    assert_equal(ascconv_dict['ucScanRegionPosValid'], 1)
    assert_array_almost_equal(ascconv_dict['sProtConsistencyInfo.flNominalB0'],
                              2.89362)
    assert_equal(ascconv_dict['sProtConsistencyInfo.flGMax'], 26)

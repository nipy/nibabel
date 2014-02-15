""" Testing parrec module
"""

from os.path import join as pjoin, dirname

import numpy as np

from ..parrec import parse_PAR_header, PARRECHeader
from ..openers import Opener

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


DATA_PATH = pjoin(dirname(__file__), 'data')
EG_PAR = pjoin(DATA_PATH, 'phantom_EPI_asc_CLEAR_2_1.PAR')
EG_REC = pjoin(DATA_PATH, 'phantom_EPI_asc_CLEAR_2_1.REC')
with Opener(EG_PAR, 'rt') as _fobj:
    HDR_INFO, HDR_DEFS = parse_PAR_header(_fobj)


def test_header():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_equal(hdr.get_data_shape(), (64, 64, 9, 3))
    assert_equal(hdr.get_data_dtype(), np.dtype(np.int16))
    assert_equal(hdr.get_zooms(), (3.75, 3.75, 8.0, 2.0))
    assert_equal(hdr.get_data_offset(), 0)
    assert_almost_equal(hdr.get_slope_inter(),
                        (1.2903541326522827, 0.0), 5)


def test_header_scaling():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    fp_scaling = np.squeeze(hdr.get_data_scaling('fp'))
    dv_scaling = np.squeeze(hdr.get_data_scaling('dv'))
    # Check default is dv scaling
    assert_array_equal(np.squeeze(hdr.get_data_scaling()), dv_scaling)
    # And that it's almost the same as that from the converted nifti
    assert_almost_equal(dv_scaling, (1.2903541326522827, 0.0), 5)
    # Check that default for get_slope_inter is dv scaling
    for hdr in (hdr, PARRECHeader(HDR_INFO, HDR_DEFS, default_scaling='dv')):
        assert_array_equal(hdr.get_slope_inter(), dv_scaling)
    # Check we can change the default
    assert_false(np.all(fp_scaling == dv_scaling))
    fp_hdr = PARRECHeader(HDR_INFO, HDR_DEFS, default_scaling='fp')
    assert_array_equal(fp_hdr.get_slope_inter(), fp_scaling)


def test_orientation():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_array_equal(HDR_DEFS['slice orientation'], 1)
    assert_equal(hdr.get_slice_orientation(), 'transverse')
    hdr_defc = HDR_DEFS.copy()
    hdr = PARRECHeader(HDR_INFO, hdr_defc)
    hdr_defc['slice orientation'] = 2
    assert_equal(hdr.get_slice_orientation(), 'sagittal')
    hdr_defc['slice orientation'] = 3
    hdr = PARRECHeader(HDR_INFO, hdr_defc)
    assert_equal(hdr.get_slice_orientation(), 'coronal')

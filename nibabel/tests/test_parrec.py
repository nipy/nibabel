""" Testing parrec module
"""

from os.path import join as pjoin, dirname

import numpy as np

from ..parrec import parse_PAR_header, PARRECHeader, PARRECError
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
# Affine as we determined it mid-2014
AN_OLD_AFFINE = np.array(
    [[-3.64994708, 0.,   1.83564171, 123.66276611],
     [0.,         -3.75, 0.,          115.617    ],
     [0.86045705,  0.,   7.78655376, -27.91161211],
     [0.,          0.,   0.,           1.        ]])
# Affine from Philips-created NIfTI
PHILIPS_AFFINE = np.array(
    [[  -3.65  ,   -0.0016,    1.8356,  125.4881],
     [   0.0016,   -3.75  ,   -0.0004,  117.4916],
     [   0.8604,    0.0002,    7.7866,  -28.3411],
     [   0.    ,    0.    ,    0.    ,    1.    ]])


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


def test_data_offset():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_equal(hdr.get_data_offset(), 0)
    # Can set 0
    hdr.set_data_offset(0)
    # Can't set anything else
    assert_raises(PARRECError, hdr.set_data_offset, 1)


def test_affine():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    default = hdr.get_affine()
    scanner = hdr.get_affine(origin='scanner')
    fov = hdr.get_affine(origin='fov')
    assert_array_equal(default, scanner)
    # rotation part is same
    assert_array_equal(scanner[:3, :3], fov[:3,:3])
    # offset not
    assert_false(np.all(scanner[:3, 3] == fov[:3, 3]))
    # Regression test against what we were getting before
    assert_almost_equal(default, AN_OLD_AFFINE)
    # Test against RZS of Philips affine
    assert_almost_equal(default[:3, :3], PHILIPS_AFFINE[:3, :3], 2)

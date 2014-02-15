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
    # hdr appears to have ms as zoom; should probably fix
    assert_equal(hdr.get_zooms(), (3.75, 3.75, 8.0, 2.0 * 1000))

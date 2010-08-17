# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Testing header scaling

"""

import numpy as np

from .. import AnalyzeHeader, Nifti1Header
from ..spatialimages import HeaderDataError, HeaderTypeError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import parametric


@parametric
def test_hdr_scaling():
    hdr = AnalyzeHeader()
    hdr.set_data_dtype(np.float32)
    data = np.arange(6, dtype=np.float32).reshape(1,2,3)
    res = hdr.scaling_from_data(data)
    yield assert_equal(res, (1.0, 0.0, None, None))
     # The Analyze header cannot scale
    hdr.set_data_dtype(np.uint8)
    yield assert_raises(HeaderTypeError, hdr.scaling_from_data, data)
     # The nifti header can scale
    hdr = Nifti1Header()
    hdr.set_data_dtype(np.uint8)
    slope, inter, mx, mn = hdr.scaling_from_data(data)
    yield assert_equal((inter, mx, mn), (0.0, 0, 5))
    yield assert_array_almost_equal(slope, 5.0/255)


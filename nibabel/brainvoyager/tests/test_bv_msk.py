# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test BV module for MSK files."""

from os.path import join as pjoin
import numpy as np
from ..bv_msk import BvMskImage, BvMskHeader
from ...testing import (assert_equal, assert_raises, data_path)
from ...spatialimages import HeaderDataError
from ...externals import OrderedDict

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
BVMSK_EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.msk'),
        shape=(10, 10, 10),
        dtype=np.uint8,
        affine=np.array([[-3., 0, 0, -21.],
                         [0, 0, -3., -21.],
                         [0, -3., 0, -21.],
                         [0, 0, 0, 1.]]),
        zooms=(3., 3., 3.),
        fileformat=BvMskImage,
        # These values are from NeuroElf
        data_summary=dict(
            min=0,
            max=1,
            mean=0.499),
        is_proxy=True)
]

BVMSK_EXAMPLE_HDRS = [
    OrderedDict([('resolution', 3),
                 ('x_start', 120),
                 ('x_end', 150),
                 ('y_start', 120),
                 ('y_end', 150),
                 ('z_start', 120),
                 ('z_end', 150)])
]


def test_BvMskHeader_set_data_shape():
    msk = BvMskHeader()
    assert_equal(msk.get_data_shape(), (46, 40, 58))
    msk.set_data_shape((45, 39, 57))
    assert_equal(msk.get_data_shape(), (45, 39, 57))

    # Use zyx parameter instead of shape
    msk.set_data_shape(None, [[57, 240], [52, 178], [59, 191]])
    assert_equal(msk.get_data_shape(), (61, 42, 44))

    # raise error when neither shape nor xyz is specified
    assert_raises(HeaderDataError, msk.set_data_shape, None, None)

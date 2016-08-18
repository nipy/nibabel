# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test BV module for VTC files."""

from os.path import join as pjoin
import numpy as np
from ..bv import BvError
from ..bv_vtc import BvVtcImage, BvVtcHeader
from ...testing import (data_path, assert_equal, assert_raises, assert_true,
                        assert_false)
from numpy.testing import (assert_array_equal)
from ...spatialimages import HeaderDataError
from ...externals import OrderedDict

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
BVVTC_EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.vtc'),
        shape=(10, 10, 10, 5),
        dtype=np.float32,
        affine=np.array([[-3., 0, 0, -21.],
                         [0, 0, -3., -21.],
                         [0, -3., 0, -21.],
                         [0, 0, 0, 1.]]),
        zooms=(3., 3., 3.),
        fileformat=BvVtcImage,
        # These values are from NeuroElf
        data_summary=dict(
            min=0.0096689118,
            max=199.93549,
            mean=100.19728),
        is_proxy=True)
]

BVVTC_EXAMPLE_HDRS = [
    OrderedDict([('version', 3),
                 ('fmr', b'test.fmr'),
                 ('nr_prts', 1),
                 ('prts', [OrderedDict([('filename', b'test.prt')])]),
                 ('current_prt', 0),
                 ('datatype', 2),
                 ('volumes', 5),
                 ('resolution', 3),
                 ('x_start', 120),
                 ('x_end', 150),
                 ('y_start', 120),
                 ('y_end', 150),
                 ('z_start', 120),
                 ('z_end', 150),
                 ('lr_convention', 1),
                 ('ref_space', 1),
                 ('tr', 2000.0)])
]


def test_get_base_affine():
    hdr = BvVtcHeader()
    hdr.set_data_shape((3, 5, 7, 9))
    hdr.set_zooms((3, 3, 3, 3))
    assert_array_equal(hdr.get_base_affine(),
                       np.asarray([[-3.,  0.,  0.,  193.5],
                                   [0.,  0.,  -3., 181.5],
                                   [0.,  -3.,  0., 205.5],
                                   [0.,  0.,  0.,  1.]]))


def test_BvVtcHeader_set_data_shape():
    vtc = BvVtcHeader()
    assert_equal(vtc.get_data_shape(), (46, 40, 58, 0))
    vtc.set_data_shape((45, 39, 57, 0))
    assert_equal(vtc.get_data_shape(), (45, 39, 57, 0))

    # Use zyx parameter instead of shape
    vtc.set_data_shape(None, [[57, 240], [52, 178], [59, 191]])
    assert_equal(vtc.get_data_shape(), (61, 42, 44, 0))

    # Change number of submaps
    vtc.set_data_shape(None, None, 5)  # via t parameter
    assert_equal(vtc.get_data_shape(), (61, 42, 44, 5))
    vtc.set_data_shape((61, 42, 44, 3))  # via shape parameter
    assert_equal(vtc.get_data_shape(), (61, 42, 44, 3))

    # raise error when neither shape nor zyx nor t is specified
    assert_raises(HeaderDataError, vtc.set_data_shape, None, None, None)

    # raise error when n is negative
    assert_raises(HeaderDataError, vtc.set_data_shape, (45, 39, 57, -1))
    assert_raises(HeaderDataError, vtc.set_data_shape, None, None, -1)


def test_BvVtcHeader_set_framing_cube():
    vtc = BvVtcHeader()
    assert_equal(vtc.get_framing_cube(), (256, 256, 256))
    vtc.set_framing_cube((512, 512, 512))
    assert_equal(vtc.get_framing_cube(), (512, 512, 512))
    vtc.set_framing_cube((512, 513, 514))
    assert_equal(vtc.get_framing_cube(), (512, 513, 514))


def test_BvVtcHeader_xflip():
    vtc = BvVtcHeader()
    assert_true(vtc.get_xflip())
    vtc.set_xflip(False)
    assert_false(vtc.get_xflip())
    vtc.set_xflip(True)
    assert_true(vtc.get_xflip())
    vtc.set_xflip(0)
    assert_raises(BvError, vtc.get_xflip)


def test_BvVtcHeader_guess_framing_cube():
    vtc = BvVtcHeader()
    assert_equal(vtc._guess_framing_cube(), (256, 256, 256))
    vtc._hdr_dict['x_end'] = 400
    vtc._hdr_dict['y_end'] = 400
    vtc._hdr_dict['z_end'] = 400
    assert_equal(vtc._guess_framing_cube(), (512, 512, 512))


def test_BvVtcHeader_zooms():
    vtc = BvVtcHeader()
    assert_equal(vtc.get_zooms(), (3.0, 3.0, 3.0))

    # set all zooms to one value (default for VTC files)
    vtc.set_zooms(2)
    assert_equal(vtc.get_zooms(), (2.0, 2.0, 2.0))
    vtc.set_zooms((1.0, 1.0, 1.0))
    assert_equal(vtc.get_zooms(), (1.0, 1.0, 1.0))
    vtc.set_zooms((4, 4, 4))
    assert_equal(vtc.get_zooms(), (4.0, 4.0, 4.0))

    # set zooms to different values for the three dimensions (not possible)
    assert_raises(BvError, vtc.set_zooms, (1.0, 2.0, 3.0))


def test_BvVtcHeader_fileversion_error():
    vtc = BvVtcHeader()
    vtc._hdr_dict['version'] = 4
    assert_raises(HeaderDataError, BvVtcHeader.from_header, vtc)

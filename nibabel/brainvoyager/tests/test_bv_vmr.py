# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test BV module for VMR files."""

from os.path import join as pjoin
import numpy as np
from ..bv import BvError
from ..bv_vmr import BvVmrImage, BvVmrHeader
from ...testing import (assert_equal, assert_true, assert_false, assert_raises,
                        assert_array_equal, data_path)
from ...externals import OrderedDict

vmr_file = pjoin(data_path, 'test.vmr')

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
BVVMR_EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.vmr'),
        shape=(5, 4, 3),
        dtype=np.uint8,
        affine=np.array([[-1., 0., 0., 0.],
                        [0., 0., -1., -1.],
                        [0., -1., 0., 1.],
                        [0., 0., 0., 1.]]),
        zooms=(3., 3., 3.),
        fileformat=BvVmrImage,
        # These values are from NeuroElf
        data_summary=dict(
            min=7,
            max=218,
            mean=120.3),
        is_proxy=True)
]

BVVMR_EXAMPLE_HDRS = [
    OrderedDict([('version', 4),
                 ('dim_x', 3),
                 ('dim_y', 4),
                 ('dim_z', 5),
                 ('offset_x', 0),
                 ('offset_y', 0),
                 ('offset_z', 0),
                 ('framing_cube', 256),
                 ('pos_infos_verified', 0),
                 ('coordinate_system_entry', 1),
                 ('slice_first_center_x', 127.5),
                 ('slice_first_center_y', 0.0),
                 ('slice_first_center_z', 0.0),
                 ('slice_last_center_x', -127.5),
                 ('slice_last_center_y', 0.0),
                 ('slice_last_center_z', 0.0),
                 ('row_dir_x', 0.0),
                 ('row_dir_y', 1.0),
                 ('row_dir_z', 0.0),
                 ('col_dir_x', 0.0),
                 ('col_dir_y', 0.0),
                 ('col_dir_z', -1.0),
                 ('nr_rows', 256),
                 ('nr_cols', 256),
                 ('fov_row_dir', 256.0),
                 ('fov_col_dir', 256.0),
                 ('slice_thickness', 1.0),
                 ('gap_thickness', 0.0),
                 ('nr_of_past_spatial_trans', 2),
                 ('past_spatial_trans',
                  [OrderedDict([('name', b'NoName'),
                                ('type', 2),
                                ('source_file', b'/home/test.vmr'),
                                ('nr_of_trans_val', 16),
                                ('trans_val',
                                 [OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', -1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', -1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)])])]),
                   OrderedDict([('name', b'NoName'),
                                ('type', 2),
                                ('source_file', b'/home/test_TRF.vmr'),
                                ('nr_of_trans_val', 16),
                                ('trans_val',
                                 [OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 0.0)]),
                                  OrderedDict([('value', 1.0)])])])]),
                 ('lr_convention', 1),
                 ('reference_space', 0),
                 ('vox_res_x', 1.0),
                 ('vox_res_y', 1.0),
                 ('vox_res_z', 1.0),
                 ('flag_vox_resolution', 0),
                 ('flag_tal_space', 0),
                 ('min_intensity', 0),
                 ('mean_intensity', 127),
                 ('max_intensity', 255)])
]


def test_BvVmrHeader_xflip():
    vmr = BvVmrHeader()
    assert_true(vmr.get_xflip())
    vmr.set_xflip(False)
    assert_false(vmr.get_xflip())
    assert_equal(vmr._hdr_dict['lr_convention'], 2)
    vmr.set_xflip(True)
    assert_true(vmr.get_xflip())
    assert_equal(vmr._hdr_dict['lr_convention'], 1)
    vmr.set_xflip(0)
    assert_equal(vmr._hdr_dict['lr_convention'], 0)
    assert_raises(BvError, vmr.get_xflip)

    vmr = BvVmrImage.from_filename(vmr_file)
    vmr.header.set_xflip(False)
    expected_affine = [[1., 0., 0., 0.],
                       [0., 0., -1., -1.],
                       [0., -1., 0., 1.],
                       [0., 0., 0., 1.]]
    assert_array_equal(vmr.header.get_affine(), expected_affine)


def test_BvVmrHeader_set_zooms():
    vmr = BvVmrHeader()
    assert_equal(vmr.get_zooms(), (1.0, 1.0, 1.0))
    vmr.set_zooms((1.1, 2.2, 3.3))
    assert_equal(vmr.get_zooms(), (1.1, 2.2, 3.3))
    assert_equal(vmr._hdr_dict['vox_res_z'], 1.1)

# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test NRRD files written by 3D Slicer, ITK and pynrrd

The files are the slicerio ``TestingData`` release (revision rev001). Get them
with::

    python tools/fetch_nrrd_testdata.py

Shapes and affines below were checked against SimpleITK 2.5. Voxel values are
checked against ``nrrd.read`` from pynrrd.
"""

from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from .. import Nifti1Image, aff2axcodes
from .. import load as top_load
from ..nrrd import NrrdImage
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data

pynrrd, have_pynrrd, setup_module = optional_package('nrrd')

NRRD_PATH = pjoin(get_nibabel_data(), 'nitest-nrrd')
needs_nrrd_data = needs_nibabel_data('nitest-nrrd')

# 2D files have no anatomical space, and loading them warns
pytestmark = pytest.mark.filterwarnings('ignore:NRRD file has space dimension 2:UserWarning')

# File name, shape, dtype, axis codes
FILES = [
    ('CTCardio_2Dt.seq.nrrd', (128, 104, 5), 'int16', ('R', 'A', 'S')),
    ('CTCardio_3Dt.seq.nrrd', (128, 104, 72, 5), 'int16', ('L', 'P', 'S')),
    ('CTCardio_3Dt_displacement.seq.nrrd', (64, 52, 36, 5, 3), 'float32', ('L', 'P', 'S')),
    ('CTCardio_3Dt_displacement_RAS.seq.nrrd', (64, 52, 36, 5, 3), 'float32', ('L', 'P', 'S')),
    ('CTCardio_3Dt_displacement_resampled.seq.nrrd', (128, 104, 72, 5), 'int16', ('L', 'P', 'S')),
    ('CTCardio_3Dt_velocity.seq.nrrd', (64, 52, 36, 5, 3), 'float32', ('L', 'P', 'S')),
    ('DTIBrain_3D_tensor.nrrd', (144, 144, 85, 9), 'float32', ('L', 'P', 'S')),
    ('DTIBrain_3D_tensor_symmetric.nrrd', (144, 144, 85, 6), 'float32', ('L', 'P', 'S')),
    ('DTIBrain_FA.nrrd', (144, 144, 85), 'float32', ('L', 'P', 'S')),
    ('DTIBrain_baseline.nrrd', (256, 256, 51), 'int16', ('L', 'P', 'S')),
    ('DWI_small.nrrd', (92, 119, 80, 7), 'uint16', ('L', 'P', 'S')),
    ('DWI_small_b0.nrrd', (92, 119, 80), 'uint16', ('L', 'P', 'S')),
    ('MRHead_2D.nrrd', (256, 205), 'int16', ('R', 'A', 'S')),
    ('MRHead_3D.nrrd', (256, 205, 106), 'int16', ('P', 'I', 'R')),
    ('MRHead_3D_RAS.nrrd', (256, 205, 106), 'int16', ('P', 'I', 'R')),
    ('MRHead_3D_detached.nhdr', (128, 102, 53), 'int16', ('P', 'I', 'R')),
    ('MRHead_3D_oblique.nrrd', (182, 146, 75), 'int16', ('P', 'I', 'R')),
    ('MRHead_3D_raw.nrrd', (128, 102, 53), 'int16', ('P', 'I', 'R')),
    ('MRHead_displacement.nrrd', (64, 51, 26, 3), 'float64', ('P', 'I', 'R')),
    ('MRHead_displacement_RAS.nrrd', (64, 51, 26, 3), 'float64', ('P', 'I', 'R')),
    ('MRHead_displacement_resampled.nrrd', (256, 205, 106), 'int16', ('P', 'I', 'R')),
    ('MRHead_oblique_displacement.nrrd', (46, 36, 19, 3), 'float64', ('P', 'I', 'R')),
    ('MRHead_oblique_displacement_resampled.nrrd', (182, 146, 75), 'int16', ('P', 'I', 'R')),
    ('MRHead_segmentation.seg.nrrd', (256, 205, 106), 'uint8', ('P', 'I', 'R')),
    ('MRHead_segmentation_RAS.seg.nrrd', (256, 205, 106), 'uint8', ('P', 'I', 'R')),
    ('MRHead_segmentation_oblique.seg.nrrd', (182, 146, 75), 'uint8', ('P', 'I', 'R')),
    ('MRHead_segmentation_overlapping.seg.nrrd', (256, 205, 106, 2), 'uint8', ('P', 'I', 'R')),
    ('MRHead_velocity.nrrd', (128, 102, 53, 3), 'float32', ('P', 'I', 'R')),
    ('MRHead_velocity_RAS.nrrd', (128, 102, 53, 3), 'float32', ('P', 'I', 'R')),
    ('VisibleHumanHead_2D_RGB.nrrd', (760, 800, 3), 'uint8', ('R', 'A', 'S')),
    ('VisibleHumanHead_2D_RGBA.nrrd', (760, 800, 4), 'uint8', ('R', 'A', 'S')),
    ('VisibleHumanHead_3D_RGB.nrrd', (190, 200, 80, 3), 'uint8', ('L', 'A', 'I')),
    ('VisibleHumanHead_3D_RGBA.nrrd', (190, 200, 80, 4), 'uint8', ('L', 'A', 'I')),
]

# The same image stored in LPS and in RAS, or raw and detached gzip
TWINS = [
    ('MRHead_3D.nrrd', 'MRHead_3D_RAS.nrrd'),
    ('MRHead_segmentation.seg.nrrd', 'MRHead_segmentation_RAS.seg.nrrd'),
    ('MRHead_displacement.nrrd', 'MRHead_displacement_RAS.nrrd'),
    ('MRHead_velocity.nrrd', 'MRHead_velocity_RAS.nrrd'),
    ('CTCardio_3Dt_displacement.seq.nrrd', 'CTCardio_3Dt_displacement_RAS.seq.nrrd'),
    ('MRHead_3D_raw.nrrd', 'MRHead_3D_detached.nhdr'),
]


def _load(fname, **kwargs):
    return top_load(pjoin(NRRD_PATH, fname), **kwargs)


@needs_nrrd_data
@pytest.mark.parametrize(('fname', 'shape', 'dtype', 'axcodes'), FILES)
def test_files(fname, shape, dtype, axcodes):
    img = _load(fname)
    assert isinstance(img, NrrdImage)
    assert img.shape == shape
    assert img.get_data_dtype() == np.dtype(dtype)
    assert aff2axcodes(img.affine) == axcodes
    assert img.dataobj.is_proxy
    # Voxel values match pynrrd, which reads the data itself
    expected, _ = pynrrd.read(pjoin(NRRD_PATH, fname))
    expected = expected.transpose(img.header.get_axis_order())
    assert_array_equal(np.asanyarray(img.dataobj), expected)
    assert_array_equal(img.dataobj[..., -1], expected[..., -1])
    # The geometry survives conversion to NIfTI
    nifti = Nifti1Image.from_image(img)
    assert nifti.shape == shape
    assert_almost_equal(nifti.affine, img.affine, 5)


@needs_nrrd_data
def test_mrhead_affine():
    # From SimpleITK: LPS direction, spacing and origin, converted to RAS
    img = _load('MRHead_3D.nrrd')
    expected = [
        [0, 0, 1.6, -86.6449],
        [-1, 0, 0, 133.9286],
        [0, -1.25, 0, 116.7857],
        [0, 0, 0, 1],
    ]
    assert_almost_equal(img.affine, expected, 4)
    assert_almost_equal(img.header.get_zooms(), (1, 1.25, 1.6), 6)
    assert img.header.get_space() == 'left-posterior-superior'


@needs_nrrd_data
def test_dwi():
    img = _load('DWI_small.nrrd')
    assert img.header.get_axis_order() == (1, 2, 3, 0)
    assert_almost_equal(
        img.affine, [[-1.5, 0, 0, 72], [0, -1.5, 0, 91.5], [0, 0, 1.5, -49.5], [0, 0, 0, 1]]
    )
    assert_almost_equal(img.header.get_zooms(), (1.5, 1.5, 1.5, 1))
    assert_array_equal(img.header.get_measurement_frame(), np.diag([-1, -1, 1]))
    assert 'DWMRI_b-value' in img.header.keyvalues
    assert 'DWMRI_gradient_0006' in img.header.keyvalues


@needs_nrrd_data
@pytest.mark.parametrize(('first', 'second'), TWINS)
def test_twins(first, second):
    img1, img2 = _load(first), _load(second)
    assert_almost_equal(img1.affine, img2.affine, 4)
    assert_array_equal(img1.dataobj, img2.dataobj)
    frame1 = img1.header.get_measurement_frame()
    frame2 = img2.header.get_measurement_frame()
    if frame1 is None:
        assert frame2 is None
    else:
        assert_array_equal(frame1, frame2)


@needs_nrrd_data
def test_2d_warns():
    with pytest.warns(UserWarning, match='no anatomical orientation'):
        _load('MRHead_2D.nrrd')


@needs_nrrd_data
@pytest.mark.parametrize('fname', ['MRHead_3D.nrrd', 'CTCardio_3Dt_displacement.seq.nrrd'])
def test_nifti_file_round_trip(tmp_path, fname):
    img = _load(fname)
    out = tmp_path / 'img.nii.gz'
    Nifti1Image.from_image(img).to_filename(out)
    back = top_load(out)
    assert back.shape == img.shape
    assert_almost_equal(back.affine, img.affine, 5)
    assert_array_equal(back.dataobj, img.dataobj)

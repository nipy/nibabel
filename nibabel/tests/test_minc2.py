# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import Nifti1Image, load, minc2, save
from ..minc2 import Minc2File, Minc2Image
from ..optpkg import optional_package
from ..testing import data_path
from . import test_minc1 as tm2

h5py, have_h5py, setup_module = optional_package('h5py')

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'small.mnc'),
        shape=(18, 28, 29),
        dtype=np.int16,
        affine=np.array(
            [
                [0, 0, 7.0, -98],
                [0, 8.0, 0, -134],
                [9.0, 0, 0, -72],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(9.0, 8.0, 7.0),
        # These values from mincstats
        data_summary=dict(min=0.1185331417, max=92.87690699, mean=31.2127952),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2_1_scale.mnc'),
        shape=(10, 20, 20),
        dtype=np.uint8,
        affine=np.array(
            [
                [0, 0, 2.0, -20],
                [0, 2.0, 0, -20],
                [2.0, 0, 0, -10],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(2.0, 2.0, 2.0),
        # These values from mincstats
        data_summary=dict(min=0.2082842439, max=0.2094327615, mean=0.2091292083),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2_4d.mnc'),
        shape=(2, 10, 20, 20),
        dtype=np.uint8,
        affine=np.array(
            [
                [0, 0, 2.0, -20],
                [0, 2.0, 0, -20],
                [2.0, 0, 0, -10],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(1.0, 2.0, 2.0, 2.0),
        # These values from mincstats
        data_summary=dict(min=0.2078431373, max=1.498039216, mean=0.9090422837),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2-no-att.mnc'),
        shape=(10, 20, 20),
        dtype=np.uint8,
        affine=np.array(
            [
                [0, 0, 1.0, 0],
                [0, 1.0, 0, 0],
                [1.0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
        zooms=(1.0, 1.0, 1.0),
        # These values from SPM2/mincstats
        data_summary=dict(min=0.20784314, max=0.74901961, mean=0.6061103),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc2-4d-d.mnc'),
        shape=(5, 16, 16, 16),
        dtype=np.float64,
        affine=np.array(
            [
                [1.0, 0.0, 0.0, -6.96],
                [0.0, 1.0, 0.0, -12.453],
                [0.0, 0.0, 1.0, -9.48],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        zooms=(1.0, 1.0, 1.0, 1.0),
        # These values from mincstats
        data_summary=dict(min=0.0, max=5.0, mean=2.00078125),
        is_proxy=True,
    ),
]

if have_h5py:

    class TestMinc2File(tm2._TestMincFile):
        module = minc2
        file_class = Minc2File
        opener = h5py.File
        test_files = EXAMPLE_IMAGES

    class TestMinc2Image(tm2.TestMinc1Image):
        image_class = Minc2Image
        eg_images = (pjoin(data_path, 'small.mnc'),)
        module = minc2


def test_bad_diminfo():
    fname = pjoin(data_path, 'minc2_baddim.mnc')
    # File has a bad spacing field 'xspace' when it should be
    # `irregular`, `regular__` or absent (default to regular__).
    # We interpret an invalid spacing as absent, but warn.
    with pytest.warns(UserWarning):
        Minc2Image.from_filename(fname)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('tp', EXAMPLE_IMAGES + tm2.EXAMPLE_IMAGES)
def test_save_example_images(tmp_path, tp):
    # Saving a loaded MINC1 or MINC2 image as MINC2 keeps the stored values
    # and scaling, so the data are identical
    img = load(tp['fname'])
    data = img.get_fdata()
    fname = tmp_path / 'img.mnc'
    save(Minc2Image.from_image(img), fname)
    rt_img = load(fname)
    assert isinstance(rt_img, Minc2Image)
    assert rt_img.get_data_dtype().newbyteorder('=') == np.dtype(tp['dtype'])
    assert_array_equal(rt_img.get_fdata(), data)
    assert_array_equal(rt_img.affine, img.affine)
    if len(tp['shape']) == 4:
        # Time step and start are kept
        rt_file = Minc2File(h5py.File(fname, 'r'))
        assert rt_file.get_zooms() == tp['zooms']


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64]
)
@pytest.mark.parametrize('shape', [(4, 5, 6), (3, 4, 5, 6)])
def test_image_round_trip(tmp_path, dtype, shape):
    affine = np.array([[0, 0, -2.0, 20], [0.2, 2.0, 0, -20], [3.0, 0, 0.1, -10], [0, 0, 0, 1]])
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) % 100
    img = Minc2Image(data, affine)
    assert img.get_data_dtype() == dtype
    fname = tmp_path / 'img.mnc'
    img.to_filename(fname)
    rt_img = Minc2Image.from_filename(fname)
    assert rt_img.get_data_dtype() == dtype
    assert_array_equal(rt_img.get_fdata(), data)
    assert_allclose(rt_img.affine, affine, rtol=0, atol=1e-12)
    # dtype argument and header dtype give the on-disk type
    img.to_filename(fname, dtype=np.float32)
    assert Minc2Image.from_filename(fname).get_data_dtype() == np.float32
    img.set_data_dtype(np.int16)
    img.to_filename(fname)
    assert Minc2Image.from_filename(fname).get_data_dtype() == np.int16


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_file_object_round_trip():
    data = np.arange(60, dtype=np.float32).reshape((3, 4, 5))
    img = Minc2Image(data, np.diag([2, 3, 4, 1]))
    bio = BytesIO()
    file_map = Minc2Image.make_file_map({'image': bio})
    img.to_file_map(file_map)
    rt_img = Minc2Image.from_file_map(file_map)
    assert_array_equal(rt_img.get_fdata(), data)
    assert_array_equal(rt_img.affine, img.affine)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_save_nifti_as_minc2(tmp_path):
    data = np.arange(60, dtype=np.int16).reshape((3, 4, 5))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-1, 2, 3]
    fname = tmp_path / 'img.mnc'
    save(Minc2Image.from_image(Nifti1Image(data, affine)), fname)
    rt_img = load(fname)
    assert isinstance(rt_img, Minc2Image)
    assert rt_img.get_data_dtype() == np.int16
    assert_array_equal(rt_img.get_fdata(), data)
    assert_array_equal(rt_img.affine, affine)

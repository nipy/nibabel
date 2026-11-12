# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for the MINC2 writer module, independent of ``Minc2Image``

Files are written with ``write_minc2`` and read back with ``nibabel.load``.
"""

from contextlib import nullcontext
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import load
from ..affines import from_matvec
from ..eulerangles import euler2mat
from ..minc2_write import (
    MINC_VERSION,
    Minc2WriteError,
    affine_to_dimensions,
    write_minc2,
)
from ..optpkg import optional_package
from ..testing import data_path

h5py, have_h5py, setup_module = optional_package('h5py')

DTYPES = (np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64)

MINC2_FILES = (
    'small.mnc',
    'minc2_1_scale.mnc',
    'minc2_4d.mnc',
    'minc2-no-att.mnc',
    'minc2-4d-d.mnc',
    'minc2_baddim.mnc',
)

OBLIQUE_AFFINE = from_matvec(euler2mat(0.3, -0.2, 0.5) @ np.diag([1.5, -2.0, 0.7]), [10, -20, 5])
SHEARED_AFFINE = np.array(
    [
        [0.9, 0.2, 0.0, -30.0],
        [0.1, -1.1, 0.3, 12.0],
        [0.0, 0.25, 2.5, 4.5],
        [0, 0, 0, 1],
    ]
)
AFFINES = (
    np.diag([1.0, 2.0, 3.0, 1.0]),
    np.array([[0, 0, -2.0, 20], [0, 2.0, 0, -20], [3.0, 0, 0, -10], [0, 0, 0, 1]]),
    OBLIQUE_AFFINE,
    SHEARED_AFFINE,
)


def _load_dims(fname):
    """Return {name: attrs} of the dimension variables in a MINC2 file"""
    with h5py.File(fname, 'r') as h5f:
        dims = h5f['minc-2.0/dimensions']
        return {name: dict(dims[name].attrs) for name in dims}


def _file_scaling(fname):
    """Read ``(image_min, image_max, valid_range)`` of a MINC2 file"""
    with h5py.File(fname, 'r') as h5f:
        group = h5f['minc-2.0/image/0']
        image = group['image']
        image_min = group['image-min'][()]
        image_max = group['image-max'][()]
        info = np.iinfo(image.dtype)
        valid_range = image.attrs.get('valid_range', (info.min, info.max))
    return image_min, image_max, valid_range


@pytest.mark.parametrize('affine', AFFINES)
def test_affine_to_dimensions(affine):
    names, starts, steps, cosines = affine_to_dimensions(affine)
    assert sorted(names) == ['xspace', 'yspace', 'zspace']
    # Unit cosines, pointing along the positive world axis that names them
    assert_allclose(np.sum(cosines**2, axis=0), 1)
    for i, name in enumerate(names):
        assert cosines['xyz'.index(name[0]), i] > 0
    # The reader's formula recovers the affine
    assert_allclose(cosines * steps, affine[:3, :3], atol=1e-14)
    assert_allclose(cosines @ starts, affine[:3, 3], atol=1e-12)


def test_affine_to_dimensions_errors():
    with pytest.raises(Minc2WriteError):
        affine_to_dimensions(np.eye(3))
    with pytest.raises(Minc2WriteError):
        affine_to_dimensions(np.diag([1, 0, 1, 1]))
    bad = np.eye(4)
    bad[:3, 1] = [1, 0, 0]
    with pytest.raises(Minc2WriteError):
        affine_to_dimensions(bad)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(4, 5, 6), (3, 4, 5, 6)])
@pytest.mark.parametrize('affine', AFFINES)
def test_round_trip_exact(tmp_path, dtype, shape, affine):
    # Integer-valued data in range round trip exactly, for every type
    data = (np.arange(np.prod(shape)) % 251).reshape(shape).astype(dtype)
    fname = tmp_path / 'img.mnc'
    write_minc2(data, affine, fname)
    img = load(fname)
    assert img.get_data_dtype() == np.dtype(dtype)
    assert img.shape == shape
    assert_array_equal(img.get_fdata(), data)
    assert_allclose(img.affine, affine, rtol=0, atol=1e-12)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(4, 5, 6), (3, 4, 5, 6)])
def test_round_trip_real(tmp_path, dtype, shape):
    # Real-valued data written as integers are scaled slice-wise
    rng = np.random.default_rng(12)
    data = rng.normal(-20, 300, size=shape)
    fname = tmp_path / 'img.mnc'
    write_minc2(data, OBLIQUE_AFFINE, fname, dtype=dtype)
    img = load(fname)
    assert img.get_data_dtype() == np.dtype(dtype)
    assert_allclose(img.affine, OBLIQUE_AFFINE, rtol=0, atol=1e-12)
    rt_data = img.get_fdata()
    if np.dtype(dtype).kind == 'f':
        assert_array_equal(rt_data, data.astype(dtype))
        return
    # Error is at most half a quantisation step of each slice
    info = np.iinfo(dtype)
    axes = (-2, -1)
    half_step = (data.max(axis=axes) - data.min(axis=axes)) / (2 * (int(info.max) - info.min))
    err = np.abs(rt_data - data).max(axis=axes)
    assert np.all(err <= half_step * (1 + 1e-6))
    # Slice minima and maxima are kept
    assert_allclose(rt_data.min(axis=axes), data.min(axis=axes), rtol=1e-12)
    assert_allclose(rt_data.max(axis=axes), data.max(axis=axes), rtol=1e-12)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_layout(tmp_path):
    shape = (2, 3, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    affine = np.array([[0, 0, -2.0, 20], [0, 2.0, 0, -20], [3.0, 0, 0, -10], [0, 0, 0, 1]])
    fname = tmp_path / 'img.mnc'
    write_minc2(data, affine, fname, history='my history', time_start=5, time_step=2.5)
    with h5py.File(fname, 'r') as h5f:
        minc = h5f['minc-2.0']
        assert minc.attrs['minc_version'] == MINC_VERSION.encode()
        assert minc.attrs['history'] == b'my history\n'
        ident = minc.attrs['ident'].decode().split(':')
        assert len(ident) == 5
        assert set(minc) == {'dimensions', 'image', 'info'}
        image = minc['image/0/image']
        assert image.shape == shape
        assert image.attrs['dimorder'] == b'time,zspace,yspace,xspace'
        assert_array_equal(image.attrs['valid_range'], [0, data.max()])
        for name in ('image-min', 'image-max'):
            var = minc['image/0'][name]
            assert var.shape == (2, 3)
            assert var.attrs['dimorder'] == b'time,zspace'
        assert_array_equal(minc['image/0/image-min'], data.min(axis=(2, 3)))
        assert_array_equal(minc['image/0/image-max'], data.max(axis=(2, 3)))
    dims = _load_dims(fname)
    assert set(dims) == {'time', 'xspace', 'yspace', 'zspace'}
    assert dims['time']['start'] == 5
    assert dims['time']['step'] == 2.5
    assert dims['time']['length'] == 2
    assert 'direction_cosines' not in dims['time']
    for name, length, start, step in (
        ('zspace', 3, -10, 3),
        ('yspace', 4, -20, 2),
        ('xspace', 5, 20, -2),
    ):
        attrs = dims[name]
        assert attrs['length'] == length
        assert attrs['start'] == start
        assert attrs['step'] == step
        assert_array_equal(attrs['direction_cosines'], np.eye(3)['xyz'.index(name[0])])
        assert attrs['spacing'] == b'regular__'
        assert attrs['alignment'] == b'centre'
    # Default history is one libminc style line
    write_minc2(data, affine, fname)
    with h5py.File(fname, 'r') as h5f:
        history = h5f['minc-2.0'].attrs['history'].decode()
    assert history.endswith(f'>>> write_minc2 {fname}\n')
    assert history.count('\n') == 1


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_integer_scaling_layout(tmp_path):
    fname = tmp_path / 'img.mnc'
    # Integral data in range: unscaled
    data = np.arange(60).reshape((3, 4, 5)) - 10
    write_minc2(data, np.eye(4), fname, dtype=np.int16)
    with h5py.File(fname, 'r') as h5f:
        group = h5f['minc-2.0/image/0']
        assert_array_equal(group['image'], data)
        assert_array_equal(group['image'].attrs['valid_range'], [-32768, 32767])
        assert_array_equal(group['image-min'], [-32768] * 3)
        assert_array_equal(group['image-max'], [32767] * 3)
    # Integral data out of range: scaled to the valid range per slice
    write_minc2(data, np.eye(4), fname, dtype=np.uint8)
    with h5py.File(fname, 'r') as h5f:
        group = h5f['minc-2.0/image/0']
        assert_array_equal(group['image-min'], data.min(axis=(1, 2)))
        assert_array_equal(group['image-max'], data.max(axis=(1, 2)))
        raw = group['image'][()]
        assert_array_equal(raw.min(axis=(1, 2)), 0)
        assert_array_equal(raw.max(axis=(1, 2)), 255)
    # Constant slices
    write_minc2(np.full((2, 3, 4), 0.5), np.eye(4), fname, dtype=np.int16)
    assert_array_equal(load(fname).get_fdata(), 0.5)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_write_errors(tmp_path):
    fname = tmp_path / 'img.mnc'
    with pytest.raises(Minc2WriteError):
        write_minc2(np.zeros((2, 3)), np.eye(4), fname)
    with pytest.raises(Minc2WriteError):
        write_minc2(np.zeros((2, 3, 4, 5, 6)), np.eye(4), fname)
    for dtype in (np.int8, np.int64, np.float16, np.complex64, bool):
        with pytest.raises(Minc2WriteError):
            write_minc2(np.zeros((2, 3, 4)), np.eye(4), fname, dtype=dtype)
    with pytest.raises(Minc2WriteError):
        write_minc2(np.full((2, 3, 4), np.nan), np.eye(4), fname, dtype=np.int16)
    with pytest.raises(Minc2WriteError):
        write_minc2(
            np.zeros((2, 3, 4)), np.eye(4), fname, dtype=np.int16, scaling=(0, [1, 2], (0, 1))
        )


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_non_finite_float(tmp_path):
    data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    data[0, 1, 1] = np.nan
    data[1, 0, 0] = np.inf
    fname = tmp_path / 'img.mnc'
    write_minc2(data, np.eye(4), fname)
    assert_array_equal(load(fname).get_fdata(), data)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('basename', MINC2_FILES)
def test_rewrite_minc2_files(tmp_path, basename):
    # Load, write with the file's scaling, load: identical data and affine
    fname = pjoin(data_path, basename)
    with pytest.warns(UserWarning) if 'baddim' in basename else nullcontext():
        img = load(fname)
    dtype = img.get_data_dtype()
    scaling = None if dtype.kind == 'f' else _file_scaling(fname)
    out = tmp_path / basename
    write_minc2(img.get_fdata(), img.affine, out, dtype=dtype, scaling=scaling)
    rt_img = load(out)
    assert rt_img.get_data_dtype() == dtype
    assert rt_img.shape == img.shape
    assert_array_equal(rt_img.get_fdata(), img.get_fdata())
    assert_array_equal(rt_img.affine, img.affine)
    with h5py.File(fname, 'r') as src, h5py.File(out, 'r') as dst:
        assert_array_equal(dst['minc-2.0/image/0/image'], src['minc-2.0/image/0/image'])

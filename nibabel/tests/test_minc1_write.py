# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for the MINC1 writer module, independent of ``Minc1Image``"""

from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import load
from ..affines import from_matvec
from ..eulerangles import euler2mat
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1File, Minc1Image
from ..minc1_write import MINC1_DTYPES, write_minc1
from ..minc2_write import MincWriteError
from ..optpkg import optional_package
from ..testing import data_path

h5py, have_h5py, _ = optional_package('h5py')

OBLIQUE_AFFINE = from_matvec(euler2mat(0.3, -0.2, 0.5) @ np.diag([1.5, -2.0, 0.7]), [10, -20, 5])
AFFINES = (
    np.diag([1.0, 2.0, 3.0, 1.0]),
    np.array([[0, 0, -2.0, 20], [0, 2.0, 0, -20], [3.0, 0, 0, -10], [0, 0, 0, 1]]),
    OBLIQUE_AFFINE,
)


def _netcdf(fname):
    with open(fname, 'rb') as fobj:
        return netcdf_file(BytesIO(fobj.read()), 'r', mmap=False)


@pytest.mark.parametrize('dtype', MINC1_DTYPES)
@pytest.mark.parametrize('shape', [(4, 5, 6), (3, 4, 5, 6)])
@pytest.mark.parametrize('affine', AFFINES)
def test_round_trip_exact(tmp_path, dtype, shape, affine):
    data = (np.arange(np.prod(shape)).reshape(shape) % 100).astype(dtype)
    if np.dtype(dtype).kind == 'i':
        data = data - 50
    fname = tmp_path / 'img.mnc'
    write_minc1(data, affine, fname)
    img = load(fname)
    assert isinstance(img, Minc1Image)
    assert img.get_data_dtype().newbyteorder('=') == dtype
    assert_array_equal(img.get_fdata(), data)
    assert_allclose(img.affine, affine, rtol=0, atol=1e-12)


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.int32])
def test_round_trip_real(tmp_path, dtype):
    rng = np.random.default_rng(1)
    data = rng.normal(100, 30, (3, 4, 5))
    fname = tmp_path / 'img.mnc'
    write_minc1(data, OBLIQUE_AFFINE, fname, dtype=dtype)
    rt_data = load(fname).get_fdata()
    half_step = (data.max(axis=(1, 2)) - data.min(axis=(1, 2))) / (
        2 * (np.iinfo(dtype).max - np.iinfo(dtype).min)
    )
    assert np.all(np.abs(rt_data - data) <= half_step[:, None, None] * (1 + 1e-6))


def test_layout(tmp_path):
    shape = (2, 3, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    fname = tmp_path / 'img.mnc'
    affine = np.array([[0, 0, 1.0, 0], [0, 2.0, 0, 0], [3.0, 0, 0, 0], [0, 0, 0, 1]])
    write_minc1(data, affine, fname, history='my history')
    nc = _netcdf(fname)
    # Variables in the order libminc makes them, with the image last
    assert list(nc.variables) == [
        'time',
        'zspace',
        'yspace',
        'xspace',
        'image-max',
        'image-min',
        'study',
        'rootvariable',
        'patient',
        'acquisition',
        'image',
    ]
    assert nc.history == b'my history\n'
    assert len(nc.ident.split(b':')) == 5
    assert nc.minc_version == b'2.4.01'
    image = nc.variables['image']
    assert image.dimensions == ('time', 'zspace', 'yspace', 'xspace')
    assert image.typecode() == 'h'
    assert image.signtype == b'unsigned'
    assert image.parent == b'rootvariable'
    assert image._attributes['image-max'] == b'--->image-max'
    assert_array_equal(image.valid_range, [0, 65535])
    for name in ('image-max', 'image-min'):
        var = nc.variables[name]
        assert var.dimensions == ('time', 'zspace')
        assert var.parent == b'image'
        assert var.vartype == b'var_attribute'
    assert nc.variables['rootvariable'].children == b'study\npatient\nacquisition\nimage'
    assert nc.variables['study'].parent == b'rootvariable'
    time = nc.variables['time']
    assert time.shape == ()
    assert time.alignment == b'start_'
    assert time.units == b's'
    assert not hasattr(time, 'length')
    xspace = nc.variables['xspace']
    assert list(xspace._attributes) == [
        'varid',
        'vartype',
        'version',
        'comments',
        'spacing',
        'alignment',
        'step',
        'units',
        'start',
        'direction_cosines',
    ]
    assert xspace.step == 1
    assert_array_equal(xspace.direction_cosines, [1, 0, 0])
    del nc


def test_float_valid_range(tmp_path):
    # libminc writes the valid range of float images as float
    fname = tmp_path / 'img.mnc'
    data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    write_minc1(data, np.eye(4), fname)
    nc = _netcdf(fname)
    assert nc.variables['image'].typecode() == 'f'
    assert nc.variables['image'].valid_range.dtype == np.dtype('>f4')
    assert not hasattr(nc.variables['image'], 'signtype')
    del nc
    assert_array_equal(load(fname).get_fdata(), data)


def test_file_object(tmp_path):
    data = np.arange(60, dtype=np.int16).reshape((3, 4, 5))
    bio = BytesIO(b'x' * 100000)
    write_minc1(data, np.eye(4), bio)
    bio.seek(0)
    minc_file = Minc1File(netcdf_file(bio, 'r', mmap=False))
    assert_array_equal(minc_file.get_scaled_data(), data)


def test_write_errors(tmp_path):
    fname = tmp_path / 'img.mnc'
    for dtype in (np.int64, np.float16, np.complex64):
        with pytest.raises(MincWriteError):
            write_minc1(np.zeros((2, 3, 4)), np.eye(4), fname, dtype=dtype)
    with pytest.raises(MincWriteError):
        write_minc1(np.zeros((2, 3)), np.eye(4), fname)


@pytest.mark.parametrize('src', ['minc1_4d.mnc', 'tiny.mnc', 'minc1_1_scale.mnc'])
def test_write_metadata_minc1(tmp_path, src):
    img = load(pjoin(data_path, src))
    meta = img.minc_meta
    fname = tmp_path / 'img.mnc'
    write_minc1(img.get_fdata(), img.affine, fname, metadata=meta)
    rt_meta = load(fname).minc_meta
    assert set(rt_meta.info) == set(meta.info)
    for name, var in meta.info.items():
        assert dict(rt_meta.info[name].attrs) == dict(var.attrs) | {'parent': 'rootvariable'}
    for name, var in meta.dimensions.items():
        assert dict(rt_meta.dimensions[name].attrs).keys() == dict(var.attrs).keys()
    assert rt_meta.history.startswith(meta.history)
    assert rt_meta.history.endswith(f'>>> write_minc1 {fname}\n')


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_write_metadata_minc2(tmp_path):
    # MINC2 metadata: the info groups become variables under the root
    img = load(pjoin(data_path, 'minc2_baddim.mnc'))
    fname = tmp_path / 'img.mnc'
    write_minc1(img.get_fdata(), img.affine, fname, metadata=img.minc_meta, dtype=np.int16)
    nc = _netcdf(fname)
    processing = nc.variables['processing']
    assert processing.parent == b'rootvariable'
    assert processing._attributes['transformation0-filename'] == b'D4600.xfm'
    assert nc.variables['rootvariable'].children == b'processing\nimage'
    del nc
    assert_array_equal(load(fname).get_fdata(), img.get_fdata())


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_irregular_time(tmp_path):
    # Irregular times from a MINC2 file are written to MINC1
    from ..minc2_write import write_minc2
    from .test_minc2_write import _make_irregular

    src = tmp_path / 'src.mnc'
    data = np.arange(120, dtype=np.float32).reshape((5, 2, 3, 4))
    write_minc2(data, np.eye(4), src)
    times = [0, 1, 3, 7, 15]
    _make_irregular(src, 'time', times, [1, 2, 4, 8, 16])
    img = load(src)
    out = tmp_path / 'out.mnc'
    write_minc1(img.get_fdata(), img.affine, out, metadata=img.minc_meta)
    nc = _netcdf(out)
    time = nc.variables['time']
    assert time.spacing == b'irregular'
    assert time.dimensions == ('time',)
    assert_array_equal(time[:], times)
    assert_array_equal(nc.variables['time-width'][:], [1, 2, 4, 8, 16])
    del nc
    rt_img = load(out)
    assert_array_equal(rt_img.get_fdata(), data)
    assert_array_equal(rt_img.minc_meta.variables['time'].data, times)
    assert rt_img.minc_meta.variables['time-width'].group == 'dimensions'

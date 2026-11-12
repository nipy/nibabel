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
The tests against the MINC tools need ``mincinfo``, ``mincstats``,
``mnc2nii`` and ``nii2mnc`` on the ``PATH``, and skip without them.  Tests
that run the tools on MINC2 files also skip if the tools were built without
HDF5 support (libminc 1.x).
"""

import shutil
import subprocess
from contextlib import nullcontext
from functools import cache
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import Nifti1Header, Nifti1Image, load
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


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_write_metadata(tmp_path):
    # Info variables, and global, image and dimension attributes are copied
    src = load(pjoin(data_path, 'minc2_baddim.mnc'))
    meta = src.minc_meta
    fname = tmp_path / 'img.mnc'
    affine = np.array([[0, 0, 1.0, 0], [0, 2.0, 0, 0], [3.0, 0, 0, 0], [0, 0, 0, 1]])
    write_minc2(np.zeros(src.shape, np.int16), affine, fname, metadata=meta)
    with h5py.File(fname, 'r') as h5f:
        info = h5f['minc-2.0/info']
        assert set(info) == {'processing'}
        attrs = info['processing'].attrs
        assert attrs['transformation0-filename'] == b'D4600.xfm'
        assert attrs['transformation0-filedata'].decode().startswith('MNI Transform File')
    # The geometry comes from the affine, not from the metadata
    dims = _load_dims(fname)
    assert dims['zspace']['start'] == 0
    assert dims['zspace']['step'] == 3
    assert dims['xspace']['step'] == 1
    assert dims['xspace']['spacing'] == b'regular__'
    assert dims['xspace']['length'] == src.shape[2]
    # MINC1 metadata gives the same MINC2 variables
    meta = load(pjoin(data_path, 'minc1_4d.mnc')).minc_meta
    write_minc2(np.zeros((2, 3, 4, 5)), np.eye(4), fname, metadata=meta)
    with h5py.File(fname, 'r') as h5f:
        study = h5f['minc-2.0/info/study']
        assert study.attrs['modality'] == b'MRI__'
        assert 'parent' not in study.attrs
        image = h5f['minc-2.0/image/0/image']
        assert 'signtype' not in image.attrs
        assert 'image-max' not in image.attrs
    # Extra global attributes are copied, but not ident and minc_version
    meta = load(pjoin(data_path, 'minc2-4d-d.mnc')).minc_meta
    write_minc2(np.zeros((2, 3, 4, 5)), np.eye(4), fname, metadata=meta)
    with h5py.File(fname, 'r') as h5f:
        attrs = h5f['minc-2.0'].attrs
        assert attrs['class'] == b'real___'
        assert attrs['minc_version'] == MINC_VERSION.encode()
        assert attrs['ident'] != meta.global_attrs['ident'].encode()
        assert h5f['minc-2.0/image/0/image'].attrs['version'] == b'MINC Version    2.0'


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_history_appended(tmp_path):
    meta = load(pjoin(data_path, 'minc2_1_scale.mnc')).minc_meta
    old = meta.history
    assert old.endswith('\n')
    fname = tmp_path / 'img.mnc'
    write_minc2(np.zeros((2, 3, 4)), np.eye(4), fname, metadata=meta)
    with h5py.File(fname, 'r') as h5f:
        history = h5f['minc-2.0'].attrs['history'].decode()
    assert history.startswith(old)
    assert history[len(old) :].endswith(f'>>> write_minc2 {fname}\n')
    assert history.count('\n') == old.count('\n') + 1
    # An explicit history replaces the old one
    write_minc2(np.zeros((2, 3, 4)), np.eye(4), fname, metadata=meta, history='new')
    with h5py.File(fname, 'r') as h5f:
        assert h5f['minc-2.0'].attrs['history'] == b'new\n'
    # Metadata without history gives a single line
    meta = load(pjoin(data_path, 'minc2-4d-d.mnc')).minc_meta
    write_minc2(np.zeros((2, 3, 4, 5)), np.eye(4), fname, metadata=meta)
    with h5py.File(fname, 'r') as h5f:
        assert h5f['minc-2.0'].attrs['history'].decode().count('\n') == 1


# Tests against the MINC tools


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f'{name} not on PATH')
    return path


def _run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


@cache
def _tools_read_minc2():
    """True if the MINC tools on the PATH can read MINC2 (HDF5) files"""
    path = shutil.which('mincinfo')
    if path is None:
        return False
    fname = pjoin(data_path, 'minc2_1_scale.mnc')
    result = subprocess.run([path, fname], capture_output=True, text=True)
    return result.returncode == 0


def _need_minc2_tools(*names):
    for name in names:
        _tool(name)
    if not _tools_read_minc2():
        pytest.skip('MINC tools cannot read MINC2 (built without HDF5)')


def _mincinfo_dims(fname):
    """Return [(name, length, start, step, cosines)] from mincinfo"""
    mincinfo = _tool('mincinfo')
    names = _run(mincinfo, '-dimnames', fname).split()
    dims = []
    for name in names:
        length = int(_run(mincinfo, '-dimlength', name, fname))
        start = float(_run(mincinfo, '-attvalue', f'{name}:start', fname))
        step = float(_run(mincinfo, '-attvalue', f'{name}:step', fname))
        cosines = None
        if name.endswith('space'):
            cos_text = _run(mincinfo, '-attvalue', f'{name}:direction_cosines', fname)
            cosines = np.array(cos_text.split(), dtype=float)
        dims.append((name, length, start, step, cosines))
    return dims


def _minc2_dims(fname):
    """Return [(name, length, start, step, cosines)] read with h5py"""
    with h5py.File(fname, 'r') as h5f:
        dimorder = h5f['minc-2.0/image/0/image'].attrs['dimorder'].decode().split(',')
    attrs = _load_dims(fname)
    return [
        (
            name,
            int(attrs[name]['length']),
            float(attrs[name]['start']),
            float(attrs[name]['step']),
            attrs[name].get('direction_cosines'),
        )
        for name in dimorder
    ]


def _assert_dims_close(dims_a, dims_b):
    assert [d[:2] for d in dims_a] == [d[:2] for d in dims_b]
    for (_, _, start_a, step_a, cos_a), (_, _, start_b, step_b, cos_b) in zip(dims_a, dims_b):
        assert_allclose(start_a, start_b, rtol=1e-5, atol=1e-5)
        assert_allclose(step_a, step_b, rtol=1e-6)
        if cos_a is not None:
            assert_allclose(cos_a, cos_b, atol=1e-6)


def _mincstats(fname):
    mincstats = _tool('mincstats')
    out = _run(mincstats, '-quiet', '-min', '-max', '-mean', fname)
    return np.array(out.split(), dtype=float)


def _assert_same_in_world(ref_data, ref_affine, data, affine, atol):
    """Compare two images whose voxel grids coincide up to axis order and flips"""
    assert data.size == ref_data.size
    ijk = np.indices(data.shape).reshape(data.ndim, -1)
    vox2vox = np.linalg.solve(ref_affine, affine)
    ref_ijk = vox2vox[:3, :3] @ ijk + vox2vox[:3, 3:]
    assert_allclose(ref_ijk, np.round(ref_ijk), atol=1e-3)
    ref_ijk = np.round(ref_ijk).astype(int)
    assert_allclose(data.reshape(-1), ref_data[tuple(ref_ijk)], atol=atol)


def _read_mnc2nii(fname):
    """Read data, affine and zooms of a NIfTI file written by ``mnc2nii``

    ``mnc2nii`` from libminc 1.5 writes a single file with ``vox_offset``
    348, which nibabel's loader rejects, and sets neither the sform nor the
    qform, so we parse the header and data directly.  The affine is None if
    the file has no sform or qform.
    """
    with open(fname, 'rb') as fobj:
        raw = fobj.read()
    hdr = Nifti1Header(binaryblock=raw[:348], check=False)
    shape = hdr.get_data_shape()
    data = np.frombuffer(
        raw, hdr.get_data_dtype(), count=int(np.prod(shape)), offset=int(hdr['vox_offset'])
    ).reshape(shape, order='F')
    has_affine = hdr['sform_code'] > 0 or hdr['qform_code'] > 0
    return data, hdr.get_best_affine() if has_affine else None, hdr.get_zooms()


def _check_mnc2nii(minc_fname, data, affine, tmp_path, atol):
    """Convert `minc_fname` with mnc2nii and compare with `data`, `affine`"""
    nii_fname = str(tmp_path / 'mnc2nii.nii')
    _run(_tool('mnc2nii'), '-quiet', '-float', minc_fname, nii_fname)
    nii_data, nii_affine, nii_zooms = _read_mnc2nii(nii_fname)
    if nii_affine is not None:
        _assert_same_in_world(data, affine, nii_data, nii_affine, atol)
        return
    # Without an affine, mnc2nii's axes are xspace, yspace, zspace
    dims = _mincinfo_dims(minc_fname)
    names = [dim[0] for dim in dims]
    steps = np.array([dim[3] for dim in dims])
    assert np.all(steps > 0)
    order = [names.index(name) for name in ('xspace', 'yspace', 'zspace')]
    assert_allclose(nii_data, data.transpose(order), atol=atol)
    assert_allclose(nii_zooms, steps[order], rtol=1e-6)


def _write_oblique(fname, dtype=np.float32):
    rng = np.random.default_rng(3)
    data = rng.normal(100, 30, (5, 6, 7)).astype(dtype)
    write_minc2(data, OBLIQUE_AFFINE, fname)
    return data


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_mincinfo_minc2(tmp_path):
    _need_minc2_tools('mincinfo')
    fname = str(tmp_path / 'img.mnc')
    _write_oblique(fname)
    _assert_dims_close(_mincinfo_dims(fname), _minc2_dims(fname))


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('dtype', DTYPES)
def test_mincstats_minc2(tmp_path, dtype):
    _need_minc2_tools('mincstats')
    fname = str(tmp_path / 'img.mnc')
    rng = np.random.default_rng(4)
    write_minc2(rng.normal(100, 30, (5, 6, 7)), OBLIQUE_AFFINE, fname, dtype=dtype)
    data = load(fname).get_fdata()
    assert_allclose(_mincstats(fname), [data.min(), data.max(), data.mean()], rtol=1e-6)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_mnc2nii_minc2(tmp_path):
    _need_minc2_tools('mnc2nii')
    fname = str(tmp_path / 'img.mnc')
    data = _write_oblique(fname)
    _check_mnc2nii(fname, data, OBLIQUE_AFFINE, tmp_path, atol=1e-4)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('dtype', [np.float32, np.int16, np.uint8])
def test_nii2mnc_rewrite(tmp_path, dtype):
    # NIfTI -> nii2mnc (libminc) -> nibabel -> write_minc2.  This checks the
    # start / step / cosine conversion against libminc's own, and works with
    # MINC tools that write MINC1 only.
    nii2mnc = _tool('nii2mnc')
    _tool('mnc2nii')
    rng = np.random.default_rng(5)
    affine = from_matvec(euler2mat(0.3, -0.2, 0.5) @ np.diag([1.5, 2.0, 0.7]), [10, -20, 5])
    nii_data = rng.uniform(0, 200, (7, 6, 5)).astype(dtype)
    nii_fname = str(tmp_path / 'in.nii')
    Nifti1Image(nii_data, affine).to_filename(nii_fname)
    lib_fname = str(tmp_path / 'libminc.mnc')
    _run(nii2mnc, '-quiet', nii_fname, lib_fname)
    lib_img = load(lib_fname)
    lib_data = lib_img.get_fdata()
    _assert_same_in_world(nii_data, affine, lib_data, lib_img.affine, atol=1e-4)
    out_fname = str(tmp_path / 'out.mnc')
    write_minc2(lib_data, lib_img.affine, out_fname, dtype=lib_img.get_data_dtype())
    out_img = load(out_fname)
    out_data = out_img.get_fdata()
    tol = 0 if np.dtype(dtype).kind == 'f' else 1e-6 * np.ptp(lib_data)
    assert_allclose(out_data, lib_data, rtol=0, atol=tol)
    assert_allclose(out_img.affine, lib_img.affine, rtol=0, atol=1e-12)
    # Dimension names, order, starts, steps and cosines are those libminc chose
    _assert_dims_close(_minc2_dims(out_fname), _mincinfo_dims(lib_fname))
    # mincstats of the libminc file matches numpy on the re-written data
    assert_allclose(
        _mincstats(lib_fname), [out_data.min(), out_data.max(), out_data.mean()], rtol=1e-6
    )
    # mnc2nii of the libminc file matches the re-written data
    _check_mnc2nii(lib_fname, out_data, out_img.affine, tmp_path, atol=1e-3)


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_nii2mnc_rewrite_mincinfo(tmp_path):
    # As above, then mincinfo on the re-written MINC2 file
    _need_minc2_tools('nii2mnc', 'mincinfo')
    nii_fname = str(tmp_path / 'in.nii')
    affine = from_matvec(euler2mat(0.3, -0.2, 0.5) @ np.diag([1.5, 2.0, 0.7]), [10, -20, 5])
    Nifti1Image(np.ones((7, 6, 5), dtype=np.float32), affine).to_filename(nii_fname)
    lib_fname = str(tmp_path / 'libminc.mnc')
    _run(_tool('nii2mnc'), '-quiet', nii_fname, lib_fname)
    lib_img = load(lib_fname)
    out_fname = str(tmp_path / 'out.mnc')
    write_minc2(lib_img.get_fdata(), lib_img.affine, out_fname)
    _assert_dims_close(_mincinfo_dims(out_fname), _mincinfo_dims(lib_fname))

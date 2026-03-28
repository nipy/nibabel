# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for MIF (.mif / .mif.gz) image format support."""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pytest

import nibabel as nib
from nibabel.imageclasses import all_image_classes, KNOWN_SPATIAL_FIRST
from nibabel.mif import (
    MifArrayProxy,
    MifHeader,
    MifImage,
    _mif_apply_layout,
    _mif_apply_layout_for_write,
    _mif_dtype_to_str,
    _mif_layout_to_str,
    _mif_parse_dtype,
    _mif_parse_layout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mif_bytes(shape=(3, 4, 5), dtype=np.float32, layout=None, transform=None,
                    intensity_offset=0.0, intensity_scale=1.0, keyval=None):
    """Build a MIF file in memory and return the bytes."""
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    affine = np.eye(4)
    if layout is None:
        layout = list(range(1, len(shape) + 1))
    hdr = MifHeader(
        shape=shape,
        zooms=(1.0,) * len(shape),
        layout=layout,
        dtype=np.dtype(dtype),
        transform=transform if transform is not None else np.eye(3, 4),
        intensity_offset=intensity_offset,
        intensity_scale=intensity_scale,
        keyval=keyval or {},
    )
    img = MifImage(data, affine, header=hdr)
    buf = io.BytesIO()
    img.to_stream(buf)
    return buf.getvalue(), data


# ---------------------------------------------------------------------------
# _mif_parse_dtype / _mif_dtype_to_str
# ---------------------------------------------------------------------------

class TestMifDtype:
    @pytest.mark.parametrize('mif_str, np_str', [
        ('Int8', 'i1'),
        ('UInt8', 'u1'),
        ('Int16LE', '<i2'),
        ('Int16BE', '>i2'),
        ('UInt16LE', '<u2'),
        ('Float32LE', '<f4'),
        ('Float32BE', '>f4'),
        ('Float64LE', '<f8'),
        ('CFloat32LE', '<c8'),
        ('CFloat64BE', '>c16'),
    ])
    def test_parse(self, mif_str, np_str):
        assert _mif_parse_dtype(mif_str) == np.dtype(np_str)

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown MIF datatype'):
            _mif_parse_dtype('Bogus32LE')

    @pytest.mark.parametrize('dtype', [
        np.dtype('i1'), np.dtype('u1'),
        np.dtype('<i2'), np.dtype('>i2'),
        np.dtype('<f4'), np.dtype('<f8'),
        np.dtype('<c8'), np.dtype('<c16'),
    ])
    def test_roundtrip(self, dtype):
        assert _mif_parse_dtype(_mif_dtype_to_str(dtype)) == dtype

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match='Cannot represent'):
            _mif_dtype_to_str(np.dtype('f2'))  # float16 not in MIF spec


# ---------------------------------------------------------------------------
# _mif_parse_layout / _mif_layout_to_str
# ---------------------------------------------------------------------------

class TestMifLayout:
    def test_parse_positive(self):
        assert _mif_parse_layout('+0,+1,+2', 3) == [1, 2, 3]

    def test_parse_negative(self):
        assert _mif_parse_layout('-0,-1,+2', 3) == [-1, -2, 3]

    def test_parse_no_sign(self):
        assert _mif_parse_layout('0,1,2', 3) == [1, 2, 3]

    def test_parse_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match='Layout has'):
            _mif_parse_layout('+0,+1', 3)

    def test_layout_str_roundtrip(self):
        for layout in ([1, 2, 3], [-1, 2, -3], [3, 1, 2]):
            assert _mif_parse_layout(_mif_layout_to_str(layout), len(layout)) == layout


# ---------------------------------------------------------------------------
# _mif_apply_layout / _mif_apply_layout_for_write
# ---------------------------------------------------------------------------

class TestMifApplyLayout:
    def test_identity_layout(self):
        shape = (2, 3, 4)
        data = np.arange(24).reshape(shape)
        flat = _mif_apply_layout_for_write(data, [1, 2, 3])
        # disk layout for +0,+1,+2: axis 0 fastest → C-order with axes reversed
        recovered = _mif_apply_layout(flat.ravel(), shape, [1, 2, 3])
        assert np.array_equal(recovered, data)

    def test_reversed_layout_roundtrip(self):
        """Reversed layout [-1,-2,-3]: data stored with axis 0 slowest."""
        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.float32).reshape(shape)
        layout = [-1, -2, -3]
        disk = _mif_apply_layout_for_write(data, layout)
        recovered = _mif_apply_layout(disk.ravel(), shape, layout)
        assert np.array_equal(recovered, data)

    def test_permuted_layout_roundtrip(self):
        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.float32).reshape(shape)
        layout = [3, 1, 2]  # axis 1 is fastest on disk
        disk = _mif_apply_layout_for_write(data, layout)
        recovered = _mif_apply_layout(disk.ravel(), shape, layout)
        assert np.array_equal(recovered, data)


# ---------------------------------------------------------------------------
# MifHeader
# ---------------------------------------------------------------------------

class TestMifHeader:
    def test_default_init(self):
        hdr = MifHeader()
        assert hdr.get_data_shape() == (1,)
        assert hdr.get_zooms() == (1.0,)
        assert hdr.get_data_dtype() == np.dtype('f4')
        assert hdr.get_layout() == [1]

    def test_custom_init(self):
        hdr = MifHeader(shape=(3, 4, 5), zooms=(2.0, 2.0, 3.0), dtype=np.dtype('i2'))
        assert hdr.get_data_shape() == (3, 4, 5)
        assert hdr.get_zooms() == (2.0, 2.0, 3.0)
        assert hdr.get_data_dtype() == np.dtype('i2')

    def test_from_fileobj_roundtrip(self):
        hdr = MifHeader(
            shape=(3, 4, 5),
            zooms=(1.5, 2.0, 2.5),
            layout=[1, 2, 3],
            dtype=np.dtype('<f4'),
            intensity_offset=1.0,
            intensity_scale=2.0,
            keyval={'comment': 'hello'},
        )
        buf = io.BytesIO()
        hdr.write_to(buf)
        buf.seek(0)
        hdr2 = MifHeader.from_fileobj(buf)
        assert hdr == hdr2

    def test_may_contain_header(self):
        assert MifHeader.may_contain_header(b'mrtrix image\n')
        assert MifHeader.may_contain_header(b'mrtrix image\nextra bytes')
        assert not MifHeader.may_contain_header(b'not a mif')
        assert not MifHeader.may_contain_header(b'')

    def test_get_best_affine_identity(self):
        hdr = MifHeader(shape=(3, 4, 5), zooms=(1.0, 1.0, 1.0))
        affine = hdr.get_best_affine()
        assert affine.shape == (4, 4)
        assert affine[3, 3] == 1.0

    def test_get_best_affine_zooms(self):
        hdr = MifHeader(shape=(3, 4, 5), zooms=(2.0, 3.0, 4.0))
        affine = hdr.get_best_affine()
        assert affine[0, 0] == pytest.approx(2.0)
        assert affine[1, 1] == pytest.approx(3.0)
        assert affine[2, 2] == pytest.approx(4.0)

    def test_get_best_affine_negative_layout(self):
        """Negative layout strides flip the affine column and shift the origin."""
        hdr = MifHeader(shape=(4, 4, 4), zooms=(1.0, 1.0, 1.0), layout=[-1, 2, 3])
        affine_pos = MifHeader(shape=(4, 4, 4), zooms=(1.0, 1.0, 1.0), layout=[1, 2, 3])
        aff_neg = hdr.get_best_affine()
        aff_pos = affine_pos.get_best_affine()
        # Column 0 should be negated
        assert np.allclose(aff_neg[:3, 0], -aff_pos[:3, 0])

    def test_intensity_scaling(self):
        hdr = MifHeader(intensity_offset=1.5, intensity_scale=0.5)
        assert hdr.get_intensity_scaling() == (1.5, 0.5)

    def test_keyval_stored(self):
        hdr = MifHeader(keyval={'foo': 'bar', 'baz': 'qux'})
        kv = hdr.get_keyval()
        assert kv['foo'] == 'bar'
        assert kv['baz'] == 'qux'

    def test_equality(self):
        hdr1 = MifHeader(shape=(3, 4, 5))
        hdr2 = MifHeader(shape=(3, 4, 5))
        hdr3 = MifHeader(shape=(3, 4, 6))
        assert hdr1 == hdr2
        assert hdr1 != hdr3

    def test_copy(self):
        hdr = MifHeader(shape=(3, 4, 5), keyval={'x': 'y'})
        hdr2 = hdr.copy()
        assert hdr == hdr2
        hdr2._keyval['x'] = 'z'
        assert hdr.get_keyval()['x'] == 'y'  # original unaffected

    def test_missing_dim_raises(self):
        buf = io.BytesIO(b'mrtrix image\nvox: 1,1,1\ndatatype: Float32LE\nlayout: +0,+1,+2\nfile: . 64\nEND\n')
        with pytest.raises(ValueError, match='Missing "dim"'):
            MifHeader.from_fileobj(buf)

    def test_invalid_magic_raises(self):
        buf = io.BytesIO(b'not a mif file\n')
        with pytest.raises(ValueError, match='Not a MIF file'):
            MifHeader.from_fileobj(buf)

    def test_set_data_shape(self):
        hdr = MifHeader()
        hdr.set_data_shape((2, 3, 4))
        assert hdr.get_data_shape() == (2, 3, 4)

    def test_set_zooms(self):
        hdr = MifHeader(shape=(2, 3))
        hdr.set_zooms((2.5, 3.5))
        assert hdr.get_zooms() == (2.5, 3.5)

    def test_set_data_dtype(self):
        hdr = MifHeader()
        hdr.set_data_dtype(np.dtype('i2'))
        assert hdr.get_data_dtype() == np.dtype('i2')


# ---------------------------------------------------------------------------
# MifArrayProxy
# ---------------------------------------------------------------------------

class TestMifArrayProxy:
    def test_lazy_load(self, tmp_path):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)

        img2 = MifImage.load(fname)
        assert not img2.in_memory
        loaded = img2.get_fdata()
        assert img2.in_memory
        assert np.allclose(loaded, data)

    def test_proxy_shape(self, tmp_path):
        data = np.zeros((5, 6, 7), dtype=np.float32)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)

        img2 = MifImage.load(fname)
        assert img2.shape == (5, 6, 7)

    def test_proxy_slicing(self, tmp_path):
        data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)

        img2 = MifImage.load(fname)
        sliced = img2.dataobj[0, :, :, :]
        assert np.allclose(sliced, data[0])

    def test_intensity_scaling_proxy(self, tmp_path):
        """Proxy should apply intensity scaling from the header."""
        raw = np.ones((3, 3, 3), dtype=np.int16) * 10
        hdr = MifHeader(
            shape=(3, 3, 3), zooms=(1.0, 1.0, 1.0), dtype=np.dtype('<i2'),
            intensity_offset=5.0, intensity_scale=2.0,
        )
        img = MifImage(raw, np.eye(4), header=hdr)
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)

        img2 = MifImage.load(fname)
        fdata = img2.get_fdata()
        # expected: 10 * 2.0 + 5.0 = 25.0
        assert np.allclose(fdata, 25.0)


# ---------------------------------------------------------------------------
# MifImage
# ---------------------------------------------------------------------------

class TestMifImage:
    def test_create_from_array(self):
        data = np.zeros((3, 4, 5))
        img = MifImage(data, np.eye(4))
        assert img.shape == (3, 4, 5)

    def test_load_save_roundtrip_3d(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert np.allclose(img2.get_fdata(), data)

    def test_load_save_roundtrip_4d(self, tmp_path):
        data = np.random.default_rng(1).random((4, 5, 6, 7)).astype(np.float32)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert np.allclose(img2.get_fdata(), data)

    def test_affine_preserved(self, tmp_path):
        data = np.ones((3, 4, 5), dtype=np.float32)
        affine = np.array([
            [2, 0, 0, -10],
            [0, 2, 0, -12],
            [0, 0, 2, -14],
            [0, 0, 0,   1],
        ], dtype=np.float64)
        img = MifImage(data, affine)
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert np.allclose(img2.affine, affine)

    @pytest.mark.parametrize('dtype', [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.float32, np.float64,
    ])
    def test_dtypes(self, tmp_path, dtype):
        data = np.arange(60, dtype=dtype).reshape(3, 4, 5)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert np.allclose(img2.get_fdata(), data)

    def test_gz_roundtrip(self, tmp_path):
        data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif.gz')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert np.allclose(img2.get_fdata(), data)

    def test_nibabel_load_detection(self, tmp_path):
        """nibabel.load() should auto-detect and return a MifImage."""
        data = np.ones((2, 3, 4), dtype=np.float32)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = nib.load(fname)
        assert isinstance(img2, MifImage)

    def test_intensity_scaling_roundtrip(self, tmp_path):
        raw = np.ones((3, 3, 3), dtype=np.int16)
        hdr = MifHeader(
            shape=(3, 3, 3), zooms=(1.0, 1.0, 1.0), dtype=np.dtype('<i2'),
            intensity_offset=0.5, intensity_scale=3.0,
        )
        img = MifImage(raw, np.eye(4), header=hdr)
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)

        img2 = MifImage.load(fname)
        # expected: 1 * 3.0 + 0.5 = 3.5
        assert np.allclose(img2.get_fdata(), 3.5)
        # Raw on-disk data should still be 1 (unscaled)
        assert np.all(img2.dataobj.get_unscaled() == 1)

    def test_intensity_scaling_save_reload(self, tmp_path):
        """Re-saving a proxy-backed image should not double-apply scaling."""
        raw = np.ones((3, 3, 3), dtype=np.int16)
        hdr = MifHeader(
            shape=(3, 3, 3), zooms=(1.0, 1.0, 1.0), dtype=np.dtype('<i2'),
            intensity_offset=0.0, intensity_scale=2.0,
        )
        img = MifImage(raw, np.eye(4), header=hdr)
        fname1 = str(tmp_path / 'a.mif')
        img.to_filename(fname1)

        img2 = MifImage.load(fname1)
        fname2 = str(tmp_path / 'b.mif')
        img2.to_filename(fname2)  # re-save proxy-backed image

        img3 = MifImage.load(fname2)
        # Should still be 1*2+0=2, not 1*2*2=4
        assert np.allclose(img3.get_fdata(), 2.0)

    @pytest.mark.parametrize('layout', [
        [1, 2, 3],
        [-1, 2, 3],
        [3, 1, 2],
        [-3, -2, -1],
    ])
    def test_layout_roundtrip(self, tmp_path, layout):
        shape = (3, 4, 5)
        data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        hdr = MifHeader(shape=shape, zooms=(1.0,) * 3, layout=layout)
        img = MifImage(data, np.eye(4), header=hdr)
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert np.allclose(img2.get_fdata(), data)

    def test_bytes_roundtrip(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        img = MifImage(data, np.eye(4))
        buf = img.to_bytes()
        img2 = MifImage.from_bytes(buf)
        assert np.allclose(img2.get_fdata(), data)

    def test_keyval_preserved(self, tmp_path):
        data = np.ones((2, 2, 2), dtype=np.float32)
        hdr = MifHeader(shape=(2, 2, 2), zooms=(1.0, 1.0, 1.0),
                        keyval={'comment': 'hello', 'mrtrix_version': '3.0'})
        img = MifImage(data, np.eye(4), header=hdr)
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        kv = img2.header.get_keyval()
        assert kv.get('comment') == 'hello'
        assert kv.get('mrtrix_version') == '3.0'

    def test_uncache(self, tmp_path):
        data = np.ones((2, 3, 4), dtype=np.float32)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        _ = img2.get_fdata()
        assert img2.in_memory
        img2.uncache()
        assert not img2.in_memory

    def test_shape_from_header(self, tmp_path):
        data = np.zeros((5, 6, 7, 3), dtype=np.float32)
        img = MifImage(data, np.eye(4))
        fname = str(tmp_path / 'test.mif')
        img.to_filename(fname)
        img2 = MifImage.load(fname)
        assert img2.shape == (5, 6, 7, 3)


# ---------------------------------------------------------------------------
# Integration / registration
# ---------------------------------------------------------------------------

class TestMifRegistration:
    def test_in_all_image_classes(self):
        assert any(c is MifImage for c in all_image_classes)

    def test_in_known_spatial_first(self):
        assert MifImage in KNOWN_SPATIAL_FIRST

    def test_in_nibabel_namespace(self):
        assert nib.MifImage is MifImage
        assert nib.MifHeader is MifHeader

    def test_meta_sniff_len(self):
        assert MifImage._meta_sniff_len == 13

    def test_compressed_suffixes(self):
        assert '.gz' in MifImage._compressed_suffixes
        assert '.bz2' in MifImage._compressed_suffixes
        assert '.zst' in MifImage._compressed_suffixes

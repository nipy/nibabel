"""Tests for optional ISA-L accelerated gzip writing."""

import gzip
import io
import struct

import pytest

from nibabel import _compression


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_isal_writer_round_trip_and_header(tmp_path, level):
    isal = pytest.importorskip('isal.igzip')
    payload = b'neuroimaging' * 1024
    outputs = []
    for name in ('one.gz', 'other.gz'):
        path = tmp_path / name
        with _compression.gzip_open(str(path), 'wb', compresslevel=level, mtime=1234) as stream:
            assert isinstance(stream, isal.IGzipFile)
            stream.write(payload)
        outputs.append(path.read_bytes())
        assert gzip.decompress(outputs[-1]) == payload
        assert struct.unpack('<I', outputs[-1][4:8])[0] == 1234
        assert not outputs[-1][3] & 8  # No filename in the gzip header.
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('level', [None, 4, 9])
def test_unsupported_level_uses_zlib(tmp_path, level):
    path = tmp_path / 'fallback.gz'
    with _compression.gzip_open(str(path), 'wb', compresslevel=level) as stream:
        assert isinstance(stream, _compression.DeterministicGzipFile)
        stream.write(b'fallback')
    assert gzip.decompress(path.read_bytes()) == b'fallback'


def test_without_isal_uses_zlib(tmp_path, monkeypatch):
    monkeypatch.setattr(_compression, 'HAVE_ISAL', False)
    path = tmp_path / 'fallback.gz'
    with _compression.gzip_open(str(path), 'wb', compresslevel=1) as stream:
        assert isinstance(stream, _compression.DeterministicGzipFile)
        stream.write(b'fallback')
    assert gzip.decompress(path.read_bytes()) == b'fallback'


def test_isal_opt_out_uses_zlib(tmp_path, monkeypatch):
    pytest.importorskip('isal.igzip')
    monkeypatch.setenv('NIBABEL_USE_ISAL', '0')
    path = tmp_path / 'fallback.gz'
    with _compression.gzip_open(str(path), 'wb', compresslevel=1) as stream:
        assert isinstance(stream, _compression.DeterministicGzipFile)
        stream.write(b'fallback')
    expected = io.BytesIO()
    with gzip.GzipFile(
        filename='', fileobj=expected, mode='wb', compresslevel=1, mtime=0
    ) as stream:
        stream.write(b'fallback')
    assert path.read_bytes() == expected.getvalue()


def test_nifti_save_round_trip_with_isal(tmp_path):
    pytest.importorskip('isal.igzip')
    import numpy as np

    import nibabel as nib

    data = np.arange(4 * 5 * 6, dtype=np.int16).reshape((4, 5, 6))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    path = tmp_path / 'image.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), path)
    image = nib.load(path)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), data)
    np.testing.assert_array_equal(image.affine, affine)

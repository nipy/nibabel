# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for NRRD reading

The small files in ``nibabel/tests/data`` are written by
``nibabel/tests/data/gen_nrrd.py``. Malformed files are built in the tests.
"""

import bz2
import gc
import gzip
import io
import re
import shutil
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

import nibabel as nib

from .. import nrrd
from ..fileholders import FileHolder
from ..imageclasses import all_image_classes
from ..nrrd import NrrdHeader, NrrdHeaderError, NrrdImage, NrrdImageError
from ..optpkg import optional_package
from ..orientations import aff2axcodes
from ..spatialimages import HeaderDataError, ImageDataError
from ..testing import data_path
from ..tripwire import TripWireError
from .test_fileslice import slicer_samples

pynrrd, have_pynrrd, _ = optional_package('nrrd')
needs_pynrrd = pytest.mark.skipif(not have_pynrrd, reason='pynrrd not installed')
# Most headers built here have no space directions. Tests that check this
# warning use pytest.warns, which sees it regardless of this filter.
pytestmark = pytest.mark.filterwarnings('ignore:NRRD file has no space directions:UserWarning')

# Voxel values and RAS affine shared by the 3D test files (see gen_nrrd.py)
DATA_3D = np.arange(60, dtype=np.int16).reshape((3, 4, 5)) - 30
AFFINE_3D = np.array(
    [
        [-1.2, 1.2, 0.0, -10.0],
        [-0.9, -1.6, 0.0, 20.0],
        [0.0, 0.0, 2.5, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
RAW_3D = DATA_3D.astype('<i2').tobytes(order='F')

# Loaded order of nrrd_list_vector.nrrd is (x, y, z, time, component)
_x, _y, _z, _t, _c = np.indices((3, 4, 5, 2, 3))
DATA_LIST_VECTOR = (_c + 3 * _x + 9 * _y + 36 * _z + 180 * _t).astype(np.float32)

EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'nrrd_raw_lps.nrrd'),
        shape=(3, 4, 5),
        dtype=np.int16,
        affine=AFFINE_3D,
        zooms=(1.5, 2.0, 2.5),
        data=DATA_3D,
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'nrrd_gzip_ras.nrrd'),
        shape=(3, 4, 5),
        dtype=np.int16,
        affine=AFFINE_3D,
        zooms=(1.5, 2.0, 2.5),
        data=DATA_3D,
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'nrrd_detached.nhdr'),
        shape=(3, 4, 5),
        dtype=np.int16,
        affine=AFFINE_3D,
        zooms=(1.5, 2.0, 2.5),
        data=DATA_3D,
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'nrrd_detached_be.nhdr'),
        shape=(3, 4, 5),
        dtype=np.int16,
        affine=AFFINE_3D,
        zooms=(1.5, 2.0, 2.5),
        data=DATA_3D,
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'nrrd_list_vector.nrrd'),
        shape=(3, 4, 5, 2, 3),
        dtype=np.float32,
        affine=AFFINE_3D,
        zooms=(1.5, 2.0, 2.5, 0.5, 1.0),
        data=DATA_LIST_VECTOR,
        is_proxy=True,
    ),
]

AFFINE_2D = np.array(
    [
        [0.5, 0.0, 0.0, 1.0],
        [0.0, 0.25, 0.0, 2.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

BASE_FIELDS = 'type: short\ndimension: 3\nsizes: 3 4 5\nendian: little\n'
LPS_FIELDS = (
    'space: left-posterior-superior\n'
    'space directions: (1.2,0.9,0) (-1.2,1.6,0) (0,0,2.5)\n'
    'space origin: (10,-20,30)\n'
)


def nrrd_bytes(fields, data=b'', version=4):
    """Return the bytes of an attached NRRD file with header `fields`"""
    return f'NRRD000{version}\n{fields}\n'.encode() + data


def write_nrrd(path, fields, data=b'', version=4):
    path.write_bytes(nrrd_bytes(fields, data, version))
    return path


def load_bytes(contents, **kwargs):
    """Load an attached NRRD file from memory"""
    file_map = {'image': FileHolder(fileobj=io.BytesIO(contents))}
    return NrrdImage.from_file_map(file_map, **kwargs)


class CountingBytesIO(io.BytesIO):
    """BytesIO that counts the bytes read from it"""

    def __init__(self, *args):
        super().__init__(*args)
        self.bytes_read = 0

    def read(self, size=-1):
        out = super().read(size)
        self.bytes_read += len(out)
        return out

    def readinto(self, buffer):
        n_read = super().readinto(buffer)
        self.bytes_read += n_read
        return n_read


def test_exceptions():
    assert issubclass(NrrdHeaderError, nrrd.NrrdError)
    assert issubclass(NrrdHeaderError, HeaderDataError)
    assert issubclass(NrrdImageError, nrrd.NrrdError)
    assert issubclass(NrrdImageError, ImageDataError)


def test_image_class_registered():
    assert NrrdImage in all_image_classes
    assert NrrdImage.valid_exts == ('.nrrd', '.nhdr')
    assert NrrdImage.rw is False
    assert NrrdImage.makeable is False


@pytest.mark.thread_unsafe
def test_missing_pynrrd(monkeypatch):
    # Hide pynrrd so the optional import fails
    monkeypatch.setitem(sys.modules, 'nrrd', None)
    with pytest.raises(TripWireError, match=r'pip install nibabel\[nrrd\]'):
        NrrdImage.from_filename(pjoin(data_path, 'nrrd_raw_lps.nrrd'))


@needs_pynrrd
@pytest.mark.parametrize('params', EXAMPLE_IMAGES, ids=lambda p: Path(p['fname']).name)
def test_examples(params):
    img = nib.load(params['fname'])
    assert isinstance(img, NrrdImage)
    assert img.shape == params['shape']
    assert img.get_data_dtype().type == params['dtype']
    assert img.dataobj.dtype.type == params['dtype']
    assert_almost_equal(img.affine, params['affine'])
    assert_almost_equal(img.header.get_zooms(), params['zooms'])
    data = np.asanyarray(img.dataobj)
    assert data.dtype.type == params['dtype']
    assert_array_equal(data, params['data'])
    assert aff2axcodes(img.affine)[2] == 'S'


@needs_pynrrd
def test_pathlike_and_str():
    for fname in ('nrrd_raw_lps.nrrd', 'nrrd_detached.nhdr'):
        path = Path(data_path) / fname
        img_path = nib.load(path)
        img_str = nib.load(str(path))
        assert_array_equal(img_path.dataobj, img_str.dataobj)
        assert_array_equal(img_path.dataobj, DATA_3D)
        assert NrrdImage.path_maybe_image(path)[0]


@needs_pynrrd
def test_header_api():
    img = nib.load(pjoin(data_path, 'nrrd_raw_lps.nrrd'))
    hdr = img.header
    assert isinstance(hdr, NrrdHeader)
    assert hdr.get_data_shape() == (3, 4, 5)
    assert hdr.get_data_dtype() == np.dtype('<i2')
    assert hdr.get_axis_order() == (0, 1, 2)
    assert hdr.get_space() == 'left-posterior-superior'
    assert hdr.get_measurement_frame() is None
    assert_almost_equal(hdr.get_best_affine(), AFFINE_3D)
    assert hdr.keyvalues == {}
    # Raw fields keep their on-disk values
    assert hdr.fields['type'] == 'short'
    assert hdr.fields['encoding'] == 'raw'
    assert_array_equal(hdr.fields['sizes'], [3, 4, 5])
    assert hdr.fields['kinds'] == ['domain', 'domain', 'domain']
    assert_almost_equal(hdr.fields['space origin'], [10, -20, 30])
    assert_almost_equal(
        hdr.fields['space directions'], [[1.2, 0.9, 0], [-1.2, 1.6, 0], [0, 0, 2.5]]
    )
    ras = nib.load(pjoin(data_path, 'nrrd_gzip_ras.nrrd')).header
    assert ras.fields['space'] == 'RAS'
    assert ras.get_space() == 'right-anterior-superior'
    be_hdr = nib.load(pjoin(data_path, 'nrrd_detached_be.nhdr')).header
    assert be_hdr.get_data_dtype() == np.dtype('>i2')


@needs_pynrrd
def test_header_copy_and_eq():
    hdr = nib.load(pjoin(data_path, 'nrrd_list_vector.nrrd')).header
    copied = hdr.copy()
    assert copied == hdr
    assert copied is not hdr
    assert_almost_equal(copied.get_best_affine(), hdr.get_best_affine())
    # Copies do not share mutable state
    copied.keyvalues['modality'] = 'changed'
    copied.fields['space directions'][1, 0] = 100
    assert hdr.keyvalues['modality'] == 'test'
    assert hdr.fields['space directions'][1, 0] == 1.2
    assert copied != hdr
    # Headers from other formats keep shape, dtype and zooms
    nifti_hdr = nib.Nifti1Header()
    nifti_hdr.set_data_shape((2, 3, 4))
    nifti_hdr.set_zooms((2, 3, 4))
    converted = NrrdHeader.from_header(nifti_hdr)
    assert converted.get_data_shape() == (2, 3, 4)
    assert converted.get_zooms() == (2, 3, 4)
    assert converted.fields == {}
    assert converted.get_axis_order() == (0, 1, 2)
    assert converted.get_space() is None
    # With no NRRD fields the affine comes from the zooms, as for other headers
    assert_array_equal(converted.get_best_affine()[:3, :3], np.diag([-2, 3, 4]))


@needs_pynrrd
def test_list_vector():
    img = nib.load(pjoin(data_path, 'nrrd_list_vector.nrrd'))
    hdr = img.header
    assert hdr.get_axis_order() == (1, 2, 3, 4, 0)
    # Fields stay in on-disk axis order
    assert hdr.fields['kinds'] == ['vector', 'domain', 'domain', 'domain', 'list']
    assert hdr.fields['labels'] == ['velocity', '', '', '', 'time']
    assert_array_equal(hdr.fields['sizes'], [3, 3, 4, 5, 2])
    assert np.isnan(hdr.fields['space directions'][0]).all()
    assert np.isnan(hdr.fields['space directions'][4]).all()
    # Measurement frame vectors are columns; returned in RAS
    assert_almost_equal(hdr.get_measurement_frame(), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert hdr.keyvalues == {
        'description': 'two\nlines and one \\ backslash',
        'modality': 'test',
    }
    assert_array_equal(img.dataobj[..., 1], DATA_LIST_VECTOR[..., 1])
    assert_array_equal(img.dataobj[1, :, 3, 1, :], DATA_LIST_VECTOR[1, :, 3, 1, :])


@needs_pynrrd
def test_2d():
    with pytest.warns(UserWarning, match='no anatomical'):
        img = nib.load(pjoin(data_path, 'nrrd_2d.nrrd'))
    assert img.shape == (4, 5)
    assert img.get_data_dtype() == np.uint8
    assert_array_equal(img.affine, AFFINE_2D)
    assert img.header.get_zooms() == (0.5, 0.25)
    assert_array_equal(img.dataobj, np.arange(20, dtype=np.uint8).reshape((4, 5)))


@needs_pynrrd
@pytest.mark.parametrize(
    ('space', 'flip'),
    [
        ('left-posterior-superior', (-1, -1, 1)),
        ('LPS', (-1, -1, 1)),
        ('Left-Posterior-Superior', (-1, -1, 1)),
        ('scanner-xyz', (-1, -1, 1)),
        ('right-anterior-superior', (1, 1, 1)),
        ('RAS', (1, 1, 1)),
        ('left-anterior-superior', (-1, 1, 1)),
        ('LAS', (-1, 1, 1)),
    ],
)
def test_spaces(tmp_path, space, flip):
    fields = (
        BASE_FIELDS + 'encoding: raw\n'
        f'space: {space}\n'
        'space directions: (1,2,3) (0,4,0) (0,0,5)\n'
        'space origin: (6,7,8)\n'
    )
    img = NrrdImage.from_filename(write_nrrd(tmp_path / 'space.nrrd', fields, RAW_3D))
    flip = np.diag(flip)
    expected = np.eye(4)
    expected[:3, :3] = flip @ np.array([[1, 2, 3], [0, 4, 0], [0, 0, 5]]).T
    expected[:3, 3] = flip @ [6, 7, 8]
    assert_array_equal(img.affine, expected)
    assert_almost_equal(img.header.get_zooms(), (np.sqrt(14), 4, 5))


@needs_pynrrd
def test_right_handed_space(tmp_path):
    fields = BASE_FIELDS + 'encoding: raw\nspace: 3D-right-handed\n'
    fields += 'space directions: (2,0,0) (0,3,0) (0,0,4)\nspace origin: (1,2,3)\n'
    with pytest.warns(UserWarning, match='no anatomical'):
        img = NrrdImage.from_filename(write_nrrd(tmp_path / 'rh.nrrd', fields, RAW_3D))
    assert_array_equal(img.affine, [[2, 0, 0, 1], [0, 3, 0, 2], [0, 0, 4, 3], [0, 0, 0, 1]])


@needs_pynrrd
@pytest.mark.parametrize(
    'space', ['3D-left-handed', 'right-anterior-superior-time', 'RAST', 'scanner-xyz-time']
)
def test_unsupported_spaces(tmp_path, space):
    fields = BASE_FIELDS + f'encoding: raw\nspace: {space}\n'
    with pytest.raises(NrrdHeaderError, match='space'):
        NrrdImage.from_filename(write_nrrd(tmp_path / 'bad.nrrd', fields, RAW_3D))


@needs_pynrrd
def test_spacings_only(tmp_path):
    fields = BASE_FIELDS + 'encoding: raw\nspacings: 2 -3 4\n'
    with pytest.warns(UserWarning, match='no anatomical'):
        img = NrrdImage.from_filename(write_nrrd(tmp_path / 'sp.nrrd', fields, RAW_3D))
    assert_array_equal(img.affine, np.diag([2, -3, 4, 1]))
    assert img.header.get_zooms() == (2, 3, 4)
    # No geometry at all gives unit voxels
    fields = BASE_FIELDS + 'encoding: raw\n'
    with pytest.warns(UserWarning, match='no anatomical'):
        img = NrrdImage.from_filename(write_nrrd(tmp_path / 'none.nrrd', fields, RAW_3D))
    assert_array_equal(img.affine, np.eye(4))
    assert img.header.get_zooms() == (1, 1, 1)


@needs_pynrrd
def test_no_squeeze(tmp_path):
    fields = 'type: uchar\ndimension: 4\nsizes: 3 1 5 1\nencoding: raw\n'
    fields += 'kinds: domain domain domain list\n'
    fields += 'space: RAS\nspace directions: (1,0,0) (0,1,0) (0,0,1) none\n'
    data = np.arange(15, dtype=np.uint8)
    img = NrrdImage.from_filename(write_nrrd(tmp_path / 'sq.nrrd', fields, data.tobytes()))
    assert img.shape == (3, 1, 5, 1)
    assert_array_equal(img.dataobj, data.reshape((3, 1, 5, 1), order='F'))


@needs_pynrrd
def test_field_spelling(tmp_path):
    # Field names and values are case insensitive; old spellings are accepted
    fields = (
        'Type: Short\nDIMENSION: 3\nSizes: 3 4 5\nEndian: LITTLE\nEncoding: RAW\n'
        'Space: LPS\nSpace Directions: (1.2,0.9,0) (-1.2,1.6,0) (0,0,2.5)\n'
        'space origin: (10,-20,30)\nbyteskip: 2\n'
    )
    path = write_nrrd(tmp_path / 'case.nrrd', fields, b'\0\0' + RAW_3D)
    img = NrrdImage.from_filename(path)
    assert_array_equal(img.dataobj, DATA_3D)
    assert_almost_equal(img.affine, AFFINE_3D)
    assert img.header.fields['byte skip'] == 2


@needs_pynrrd
@pytest.mark.parametrize(
    ('nrrd_type', 'dtype'),
    [
        ('signed char', np.int8),
        ('uint8', np.uint8),
        ('ushort', np.uint16),
        ('int', np.int32),
        ('unsigned int', np.uint32),
        ('long long', np.int64),
        ('uint64_t', np.uint64),
        ('float', np.float32),
        ('double', np.float64),
    ],
)
@pytest.mark.parametrize('endian', ['little', 'big'])
def test_types(tmp_path, nrrd_type, dtype, endian):
    disk_dtype = np.dtype(dtype).newbyteorder('<' if endian == 'little' else '>')
    data = np.arange(24).reshape((2, 3, 4)).astype(disk_dtype)
    fields = f'type: {nrrd_type}\ndimension: 3\nsizes: 2 3 4\nendian: {endian}\nencoding: raw\n'
    fields += 'space: RAS\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n'
    path = write_nrrd(tmp_path / 'type.nrrd', fields, data.tobytes(order='F'))
    img = NrrdImage.from_filename(path)
    assert img.get_data_dtype() == disk_dtype
    assert_array_equal(img.dataobj, data)


def _encode(encoding, raw):
    if encoding in ('gzip', 'gz'):
        return gzip.compress(raw, mtime=0)
    if encoding in ('bzip2', 'bz2'):
        return bz2.compress(raw)
    return raw


@needs_pynrrd
@pytest.mark.parametrize('encoding', ['raw', 'gzip', 'gz', 'bzip2', 'bz2'])
def test_encodings(tmp_path, encoding):
    fields = BASE_FIELDS + f'encoding: {encoding}\n' + LPS_FIELDS
    data = _encode(encoding, RAW_3D)
    path = write_nrrd(tmp_path / 'enc.nrrd', fields, data)
    for img in (NrrdImage.from_filename(path), load_bytes(path.read_bytes())):
        assert_array_equal(img.dataobj, DATA_3D)
        assert_almost_equal(img.affine, AFFINE_3D)
    # Detached, with a data file name that does not hint at the encoding
    (tmp_path / 'data.bin').write_bytes(data)
    fields += 'data file: data.bin\n'
    img = NrrdImage.from_filename(write_nrrd(tmp_path / 'enc.nhdr', fields))
    assert_array_equal(img.dataobj, DATA_3D)


@needs_pynrrd
@pytest.mark.parametrize('encoding', ['raw', 'gzip', 'bzip2'])
def test_line_and_byte_skip(tmp_path, encoding):
    # Line skip counts lines of the file as stored; byte skip counts bytes after
    # decompression
    payload = _encode(encoding, b'\x01\x02\x03' + RAW_3D)
    fields = BASE_FIELDS + f'encoding: {encoding}\nline skip: 2\nbyte skip: 3\n'
    contents = nrrd_bytes(fields, b'first line\nsecond line\n' + payload)
    assert_array_equal(load_bytes(contents).dataobj, DATA_3D)
    path = tmp_path / 'skip.nrrd'
    path.write_bytes(contents)
    assert_array_equal(NrrdImage.from_filename(path).dataobj, DATA_3D)


@needs_pynrrd
def test_byte_skip_minus_one(tmp_path):
    # Raw data are the last bytes of the file, whatever comes before them
    fields = BASE_FIELDS + 'encoding: raw\nbyte skip: -1\n'
    contents = nrrd_bytes(fields, b'padding bytes\n' + RAW_3D)
    assert_array_equal(load_bytes(contents).dataobj, DATA_3D)
    (tmp_path / 'data.raw').write_bytes(b'junk' + RAW_3D)
    fields += 'data file: data.raw\n'
    img = NrrdImage.from_filename(write_nrrd(tmp_path / 'skip.nhdr', fields))
    assert_array_equal(img.dataobj, DATA_3D)


@needs_pynrrd
def test_detached_names(tmp_path):
    # Data files are found relative to the header, not the working directory
    sub = tmp_path / 'sub'
    sub.mkdir()
    shutil.copy(pjoin(data_path, 'nrrd_detached.raw'), sub)
    hdr_text = Path(data_path, 'nrrd_detached.nhdr').read_bytes()
    for name in ('foo.nrrd.nhdr', 'foo.nhdr', 'foo.NHDR'):
        (sub / name).write_bytes(hdr_text)
        img = nib.load(sub / name)
        assert isinstance(img, NrrdImage)
        assert_array_equal(img.dataobj, DATA_3D)
    # Absolute data file path
    fields = BASE_FIELDS + f'encoding: raw\ndata file: {tmp_path / "abs.raw"}\n'
    (tmp_path / 'abs.raw').write_bytes(RAW_3D)
    img = NrrdImage.from_filename(write_nrrd(sub / 'abs.nhdr', fields))
    assert_array_equal(img.dataobj, DATA_3D)
    # A relative data file cannot be found from a header without a file name
    with pytest.raises(NrrdImageError, match='relative'):
        load_bytes(hdr_text)
    # Missing data file
    (sub / 'missing.nhdr').write_bytes(hdr_text.replace(b'nrrd_detached.raw', b'nope.raw'))
    with pytest.raises(FileNotFoundError):
        NrrdImage.from_filename(sub / 'missing.nhdr')


@needs_pynrrd
def test_path_maybe_image(tmp_path):
    for fname in ('nrrd_raw_lps.nrrd', 'nrrd_detached.nhdr', 'nrrd_gzip_ras.nrrd'):
        assert NrrdImage.path_maybe_image(pjoin(data_path, fname))[0]
    bad = tmp_path / 'bad.nrrd'
    bad.write_bytes(b'NOTNRRD\n' + RAW_3D)
    assert not NrrdImage.path_maybe_image(bad)[0]
    with pytest.raises(nib.filebasedimages.ImageFileError):
        nib.load(bad)
    assert not NrrdImage.path_maybe_image(tmp_path / 'image.nii')[0]


BAD_HEADERS = [
    ('NRRX0004\ntype: short\n\n', 'NRRD magic'),
    ('NRRD0006\n' + BASE_FIELDS + 'encoding: raw\n\n', 'version'),
    ('NRRD\n' + BASE_FIELDS + 'encoding: raw\n\n', 'NRRD magic'),
    ('', 'NRRD magic'),
    ('NRRD0004\n' + BASE_FIELDS + '\n', "'encoding'"),
    ('NRRD0004\ntype: short\ndimension: 3\nencoding: raw\nendian: little\n\n', "'sizes'"),
    ('NRRD0004\ndimension: 3\nsizes: 3 4 5\nendian: little\nencoding: raw\n\n', "'type'"),
    ('NRRD0004\ntype: short\nsizes: 3 4 5\nendian: little\nencoding: raw\n\n', "'dimension'"),
    ('NRRD0004\ntype: short\ndimension: 3\nsizes: 3 4 5\nencoding: raw\n\n', "'endian'"),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nendian: middle\n\n', 'more than once'),
    ('NRRD0004\n' + BASE_FIELDS.replace('little', 'middle') + 'encoding: raw\n\n', 'endian'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nthis line has no colon\n\n', 'colon'),
    ('NRRD0004\n' + BASE_FIELDS.replace('3 4 5', '3 4') + 'encoding: raw\n\n', 'sizes'),
    ('NRRD0004\n' + BASE_FIELDS.replace('3 4 5', '3 0 5') + 'encoding: raw\n\n', 'sizes'),
    ('NRRD0004\n' + BASE_FIELDS.replace('3 4 5', '3 x 5') + 'encoding: raw\n\n', 'sizes'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: zip\n\n', 'encoding'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: ascii\n\n', 'not supported'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: hex\n\n', 'not supported'),
    ('NRRD0004\n' + BASE_FIELDS.replace('short', 'quaternion') + 'encoding: raw\n\n', 'type'),
    (
        'NRRD0004\n' + BASE_FIELDS.replace('short', 'block') + 'block size: 2\nencoding: raw\n\n',
        'not supported',
    ),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\nspace dimension: 3\n\n', 'both'),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\n\n',
        'space directions',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
        'space directions: (1,0,0) (0,1,0)\n\n',
        'space directions',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\nspace origin: (1,2)\n\n',
        'space origin',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\nmeasurement frame: (1,0) (0,1)\n\n',
        'measurement frame',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
        'space directions: (1,0,0) (0,1,0) none\n\n',
        'space dimension',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\nkinds: vector domain domain\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\n\n',
        'kind',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nkinds: space space space\n'
        'space dimension: 2\nspace directions: (1,0) (0,1) none\n\n',
        'kind',
    ),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nkinds: domain domain\n\n', 'kinds'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nline skip: -1\n\n', 'line skip'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nbyte skip: -2\n\n', 'byte skip'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: gzip\nbyte skip: -1\n\n', 'byte skip'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\ndata file: LIST\na.raw\nb.raw\n', 'data file'),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\ndata file: slice%03d.raw 1 5 1 2\n',
        'data file',
    ),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\ncontent: na\xefve\n\n', 'ASCII'),
    (
        'NRRD0004\ntype: short\ndimension: 0\nsizes: 1\nendian: little\nencoding: raw\n\n',
        'dimension',
    ),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: martian\n\n', 'Unknown NRRD space'),
    ('NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace dimension: 4\n\n', 'space dimension 4'),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
        'space directions: (1,0) (0,1) (1,1)\n\n',
        'length 2',
    ),
    (
        'NRRD0004\n' + BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
        'space directions: (inf,0,0) (0,1,0) (0,0,1)\n\n',
        'not finite',
    ),
]


@needs_pynrrd
@pytest.mark.parametrize(('contents', 'match'), BAD_HEADERS)
def test_bad_headers(tmp_path, contents, match):
    path = tmp_path / 'bad.nrrd'
    path.write_bytes(contents.encode('latin-1') + RAW_3D)
    with pytest.raises(NrrdHeaderError, match=match):
        NrrdImage.from_filename(path)
    with pytest.raises(NrrdHeaderError, match=match):
        load_bytes(path.read_bytes())


@needs_pynrrd
def test_short_data(tmp_path):
    fields = BASE_FIELDS + 'encoding: raw\n'
    path = write_nrrd(tmp_path / 'short.nrrd', fields, RAW_3D[:-1])
    with pytest.raises(NrrdImageError, match='120 bytes'):
        NrrdImage.from_filename(path)
    path = write_nrrd(tmp_path / 'short.nrrd', fields + 'byte skip: -1\n', RAW_3D[:-1])
    with pytest.raises(NrrdImageError, match='120 bytes'):
        NrrdImage.from_filename(path)
    # Truncated compressed data are only found when the data are read
    for encoding in ('gzip', 'bzip2'):
        fields = BASE_FIELDS + f'encoding: {encoding}\n'
        data = _encode(encoding, RAW_3D)[:-12]
        img = NrrdImage.from_filename(write_nrrd(tmp_path / 'short.nrrd', fields, data))
        with pytest.raises(NrrdImageError, match=encoding):
            img.get_fdata()


@needs_pynrrd
def test_line_skip_past_end():
    fields = BASE_FIELDS + 'encoding: raw\nline skip: 3\n'
    with pytest.raises(NrrdImageError, match='3 lines'):
        load_bytes(nrrd_bytes(fields, b'one line\n'))


@needs_pynrrd
def test_components_without_directions():
    # Component axes move last even when there are no space directions
    fields = BASE_FIELDS + 'encoding: raw\nkinds: RGB-color domain domain\n'
    img = load_bytes(nrrd_bytes(fields, RAW_3D))
    assert img.header.get_axis_order() == (1, 2, 0)
    assert img.shape == (4, 5, 3)
    assert_array_equal(img.dataobj, DATA_3D.transpose(1, 2, 0))
    # A space with no space directions gives unit voxels
    fields = BASE_FIELDS + 'encoding: raw\nspace: RAS\n'
    with pytest.warns(UserWarning, match='no anatomical'):
        img = load_bytes(nrrd_bytes(fields, RAW_3D))
    assert_array_equal(img.affine, np.eye(4))


def test_direction_array():
    # pynrrd can also give space directions as a list with None for each none axis
    directions = nrrd._direction_array([None, [1, 0, 0], (0, 2, 0)])
    assert directions.dtype == np.float64
    assert np.isnan(directions[0]).all()
    assert_array_equal(directions[1:], [[1, 0, 0], [0, 2, 0]])
    assert nrrd._direction_array([None]).shape == (1, 0)
    with pytest.raises(NrrdHeaderError, match='different lengths'):
        nrrd._direction_array([[1, 0, 0], [0, 1]])


@needs_pynrrd
def test_header_eq_other_types():
    hdr = nib.load(pjoin(data_path, 'nrrd_raw_lps.nrrd')).header
    assert hdr != nib.Nifti1Header()
    other = hdr.copy()
    other.fields['content'] = 'extra'
    assert other != hdr
    other = hdr.copy()
    other.keyvalues['extra'] = 'value'
    assert other != hdr


def test_data_window():
    base = io.BytesIO(b'headerDATA')
    window = nrrd._DataWindow(base, 6)
    assert window.readable()
    assert window.seekable()
    assert window.read(2) == b'DA'
    assert window.seek(1, io.SEEK_CUR) == 3
    assert window.read() == b'A'
    assert window.seek(-2, io.SEEK_END) == 2
    assert window.tell() == 2
    assert window.read() == b'TA'
    with pytest.raises(ValueError, match='whence'):
        window.seek(0, 3)


def test_proxy_arguments():
    spec = ((3, 4, 5), np.dtype('<i2'), 0)
    fname = pjoin(data_path, 'nrrd_raw_lps.nrrd')
    with pytest.raises(ValueError, match='compression'):
        nrrd.NrrdArrayProxy(fname, spec, compression='zstd')
    with pytest.raises(ValueError, match='permutation'):
        nrrd.NrrdArrayProxy(fname, spec, axis_order=(0, 1, 1))


@needs_pynrrd
def test_proxy_file_object():
    # Copies of a proxy for an open file share its lock
    prox = load_bytes(Path(data_path, 'nrrd_list_vector.nrrd').read_bytes()).dataobj
    copied = prox.copy()
    assert copied._lock is prox._lock
    assert_array_equal(copied, DATA_LIST_VECTOR)
    # New axes work with reordered axes
    assert_array_equal(prox[:, None, ..., 1], DATA_LIST_VECTOR[:, None, ..., 1])


@needs_pynrrd
def test_raw_file_errors_unchanged(tmp_path):
    # Errors reading raw data are not wrapped, unlike errors in compressed data
    path = shutil.copy(pjoin(data_path, 'nrrd_raw_lps.nrrd'), tmp_path)
    img = NrrdImage.from_filename(path, mmap=False)
    Path(path).unlink()
    with pytest.raises(FileNotFoundError):
        img.get_fdata()


@needs_pynrrd
def test_file_map_and_sniff(tmp_path):
    with pytest.raises(nib.filebasedimages.ImageFileError, match='extension'):
        NrrdImage.filespec_to_file_map(tmp_path / 'image.nii')
    fname = pjoin(data_path, 'nrrd_raw_lps.nrrd')
    found, sniff = NrrdImage.path_maybe_image(fname)
    assert found
    # A sniff of the same file is reused
    assert NrrdImage.path_maybe_image(fname, sniff) == (True, sniff)
    assert NrrdImage.path_maybe_image(tmp_path / 'missing.nrrd') == (False, None)
    file_map = NrrdImage.filespec_to_file_map(fname)
    with pytest.raises(ValueError, match='mmap'):
        NrrdImage.from_file_map(file_map, mmap='r+')


@needs_pynrrd
def test_mmap():
    raw = pjoin(data_path, 'nrrd_raw_lps.nrrd')
    assert isinstance(NrrdImage.from_filename(raw, mmap=True).dataobj.get_unscaled(), np.memmap)
    assert isinstance(NrrdImage.from_filename(raw, mmap='r').dataobj.get_unscaled(), np.memmap)
    assert not isinstance(
        NrrdImage.from_filename(raw, mmap=False).dataobj.get_unscaled(), np.memmap
    )
    detached = pjoin(data_path, 'nrrd_detached.nhdr')
    img = NrrdImage.from_filename(detached, mmap=True)
    assert isinstance(img.dataobj.get_unscaled(), np.memmap)
    assert_array_equal(img.dataobj, DATA_3D)
    gz = pjoin(data_path, 'nrrd_gzip_ras.nrrd')
    assert not isinstance(NrrdImage.from_filename(gz, mmap=True).dataobj.get_unscaled(), np.memmap)
    for bad in ('r+', 'invalid'):
        with pytest.raises(ValueError):
            NrrdImage.from_filename(raw, mmap=bad)


@needs_pynrrd
@pytest.mark.parametrize('params', EXAMPLE_IMAGES, ids=lambda p: Path(p['fname']).name)
@pytest.mark.parametrize('keep_file_open', [False, True])
def test_proxy_slicing(params, keep_file_open):
    img = NrrdImage.from_filename(params['fname'], keep_file_open=keep_file_open)
    prox = img.dataobj
    assert prox.is_proxy
    assert prox.shape == params['shape']
    assert prox.ndim == len(params['shape'])
    arr = params['data']
    samples = list(slicer_samples(img.shape))
    # 5D shapes give over 100000 samples, so check an evenly spread subset
    if len(samples) > 3000:
        samples = samples[:: len(samples) // 3000 + 1]
    for sliceobj in samples:
        assert_array_equal(prox[sliceobj], arr[sliceobj])
    with pytest.raises(ValueError):
        prox[[0, 1]]


@needs_pynrrd
def test_proxy_copy_and_reshape():
    img = nib.load(pjoin(data_path, 'nrrd_list_vector.nrrd'))
    prox = img.dataobj
    copied = prox.copy()
    assert type(copied) is type(prox)
    assert copied.shape == prox.shape
    assert_array_equal(copied, DATA_LIST_VECTOR)
    # Like ArrayProxy, reshape uses Fortran order
    assert_array_equal(prox.reshape((12, 5, 6)), DATA_LIST_VECTOR.reshape((12, 5, 6), order='F'))
    raw = nib.load(pjoin(data_path, 'nrrd_raw_lps.nrrd')).dataobj
    reshaped = raw.reshape((12, 5))
    assert type(reshaped) is type(raw)
    assert_array_equal(reshaped, DATA_3D.reshape((12, 5), order='F'))
    assert_array_equal(raw.copy(), DATA_3D)


@needs_pynrrd
def test_lazy_gzip():
    # Random voxels do not compress, so the compressed size is close to the raw size
    rng = np.random.default_rng(12345)
    data = rng.integers(-(2**15), 2**15, size=(256, 256, 16), dtype=np.int16)
    fields = 'type: short\ndimension: 3\nsizes: 256 256 16\nendian: little\nencoding: gzip\n'
    compressed = gzip.compress(data.tobytes(order='F'), mtime=0)
    fobj = CountingBytesIO(nrrd_bytes(fields, compressed))
    img = NrrdImage.from_file_map({'image': FileHolder(fileobj=fobj)})
    fobj.bytes_read = 0
    assert_array_equal(img.dataobj[:, :, 0], data[:, :, 0])
    # The first slice is the first 128 kB of 2 MB
    assert fobj.bytes_read < len(compressed) // 4
    fobj.bytes_read = 0
    assert_array_equal(img.dataobj, data)
    assert fobj.bytes_read >= len(compressed)


@needs_pynrrd
def test_threaded_reads():
    img = NrrdImage.from_filename(pjoin(data_path, 'nrrd_list_vector.nrrd'), keep_file_open=True)
    slicers = [(i, slice(None), j) for i in range(3) for j in range(5)] * 4

    def read(sliceobj):
        return img.dataobj[sliceobj]

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(read, slicers))
    for sliceobj, result in zip(slicers, results, strict=True):
        assert_array_equal(result, DATA_LIST_VECTOR[sliceobj])


@needs_pynrrd
@pytest.mark.filterwarnings('error::ResourceWarning')
@pytest.mark.filterwarnings('error::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('params', EXAMPLE_IMAGES, ids=lambda p: Path(p['fname']).name)
@pytest.mark.parametrize('keep_file_open', [False, True])
def test_files_closed(params, keep_file_open):
    # Collect garbage left by earlier tests so only files opened here count
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gc.collect()
    img = NrrdImage.from_filename(params['fname'], keep_file_open=keep_file_open)
    img.get_fdata()
    img.dataobj[0]
    del img
    gc.collect()


@needs_pynrrd
def test_pynrrd_globals_untouched():
    space_type = getattr(pynrrd, 'SPACE_DIRECTIONS_TYPE', None)
    allow_duplicate = pynrrd.reader.ALLOW_DUPLICATE_FIELD
    for params in EXAMPLE_IMAGES:
        nib.load(params['fname']).get_fdata()
    assert getattr(pynrrd, 'SPACE_DIRECTIONS_TYPE', None) == space_type
    assert pynrrd.reader.ALLOW_DUPLICATE_FIELD == allow_duplicate


@needs_pynrrd
def test_read_only(tmp_path):
    img = nib.load(pjoin(data_path, 'nrrd_raw_lps.nrrd'))
    with pytest.raises(NotImplementedError):
        img.to_filename(tmp_path / 'out.nrrd')


@needs_pynrrd
@pytest.mark.parametrize('params', EXAMPLE_IMAGES, ids=lambda p: Path(p['fname']).name)
def test_nifti_round_trip(tmp_path, params):
    img = nib.load(params['fname'])
    nifti = nib.Nifti1Image.from_image(img)
    out = tmp_path / 'out.nii'
    nib.save(nifti, out)
    back = nib.load(out)
    assert_almost_equal(back.affine, params['affine'], decimal=6)
    assert_array_equal(back.dataobj, params['data'])


@needs_pynrrd
def test_nib_ls(capsys):
    from ..cmdline.ls import main as nib_ls

    fname = pjoin(data_path, 'nrrd_list_vector.nrrd')
    nib_ls([fname])
    out = capsys.readouterr().out
    assert out.startswith(fname)
    assert 'float32' in out
    assert re.search(r'\[\s*3,\s+4,\s+5,\s+2,\s+3\]', out)

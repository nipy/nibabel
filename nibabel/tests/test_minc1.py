# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import bz2
import gzip
from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import Nifti1Image, load, minc1
from .._compression import HAVE_ZSTD, zstd
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1File, Minc1Image, MincHeader
from ..testing import assert_data_similar, data_path
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from .test_fileslice import slicer_samples

EG_FNAME = pjoin(data_path, 'tiny.mnc')

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'tiny.mnc'),
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
        # These values from SPM2
        data_summary=dict(min=0.20784314, max=0.74901961, mean=0.60602819),
        is_proxy=True,
    ),
    dict(
        fname=pjoin(data_path, 'minc1_1_scale.mnc'),
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
        fname=pjoin(data_path, 'minc1_4d.mnc'),
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
        fname=pjoin(data_path, 'minc1-no-att.mnc'),
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
]


def _write_vector_minc1(path, data, irregular=False):
    with netcdf_file(path, 'w') as mnc:
        names = ['zspace', 'yspace', 'xspace']
        for name, length, step, start, direction in (
            ('zspace', 2, 4.0, -78.0, [0.0, 0.0, 1.0]),
            ('yspace', 3, -3.0, -132.0, [0.0, 1.0, 0.0]),
            ('xspace', 4, 2.0, -96.0, [1.0, 0.0, 0.0]),
        ):
            mnc.createDimension(name, length)
            irregular_dim = name == 'zspace' and irregular
            dim = mnc.createVariable(name, 'd', (name,) if irregular_dim else ())
            dim.data[...] = [-78, -73.25] if irregular_dim else 0
            dim.spacing = b'irregular' if irregular_dim else b'regular__'
            dim.step = step
            dim.start = start
            dim.direction_cosines = np.array(direction)
        if data.ndim == 4:
            mnc.createDimension('vector_dimension', data.shape[-1])
            names.append('vector_dimension')
        image = mnc.createVariable('image', data.dtype.char.lower(), names)
        image[:] = data
        image.signtype = b'unsigned' if data.dtype.kind == 'u' else b'signed__'
        image.valid_range = np.array(
            [0, 255] if data.dtype.kind == 'u' else [-100, 100], dtype=np.int16
        )
        for name, values in [('image-min', [-4.0, -7.0]), ('image-max', [16.0, 33.0])]:
            mnc.createVariable(name, 'd', ('zspace',))[:] = values


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('dtype', [np.float64, np.int16, np.uint8])
def test_vector_dimension(tmp_path, vector, dtype):
    data = np.arange(72, dtype=dtype).reshape(2, 3, 4, 3)
    data = data * 3 if dtype == np.uint8 else data - 35
    if not vector:
        data = data[..., 0]
    path = tmp_path / 'vector.mnc'
    _write_vector_minc1(path, data)
    image = load(path)
    expected = data.astype(np.float64)
    if dtype == np.int16:
        expected[0] = data[0] / 10 + 6
        expected[1] = data[1] / 5 + 13
    elif dtype == np.uint8:
        expected[0] = data[0] / 255 * 20 - 4
        expected[1] = data[1] / 255 * 40 - 7
    assert isinstance(image, Minc1Image)
    assert image.shape == data.shape
    assert image.header.get_zooms() == ((4, 3, 2, 1) if vector else (4, 3, 2))
    assert_array_equal(
        image.affine,
        [[0, 0, 2, -96], [0, -3, 0, -132], [4, 0, 0, -78], [0, 0, 0, 1]],
    )
    assert_allclose(np.asarray(image.dataobj), expected)
    assert_allclose(image.get_fdata(), expected)
    for slicer in ((slice(None), 1), (1, slice(None)), (Ellipsis, 1), (None, 1, Ellipsis)):
        assert_allclose(image.dataobj[slicer], expected[slicer])


@pytest.mark.parametrize('vector', [False, True])
def test_vector_dimension_irregular_spatial_spacing(tmp_path, vector):
    path = tmp_path / 'irregular-spatial.mnc'
    shape = (2, 3, 4, 3) if vector else (2, 3, 4)
    _write_vector_minc1(path, np.zeros(shape), irregular=True)
    with pytest.raises(ValueError, match='Irregular spacing not supported'):
        load(path)


class _TestMincFile:
    module = minc1
    file_class = Minc1File
    fname = EG_FNAME
    opener = netcdf_file
    test_files = EXAMPLE_IMAGES

    def test_mincfile(self):
        for tp in self.test_files:
            mnc_obj = self.opener(tp['fname'], 'r')
            mnc = self.file_class(mnc_obj)
            assert mnc.get_data_dtype().type == tp['dtype']
            assert mnc.get_data_shape() == tp['shape']
            assert mnc.get_zooms() == tp['zooms']
            assert_array_equal(mnc.get_affine(), tp['affine'])
            data = mnc.get_scaled_data()
            assert data.shape == tp['shape']
            # Can't close mmapped NetCDF with live mmap arrays
            del mnc, data

    def test_mincfile_slicing(self):
        # Test slicing and scaling of mincfile data
        for tp in self.test_files:
            mnc_obj = self.opener(tp['fname'], 'r')
            mnc = self.file_class(mnc_obj)
            data = mnc.get_scaled_data()
            for slicedef in (
                (slice(None),),
                (1,),
                (slice(None), 1),
                (1, slice(None)),
                (slice(None), 1, 1),
                (1, slice(None), 1),
                (1, 1, slice(None)),
            ):
                sliced_data = mnc.get_scaled_data(slicedef)
                assert_array_equal(sliced_data, data[slicedef])
            # Can't close mmapped NetCDF with live mmap arrays
            del mnc, data

    def test_load(self):
        # Check highest level load of minc works
        for tp in self.test_files:
            img = load(tp['fname'])
            data = img.get_fdata()
            assert data.shape == tp['shape']
            # min, max, mean values from read in SPM2 / minctools
            assert_data_similar(data, tp)
            # check if mnc can be converted to nifti
            ni_img = Nifti1Image.from_image(img)
            assert_array_equal(ni_img.affine, tp['affine'])
            assert_array_equal(ni_img.get_fdata(), data)

    def test_array_proxy_slicing(self):
        # Test slicing of array proxy
        for tp in self.test_files:
            img = load(tp['fname'])
            arr = img.get_fdata()
            prox = img.dataobj
            assert prox.is_proxy
            for sliceobj in slicer_samples(img.shape):
                assert_array_equal(arr[sliceobj], prox[sliceobj])


class TestMinc1File(_TestMincFile):
    @pytest.mark.thread_unsafe
    def test_compressed(self):
        # we can read minc compressed
        # Not so for MINC2; hence this small sub-class
        for tp in self.test_files:
            content = open(tp['fname'], 'rb').read()
            openers_exts = [(gzip.open, '.gz'), (bz2.BZ2File, '.bz2')]
            if HAVE_ZSTD:  # add .zst to test if installed
                openers_exts += [(zstd.ZstdFile, '.zst')]
            with InTemporaryDirectory():
                for opener, ext in openers_exts:
                    fname = f'test.mnc{ext}'
                    fobj = opener(fname, 'wb')
                    fobj.write(content)
                    fobj.close()
                    img = self.module.load(fname)
                    data = img.get_fdata()
                    assert_data_similar(data, tp)
                    del img


# Test the Minc header
def test_header_data_io():
    bio = BytesIO()
    hdr = MincHeader()
    arr = np.arange(24).reshape((2, 3, 4))
    with pytest.raises(NotImplementedError):
        hdr.data_to_fileobj(arr, bio)
    with pytest.raises(NotImplementedError):
        hdr.data_from_fileobj(bio)


class TestMinc1Image(tsi.TestSpatialImage):
    image_class = Minc1Image
    eg_images = (pjoin(data_path, 'tiny.mnc'),)
    module = minc1

    def test_data_to_from_fileobj(self):
        # Check data_from_fileobj of header raises an error
        for fpath in self.eg_images:
            img = self.module.load(fpath)
            bio = BytesIO()
            arr = np.arange(24).reshape((2, 3, 4))
            with pytest.raises(NotImplementedError):
                img.header.data_to_fileobj(arr, bio)
            with pytest.raises(NotImplementedError):
                img.header.data_from_fileobj(bio)

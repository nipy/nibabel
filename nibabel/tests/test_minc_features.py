# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests of MINC format features, read and written in both formats"""

from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import load, save
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1Image
from ..minc1_write import write_minc1
from ..minc2 import Minc2Image
from ..minc2_write import MincWriteError, write_minc2
from ..optpkg import optional_package

h5py, have_h5py, _ = optional_package('h5py')
sitk, have_sitk, _ = optional_package('SimpleITK')

needs_h5py = pytest.mark.skipif(not have_h5py, reason='Need h5py')

WRITERS = [
    pytest.param(write_minc1, Minc1Image, id='MINC1'),
    pytest.param(write_minc2, Minc2Image, id='MINC2', marks=needs_h5py),
]


def _netcdf(fname):
    with open(fname, 'rb') as fobj:
        return netcdf_file(BytesIO(fobj.read()), 'r', mmap=False)


@pytest.mark.parametrize(('writer', 'klass'), WRITERS)
@pytest.mark.parametrize('shape', [(4, 5, 6, 3), (2, 4, 5, 6, 3)])
def test_vector_dimension(tmp_path, writer, klass, shape):
    # RGB data: the vector dimension is the last, fastest varying axis
    data = (np.arange(np.prod(shape)).reshape(shape) % 256).astype(np.uint8)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    fname = tmp_path / 'img.mnc'
    writer(data, affine, fname, vector_dimension=True)
    img = load(fname)
    assert isinstance(img, klass)
    assert img.shape == shape
    assert img.minc_meta.dimorder[-1] == 'vector_dimension'
    assert img.minc_meta.scaling_dims == img.minc_meta.dimorder[: len(shape) - 3]
    assert_array_equal(img.get_fdata(), data)
    assert_array_equal(img.affine, affine)
    assert img.header.get_zooms()[-1] == 1
    # Saving keeps the vector dimension
    out = tmp_path / 'out.mnc'
    save(img, out)
    rt_img = load(out)
    assert rt_img.minc_meta.dimorder == img.minc_meta.dimorder
    assert_array_equal(rt_img.get_fdata(), data)
    # The vector dimension comes after three spatial axes
    with pytest.raises(MincWriteError):
        writer(np.zeros((4, 5, 3)), affine, fname, vector_dimension=True)


def test_vector_dimension_minc1_layout(tmp_path):
    # libminc makes no variable for the vector dimension in MINC1
    fname = tmp_path / 'img.mnc'
    write_minc1(np.zeros((4, 5, 6, 3), np.uint8), np.eye(4), fname, vector_dimension=True)
    nc = _netcdf(fname)
    assert nc.variables['image'].dimensions[-1] == 'vector_dimension'
    assert 'vector_dimension' not in nc.variables
    assert nc.variables['image-max'].dimensions == ('xspace',)
    del nc


@needs_h5py
def test_vector_dimension_minc2_layout(tmp_path):
    # libminc's MINC2 functions make a double dataset with only a length
    fname = tmp_path / 'img.mnc'
    write_minc2(np.zeros((4, 5, 6, 3), np.uint8), np.eye(4), fname, vector_dimension=True)
    with h5py.File(fname, 'r') as h5f:
        vector = h5f['minc-2.0/dimensions/vector_dimension']
        assert vector.dtype == np.float64
        assert dict(vector.attrs) == {'length': 3}
        image = h5f['minc-2.0/image/0/image']
        assert image.attrs['dimorder'] == b'xspace,yspace,zspace,vector_dimension'


@pytest.mark.skipif(not have_h5py or not have_sitk, reason='Need h5py and SimpleITK')
def test_vector_dimension_sitk(tmp_path):
    # ITK reads the vector dimension as pixel components
    data = (np.arange(360).reshape((4, 5, 6, 3)) % 256).astype(np.uint8)
    fname = tmp_path / 'img.mnc'
    write_minc2(data, np.diag([2.0, 3.0, 4.0, 1.0]), fname, vector_dimension=True)
    image = sitk.ReadImage(str(fname))
    assert image.GetNumberOfComponentsPerPixel() == 3
    assert image.GetSpacing() == (2.0, 3.0, 4.0)
    assert_array_equal(sitk.GetArrayFromImage(image), data.transpose((2, 1, 0, 3)))

# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for the snapshot of MINC metadata"""

import dataclasses
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import Nifti1Image, load
from ..minc1 import Minc1Image
from ..minc2 import Minc2Image
from ..minc_meta import MincMeta, normalize_attr
from ..optpkg import optional_package
from ..testing import data_path

h5py, have_h5py, _ = optional_package('h5py')


def test_normalize_attr():
    assert normalize_attr(b'regular__\x00') == 'regular__'
    assert normalize_attr(np.bytes_(b'mm')) == 'mm'
    assert normalize_attr(np.array([b'a\n', b'b'])) == 'a\nb'
    value = normalize_attr(np.array([2.0], dtype='>f8'))
    assert value.shape == ()
    assert value.dtype == np.float64
    assert not value.flags.writeable
    assert normalize_attr(np.array([1, 0, 0])).shape == (3,)


def test_minc1_snapshot():
    img = load(pjoin(data_path, 'minc1_4d.mnc'))
    meta = img.minc_meta
    assert isinstance(meta, MincMeta)
    assert meta.format == 'MINC1'
    assert meta.dimorder == ('time', 'zspace', 'yspace', 'xspace')
    assert dict(meta.dim_lengths) == {'time': 2, 'zspace': 10, 'yspace': 20, 'xspace': 20}
    assert meta.image_dtype == np.uint8
    assert meta.scaling_dims == ('time', 'zspace')
    assert set(meta.dimensions) == {'time', 'zspace', 'yspace', 'xspace'}
    assert set(meta.info) == {'study'}
    assert meta.info['study'].attrs['modality'] == 'MRI__'
    assert meta.dimensions['time'].attrs['alignment'] == 'start_'
    assert_array_equal(meta.dimensions['xspace'].attrs['direction_cosines'], [1, 0, 0])
    assert meta.history.endswith('>>> mincconvert -clobber minc2_4d.mnc minc1_4d.mnc\n')
    assert meta.labels is None
    # Images made in memory have no snapshot
    assert Minc1Image(np.zeros((2, 3, 4)), np.eye(4)).minc_meta is None


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
def test_minc2_snapshot():
    meta = load(pjoin(data_path, 'minc2-4d-d.mnc')).minc_meta
    assert meta.format == 'MINC2'
    assert meta.dimorder == ('time', 'xspace', 'yspace', 'zspace')
    assert meta.image_dtype == np.float64
    assert meta.global_attrs['class'] == 'real___'
    assert meta.history == ''
    time = meta.variables['time']
    assert time.group == 'dimensions'
    assert time.dimensions == ('time',)
    assert_array_equal(time.data, [0, 100, 200, 300, 400])
    assert time.attrs['class'] == 'time___'
    assert time.attrs['width'] == 1
    width = meta.variables['time-width']
    assert width.group == 'dimensions'
    assert_array_equal(width.data, [100] * 5)
    meta = load(pjoin(data_path, 'minc2_1_scale.mnc')).minc_meta
    assert set(meta.info) == {'acquisition', 'patient', 'study'}
    assert meta.info['patient'].attrs['full_name'] == 'mnc2nii tiny.mnc tiny.nii'
    assert meta.scaling_dims == ()
    assert Minc2Image(np.zeros((2, 3, 4)), np.eye(4)).minc_meta is None


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('fname', ['minc1_1_scale.mnc', 'minc2_1_scale.mnc'])
def test_snapshot_read_only(fname):
    meta = load(pjoin(data_path, fname)).minc_meta
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.format = 'other'
    with pytest.raises(TypeError):
        meta.global_attrs['history'] = 'new'
    with pytest.raises(TypeError):
        meta.variables['xspace'].attrs['units'] = 'cm'
    with pytest.raises(ValueError):
        meta.variables['xspace'].attrs['direction_cosines'][0] = 2
    with pytest.raises(TypeError):
        del meta.variables['patient']


def test_snapshot_not_from_other_formats():
    assert not hasattr(Nifti1Image(np.zeros((2, 3, 4)), np.eye(4)), 'minc_meta')

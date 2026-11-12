# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Load, save and compare every attribute of MINC files

Each file is loaded with nibabel, saved in its own format or converted to
the other, and loaded again.  Walking the two
files, every group, attribute and small dataset must be the same, except for
those a writer must change: ``ident``, ``minc_version`` and the history line
that is appended.  The scaling (image-min, image-max and valid_range) may
differ in form, but the data read back must be identical.

Besides nibabel's test files, the files in the directories listed in the
``NIBABEL_MINC_TEST_DIRS`` environment variable (separated by ``os.pathsep``)
are checked as well.

The SimpleITK tests read the written files with ITK's libminc, which reads
MINC2 with HDF5 support.
"""

import os
import shutil
from glob import glob
from os.path import basename
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import load, save
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1Image
from ..minc2 import Minc2Image
from ..minc_meta import minc1_dtype, normalize_attr
from ..optpkg import optional_package
from ..testing import data_path

h5py, have_h5py, _ = optional_package('h5py')
sitk, have_sitk, _ = optional_package('SimpleITK')

MINC1_MAGIC = b'CDF\x01'
MINC2_MAGIC = b'\x89HDF'

# Attributes written for the new file
NEW_ATTRS = ('ident', 'minc_version', 'history')
# Scaling, which may differ in form
SCALING_VARS = ('image-min', 'image-max')

# Invalid attributes of nibabel's test files, which the writer corrects:
# minc2_baddim.mnc has an xspace spacing of "xspace" and a length of 642
# for its 10 samples
CORRECTED = {
    'minc2_baddim.mnc': {
        ('minc-2.0/dimensions/xspace', '@spacing'): b'regular__',
        ('minc-2.0/dimensions/xspace', '@length'): 10,
    },
}


def _test_files(magic):
    dirs = [data_path] + [
        d for d in os.environ.get('NIBABEL_MINC_TEST_DIRS', '').split(os.pathsep) if d
    ]
    fnames = []
    for dirname in dirs:
        for fname in sorted(glob(pjoin(dirname, '*.mnc'))):
            with open(fname, 'rb') as fobj:
                if fobj.read(4) == magic:
                    fnames.append(fname)
    return fnames


MINC1_FILES = _test_files(MINC1_MAGIC)
MINC2_FILES = _test_files(MINC2_MAGIC)


def _walk_minc2(fname):
    """Return ``{(path, key): value}`` for groups, attributes and datasets

    The key is ``'@name'`` for an attribute, and ``'dtype'``, ``'shape'``
    and ``'value'`` for a dataset; values are left out for the image and for
    datasets of more than 1000 elements.
    """
    items = {}
    with h5py.File(fname, 'r') as h5f:

        def visit(name, obj):
            items[(name, 'kind')] = type(obj).__name__
            for key, value in obj.attrs.items():
                items[(name, f'@{key}')] = value
            if isinstance(obj, h5py.Dataset):
                items[(name, 'dtype')] = obj.dtype
                items[(name, 'shape')] = obj.shape
                if obj.size <= 1000 and not name.endswith('/image'):
                    items[(name, 'value')] = obj[()]

        for key, value in h5f.attrs.items():
            items[('/', f'@{key}')] = value
        h5f.visititems(visit)
    return items


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype.kind == b.dtype.kind and np.array_equal(a, b)


def _diff(items_a, items_b):
    """Keys whose items are lost, added or changed"""
    return {
        key
        for key in set(items_a) | set(items_b)
        if key not in items_a or key not in items_b or not _same(items_a[key], items_b[key])
    }


def _is_scaling(key):
    path, item = key
    name = path.rsplit('/', 1)[-1]
    if name in SCALING_VARS:
        return True
    return name == 'image' and item == '@valid_range'


def _check_history(old, new, writer='write_minc2'):
    old = '' if old is None else normalize_attr(old)
    new = normalize_attr(new)
    if old and not old.endswith('\n'):
        old += '\n'
    assert new.startswith(old)
    added = new[len(old) :]
    assert added.count('\n') == 1
    assert added.endswith('\n')
    assert f'>>> {writer} ' in added


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize('fname', MINC2_FILES, ids=basename)
def test_minc2_attribute_round_trip(tmp_path, fname):
    img = load(fname)
    assert isinstance(img, Minc2Image)
    out = str(tmp_path / 'out.mnc')
    save(img, out)
    rt_img = load(out)
    assert_array_equal(rt_img.get_fdata(), img.get_fdata())
    assert_array_equal(rt_img.affine, img.affine)
    before, after = _walk_minc2(fname), _walk_minc2(out)
    corrected = CORRECTED.get(basename(fname), {})
    for key in _diff(before, after):
        path, item = key
        if path == 'minc-2.0' and item[1:] in NEW_ATTRS:
            continue
        if _is_scaling(key):
            continue
        if key in corrected:
            assert after[key] == corrected[key]
            continue
        raise AssertionError(f'{key}: {before.get(key)!r} -> {after.get(key)!r}')
    assert after[('minc-2.0', '@ident')] != before.get(('minc-2.0', '@ident'))
    _check_history(before.get(('minc-2.0', '@history')), after[('minc-2.0', '@history')])


def _walk_minc1(fname):
    """Return ``{(variable, key): value}`` for a MINC1 file

    The variable is ``'GLOBAL'`` for global attributes; the keys are as for
    :func:`_walk_minc2`, with ``'dims'`` for the variable dimensions.
    """
    items = {}
    with open(fname, 'rb') as fobj:
        ncfile = netcdf_file(fobj, 'r', mmap=False)
        for key, value in ncfile._attributes.items():
            items[('GLOBAL', f'@{key}')] = normalize_attr(value)
        for name, var in ncfile.variables.items():
            for key, value in var._attributes.items():
                items[(name, f'@{key}')] = normalize_attr(value)
            items[(name, 'dims')] = tuple(var.dimensions)
            items[(name, 'dtype')] = minc1_dtype(var)
            if var.data.size <= 1000 and name != 'image':
                items[(name, 'value')] = np.array(var.data)
        del var
        ncfile.close()
    return items


def _walk_model(fname):
    """Walk a MINC1 or MINC2 file into libminc's MINC1 model

    MINC2 datasets are named as the MINC1 variables libminc maps them to
    (``/minc-2.0/info/patient`` is ``patient``), and the attributes of
    ``/minc-2.0`` are global.
    """
    with open(fname, 'rb') as fobj:
        magic = fobj.read(4)
    if magic == MINC1_MAGIC:
        return _walk_minc1(fname)
    items = {}
    with h5py.File(fname, 'r') as h5f:
        minc = h5f['minc-2.0']
        for key, value in minc.attrs.items():
            items[('GLOBAL', f'@{key}')] = normalize_attr(value)

        def visit(name, obj):
            for key, value in obj.attrs.items():
                items[(name, f'@{key}')] = normalize_attr(value)
            if isinstance(obj, h5py.Dataset):
                dimorder = normalize_attr(obj.attrs.get('dimorder', ''))
                dims = tuple(dim for dim in dimorder.split(',') if dim)
                items[(name, 'dims')] = dims[: obj.ndim]
                items[(name, 'dtype')] = obj.dtype.newbyteorder('=')
                if obj.size <= 1000 and name != 'image':
                    items[(name, 'value')] = obj[()]

        for group in ('dimensions', 'image/0', 'info'):
            if group in minc:
                minc[group].visititems(visit)
    return items


# Attributes that link MINC1 variables, or describe the file structure, and
# so differ between the formats
FORMAT_ATTRS = (
    '@parent',
    '@children',
    '@image-max',
    '@image-min',
    '@_FillValue',
    '@signtype',
    '@dimorder',
    '@length',
)


@pytest.mark.parametrize('fname', MINC1_FILES, ids=basename)
def test_minc1_attribute_round_trip(tmp_path, fname):
    img = load(fname)
    assert isinstance(img, Minc1Image)
    out = str(tmp_path / 'out.mnc')
    save(img, out)
    rt_img = load(out)
    assert isinstance(rt_img, Minc1Image)
    assert_array_equal(rt_img.get_fdata(), img.get_fdata())
    assert_array_equal(rt_img.affine, img.affine)
    before, after = _walk_minc1(fname), _walk_minc1(out)
    for key in _diff(before, after):
        if key[0] == 'GLOBAL' and key[1][1:] in NEW_ATTRS:
            continue
        if _is_scaling(key):
            continue
        raise AssertionError(f'{key}: {before.get(key)!r} -> {after.get(key)!r}')
    _check_history(
        before.get(('GLOBAL', '@history')), after[('GLOBAL', '@history')], 'write_minc1'
    )


@pytest.mark.skipif(not have_h5py, reason='Need h5py')
@pytest.mark.parametrize(
    ('fname', 'klass'),
    [(fname, Minc2Image) for fname in MINC1_FILES]
    + [(fname, Minc1Image) for fname in MINC2_FILES],
    ids=lambda value: basename(value) if isinstance(value, str) else value.__name__,
)
def test_convert_attribute_round_trip(tmp_path, fname, klass):
    # MINC1 to MINC2 and MINC2 to MINC1: in libminc's MINC1 model, only the
    # attributes that describe each format's structure may differ
    img = load(fname)
    out = str(tmp_path / 'out.mnc')
    save(klass.from_image(img), out)
    rt_img = load(out)
    assert isinstance(rt_img, klass)
    assert_array_equal(rt_img.get_fdata(), img.get_fdata())
    assert_array_equal(rt_img.affine, img.affine)
    before, after = _walk_model(fname), _walk_model(out)
    corrected = {
        (path.rsplit('/', 1)[-1], item): value
        for (path, item), value in CORRECTED.get(basename(fname), {}).items()
    }
    for key in _diff(before, after):
        name, item = key
        if name == 'GLOBAL' and item[1:] in NEW_ATTRS:
            continue
        if _is_scaling(key) or name == 'rootvariable' or item in FORMAT_ATTRS:
            continue
        if key in corrected:
            assert after[key] == normalize_attr(corrected[key])
            continue
        raise AssertionError(f'{key}: {before.get(key)!r} -> {after.get(key)!r}')
    writer = 'write_minc1' if klass is Minc1Image else 'write_minc2'
    _check_history(before.get(('GLOBAL', '@history')), after[('GLOBAL', '@history')], writer)


def _sitk_geometry(fname):
    image = sitk.ReadImage(str(fname))
    return image, (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())


@pytest.mark.skipif(not have_h5py or not have_sitk, reason='Need h5py and SimpleITK')
def test_sitk_slice_scaling(tmp_path):
    # small.mnc has int16 data with one image-min / image-max per slice;
    # the written file keeps its scaling
    src = pjoin(data_path, 'small.mnc')
    img = load(src)
    out = tmp_path / 'small.mnc'
    save(img, out)
    with h5py.File(out, 'r') as h5f:
        assert h5f['minc-2.0/image/0/image-max'].shape == (img.shape[0],)
    src_image, src_geometry = _sitk_geometry(src)
    out_image, out_geometry = _sitk_geometry(out)
    assert out_geometry == src_geometry
    assert out_image.GetPixelID() == src_image.GetPixelID()
    assert_array_equal(sitk.GetArrayFromImage(out_image), sitk.GetArrayFromImage(src_image))
    # ITK reads the scaled values as float32
    assert_allclose(sitk.GetArrayFromImage(out_image), img.get_fdata(), rtol=1e-6)


@pytest.mark.skipif(not have_h5py or not have_sitk, reason='Need h5py and SimpleITK')
def test_sitk_irregular_time(tmp_path):
    # A copy of minc2-4d-d.mnc with irregular time samples
    src = tmp_path / 'irregular.mnc'
    shutil.copyfile(pjoin(data_path, 'minc2-4d-d.mnc'), src)
    times = [0, 100, 250, 300, 600]
    with h5py.File(src, 'r+') as h5f:
        time = h5f['minc-2.0/dimensions/time']
        time.attrs['spacing'] = np.bytes_(b'irregular')
        time[...] = times
    img = load(src)
    out = tmp_path / 'out.mnc'
    save(img, out)
    with h5py.File(out, 'r') as h5f:
        time = h5f['minc-2.0/dimensions/time']
        assert time.attrs['spacing'] == b'irregular'
        assert_array_equal(time, times)
        assert_array_equal(h5f['minc-2.0/dimensions/time-width'], [100] * 5)
    assert_array_equal(load(out).minc_meta.variables['time'].data, times)
    src_image, src_geometry = _sitk_geometry(src)
    out_image, out_geometry = _sitk_geometry(out)
    assert out_geometry == src_geometry
    # ITK reads time as the vector components of a 3D image, with axes in
    # the reverse order of nibabel's
    assert out_image.GetNumberOfComponentsPerPixel() == len(times)
    data = sitk.GetArrayFromImage(out_image)
    assert_array_equal(data, np.transpose(img.get_fdata(), (3, 2, 1, 0)))
    # ITK keeps only the start and step of the time dimension ("tstart" and
    # "tstep"), so it cannot express the irregular times
    assert {'tstart', 'tstep'} <= set(out_image.GetMetaDataKeys())

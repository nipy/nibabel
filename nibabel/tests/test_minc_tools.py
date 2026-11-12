# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Check written MINC files with libminc's tools

Two sets of tools are used, if present:

* the MINC tools on the ``PATH`` (``mincinfo``, ``mincheader``,
  ``mincstats``, ``mnc2nii``, ``nii2mnc``).  Builds without HDF5 (such as
  the libminc 1.5 tools of FreeSurfer) read MINC1 only; the MINC2 checks
  skip with them;
* a minc-toolkit-v2 Apptainer (Singularity) image, named by the
  ``MINC_TOOLKIT_SIF`` environment variable and run with ``apptainer exec``.
  minc-toolkit-v2 1.9.18 has libminc 2.4.05 with HDF5, and reads and writes
  both formats.
"""

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from os.path import dirname
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import Nifti1Image, load, save
from ..affines import from_matvec
from ..eulerangles import euler2mat
from ..minc1 import Minc1Image
from ..minc1_write import write_minc1
from ..minc2 import Minc2Image
from ..minc2_write import write_minc2
from ..optpkg import optional_package
from ..testing import data_path
from .test_minc_roundtrip import (
    FORMAT_ATTRS,
    _diff,
    _is_scaling,
    _walk_minc2,
    _walk_model,
)

h5py, have_h5py, _ = optional_package('h5py')

OBLIQUE_AFFINE = from_matvec(euler2mat(0.3, -0.2, 0.5) @ np.diag([1.5, -2.0, 0.7]), [10, -20, 5])
WRITERS = {'MINC1': write_minc1, 'MINC2': write_minc2}
MAGIC = {'MINC1': b'CDF\x01', 'MINC2': b'\x89HDF'}

# Header lines that belong to the file being written
NEW_ATTRS = ('ident', 'minc_version', 'history')


def _magic(fname):
    with open(fname, 'rb') as fobj:
        return fobj.read(4)


# Running the tools


class Tools:
    """Run the MINC tools on the PATH, or in a minc-toolkit-v2 image"""

    def __init__(self, sif=None):
        self.sif = sif
        self._tmp = None
        self._reads_minc2 = None

    def available(self, name):
        if self.sif is None:
            return shutil.which(name) is not None
        return shutil.which('apptainer') is not None and os.path.isfile(self.sif)

    def run(self, name, *args, check=True):
        args = [str(arg) for arg in args]
        if self.sif is None:
            command = [shutil.which(name), *args]
            env = None
        else:
            # Bind the directories of the files, and a temporary directory
            # for the tools that make one
            paths = {os.path.abspath(dirname(a)) for a in args if os.path.exists(a)}
            paths |= {os.path.abspath(dirname(a)) for a in args if os.path.exists(dirname(a))}
            tmpdir = self._tmpdir()
            binds = ','.join(sorted(paths | {tmpdir}))
            command = ['apptainer', 'exec', '--bind', binds, self.sif, name, *args]
            env = {**os.environ, 'TMPDIR': tmpdir, 'APPTAINERENV_TMPDIR': tmpdir}
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if check and result.returncode:
            raise RuntimeError(f'{name} failed: {result.stderr}')
        return result.stdout if check else result

    def _tmpdir(self):
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix='minc-tools-')
            atexit.register(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp

    def reads_minc2(self):
        if self._reads_minc2 is None:
            self._reads_minc2 = False
            if self.available('mincinfo'):
                fname = pjoin(data_path, 'minc2_1_scale.mnc')
                self._reads_minc2 = self.run('mincinfo', fname, check=False).returncode == 0
        return self._reads_minc2


PATH_TOOLS = Tools()
SIF_TOOLS = Tools(os.environ.get('MINC_TOOLKIT_SIF'))
TOOLS = [
    pytest.param(PATH_TOOLS, id='path'),
    pytest.param(SIF_TOOLS, id='minc-toolkit'),
]


def _need(tools, *names, minc2=False):
    if tools.sif is None and tools is SIF_TOOLS:
        pytest.skip('MINC_TOOLKIT_SIF not set')
    for name in names:
        if not tools.available(name):
            pytest.skip(f'{name} not available')
    if minc2 and not tools.reads_minc2():
        pytest.skip('MINC tools cannot read MINC2 (built without HDF5)')
    if minc2 and not have_h5py:
        pytest.skip('Need h5py')


def _need_format(tools, fmt, *names):
    _need(tools, *names, minc2=fmt == 'MINC2')


# Parsing tool output


def _mincinfo_dims(tools, fname):
    """Return [(name, length, start, step, cosines)] of the image dimensions

    ``mincinfo -dimnames`` lists the dimensions of a MINC2 file in the order
    of the HDF5 group, so the image dimensions come from ``-vardims``.
    """
    names = tools.run('mincinfo', '-vardims', 'image', fname).split()
    dims = []
    for name in names:
        length = int(tools.run('mincinfo', '-dimlength', name, fname))
        if name == 'vector_dimension':
            dims.append((name, length, None, None, None))
            continue
        start = float(tools.run('mincinfo', '-attvalue', f'{name}:start', fname))
        step = float(tools.run('mincinfo', '-attvalue', f'{name}:step', fname))
        cosines = None
        if name.endswith('space'):
            text = tools.run('mincinfo', '-attvalue', f'{name}:direction_cosines', fname)
            cosines = np.array(text.split(), dtype=float)
        dims.append((name, length, start, step, cosines))
    return dims


def _mincstats(tools, fname):
    out = tools.run('mincstats', '-quiet', '-min', '-max', '-mean', fname)
    return np.array(out.split(), dtype=float)


_ATT_LINE = re.compile(r'^\s*(?P<var>[^\s:]*):(?P<att>\S+) = (?P<value>.*) ;$')


def _mincheader(tools, fname):
    """Return ``{(variable, attribute): value text}`` from mincheader

    Global attributes have variable ``''``; continuation lines of string
    values are joined.  Variable declarations are under the ``'decl'``
    attribute.
    """
    text = tools.run('mincheader', fname)
    lines = text.splitlines()
    items = {}
    pending = None
    for line in lines:
        stripped = line.strip()
        if pending is not None:
            pending[1].append(stripped)
            if stripped.endswith(';'):
                key, parts = pending
                items[key] = ' '.join(parts)[: -len(' ;')].strip()
                pending = None
            continue
        match = _ATT_LINE.match(line)
        if match:
            items[(match['var'], match['att'])] = match['value']
            continue
        match = re.match(r'^\s*(?P<var>[^\s:]*):(?P<att>\S+) = (?P<value>.*)$', line)
        if match:
            pending = ((match['var'], match['att']), [match['value']])
            continue
        match = re.match(r'^\s*(\w+) ([\w-]+)(\(.*\))? ;$', stripped)
        if match and match[1] in (
            'byte',
            'char',
            'short',
            'int',
            'float',
            'double',
            'ubyte',
            'ushort',
            'uint',
        ):
            items[(match[2], 'decl')] = f'{match[1]}{match[3] or ""}'
    return items


def _header_diff(items_a, items_b, skip=()):
    keys = set(items_a) | set(items_b)
    return {
        key: (items_a.get(key), items_b.get(key))
        for key in keys
        if items_a.get(key) != items_b.get(key)
        and not (key[0] == '' and key[1] in NEW_ATTRS)
        and key not in skip
    }


def _write_oblique(fname, fmt, dtype=np.float32, shape=(5, 6, 7)):
    rng = np.random.default_rng(3)
    data = rng.normal(100, 30, shape)
    WRITERS[fmt](data, OBLIQUE_AFFINE, fname, dtype=dtype)
    return load(fname).get_fdata()


# Tests


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
@pytest.mark.parametrize('shape', [(5, 6, 7), (2, 5, 6, 7)])
def test_mincinfo(tmp_path, tools, fmt, shape):
    # libminc reads the dimensions, steps, starts and cosines we write
    _need_format(tools, fmt, 'mincinfo')
    fname = str(tmp_path / 'img.mnc')
    _write_oblique(fname, fmt, shape=shape)
    img = load(fname)
    dims = _mincinfo_dims(tools, fname)
    meta = img.minc_meta
    assert [d[0] for d in dims] == list(meta.dimorder)
    assert [d[1] for d in dims] == list(img.shape)
    for name, _, start, step, cosines in dims:
        attrs = meta.dimensions[name].attrs
        assert_allclose(start, attrs['start'], rtol=1e-5, atol=1e-5)
        assert_allclose(step, attrs['step'], rtol=1e-6)
        if cosines is not None:
            assert_allclose(cosines, attrs['direction_cosines'], atol=1e-6)


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.int32, np.float32, np.float64])
def test_mincstats(tmp_path, tools, fmt, dtype):
    _need_format(tools, fmt, 'mincstats')
    fname = str(tmp_path / 'img.mnc')
    data = _write_oblique(fname, fmt, dtype=dtype, shape=(2, 5, 6, 7))
    expected = [data.min(), data.max(), data.mean()]
    assert_allclose(_mincstats(tools, fname), expected, rtol=1e-6)


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
def test_attvalue(tmp_path, tools, fmt):
    # mincinfo -attvalue reads the attributes of info variables, dimensions,
    # the image and the history, copied from a file with metadata
    _need_format(tools, fmt, 'mincinfo')
    src = load(pjoin(data_path, 'minc1_4d.mnc'))
    fname = str(tmp_path / 'img.mnc')
    klass = Minc1Image if fmt == 'MINC1' else Minc2Image
    save(klass.from_image(src), fname)
    meta = load(fname).minc_meta

    def attvalue(name):
        return tools.run('mincinfo', '-attvalue', name, fname).strip()

    assert attvalue('study:modality') == 'MRI__'
    assert attvalue('time:alignment') == 'start_'
    assert attvalue('xspace:spacing') == 'regular__'
    assert float(attvalue('zspace:step')) == meta.dimensions['zspace'].attrs['step']
    assert attvalue('image:signtype') == 'unsigned'
    assert np.array(attvalue('image:valid_range').split(), float).tolist() == [0, 255]
    history = tools.run('mincinfo', '-attvalue', ':history', fname)
    assert history.rstrip('\n') == meta.history.rstrip('\n')
    assert attvalue(':ident') == meta.global_attrs['ident']


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
def test_mincheader_info(tmp_path, tools, fmt):
    # mincheader shows the info groups, dimensions and history
    _need_format(tools, fmt, 'mincheader')
    src = load(pjoin(data_path, 'minc2_1_scale.mnc'))
    fname = str(tmp_path / 'img.mnc')
    klass = Minc1Image if fmt == 'MINC1' else Minc2Image
    save(klass.from_image(src), fname)
    header = _mincheader(tools, fname)
    assert header[('patient', 'full_name')] == '"mnc2nii tiny.mnc tiny.nii"'
    for name in ('study', 'acquisition'):
        assert header[(name, 'vartype')] == '"group________"'
    assert header[('image', 'decl')] == 'byte(zspace, yspace, xspace)'
    assert header[('xspace', 'step')] == '2.'
    assert 'nii2mnc tiny_uint8.nii tiny_uint8.mnc' in header[('', 'history')]
    assert '>>> write_minc' in header[('', 'history')]


def _nii2mnc_reference(tools, tmp_path, fmt, data, affine):
    """Write NIfTI `data`, `affine` and convert it with nii2mnc

    nii2mnc of libminc 2.4 writes MINC2; mincconvert makes the MINC1
    reference from it.  nii2mnc of libminc 1.x writes MINC1.
    """
    nii = str(tmp_path / 'ref.nii')
    Nifti1Image(data, affine).to_filename(nii)
    ref = str(tmp_path / 'ref.mnc')
    tools.run('nii2mnc', '-quiet', nii, ref)
    if fmt == 'MINC1' and _magic(ref) != MAGIC['MINC1']:
        _need(tools, 'mincconvert')
        tools.run('mincconvert', ref, str(tmp_path / 'ref1.mnc'))
        ref = str(tmp_path / 'ref1.mnc')
    if _magic(ref) != MAGIC[fmt]:
        pytest.skip(f'Cannot make a {fmt} reference with these tools')
    return ref


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32])
@pytest.mark.parametrize('ndim', [3, 4])
def test_header_round_trip(tmp_path, tools, fmt, dtype, ndim):
    # nibabel saving a libminc-written file gives the same header, except
    # for the ident, version and history of the new file
    _need_format(tools, fmt, 'nii2mnc', 'mincheader')
    shape = (4, 5, 6, 3)[:ndim]
    data = (np.arange(np.prod(shape)).reshape(shape) % 37).astype(dtype)
    affine = from_matvec(np.diag([2.0, 3.0, 4.0]), [-10, 5, 7])
    ref = _nii2mnc_reference(tools, tmp_path, fmt, data, affine)
    out = str(tmp_path / 'out.mnc')
    save(load(ref), out)
    assert _magic(out) == MAGIC[fmt]
    diff = _header_diff(_mincheader(tools, ref), _mincheader(tools, out))
    assert diff == {}
    history = _mincheader(tools, out)[('', 'history')]
    assert '>>> write_minc' in history


# What a new file differs in from nii2mnc's: nii2mnc keeps the valid range
# of the data, rather than of the type, and a single image-min / image-max
# rather than one per slice; libminc 1.x leaves out complete
NII2MNC_DIFFERENCES = {
    ('image', 'valid_range'),
    ('image', 'complete'),
    ('image-min', 'decl'),
    ('image-max', 'decl'),
    ('image-min', 'dimorder'),
    ('image-max', 'dimorder'),
}


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
@pytest.mark.parametrize('dtype', [np.int16, np.float32])
def test_header_new_file(tmp_path, tools, fmt, dtype):
    # A new file, from the same data as nii2mnc's reference, has the same
    # header, but for the data range and slice-wise scaling
    _need_format(tools, fmt, 'nii2mnc', 'mincheader')
    shape = (4, 5, 6, 3)
    data = (np.arange(np.prod(shape)).reshape(shape) % 37).astype(dtype)
    affine = from_matvec(np.diag([2.0, 3.0, 4.0]), [-10, 5, 7])
    ref = _nii2mnc_reference(tools, tmp_path, fmt, data, affine)
    out = str(tmp_path / 'out.mnc')
    # nii2mnc puts time first and reverses the spatial axes
    minc_data = np.moveaxis(data, -1, 0).transpose((0, 3, 2, 1))
    minc_affine = affine[:, [2, 1, 0, 3]]
    WRITERS[fmt](minc_data, minc_affine, out, dtype=dtype)
    assert_array_equal(load(out).get_fdata(), load(ref).get_fdata())
    ref_header, out_header = _mincheader(tools, ref), _mincheader(tools, out)
    skip = set(NII2MNC_DIFFERENCES)
    if fmt == 'MINC1' and ('rootvariable', 'decl') not in ref_header:
        # libminc 2.4 makes MINC1 with mincconvert, which leaves out the
        # links between MINC1 variables that its MINC1 functions write
        skip |= {
            key
            for key in set(ref_header) | set(out_header)
            if key[0] == 'rootvariable' or f'@{key[1]}' in FORMAT_ATTRS
        }
    if fmt == 'MINC1' and dtype == np.float32 and ('', 'ident') not in ref_header:
        # libminc 1.x gives float images a signtype
        skip.add(('image', 'signtype'))
    assert _header_diff(ref_header, out_header, skip) == {}


@pytest.mark.parametrize('tools', TOOLS)
def test_mnc2nii(tmp_path, tools):
    # mnc2nii of our MINC1 (and, if it can, MINC2) files gives our data
    _need(tools, 'mnc2nii')
    for fmt in ('MINC1', 'MINC2'):
        if fmt == 'MINC2' and not (have_h5py and tools.reads_minc2()):
            continue
        fname = str(tmp_path / f'{fmt}.mnc')
        data = _write_oblique(fname, fmt)
        img = load(fname)
        nii = str(tmp_path / f'{fmt}.nii')
        tools.run('mnc2nii', '-quiet', '-float', fname, nii)
        _check_nifti(nii, data, img.affine)


def _check_nifti(nii, data, affine):
    """Compare NIfTI `nii` from mnc2nii with MINC `data` and `affine`"""
    from ..nifti1 import Nifti1Header

    with open(nii, 'rb') as fobj:
        raw = fobj.read()
    hdr = Nifti1Header(binaryblock=raw[:348], check=False)
    shape = hdr.get_data_shape()
    nii_data = np.frombuffer(
        raw, hdr.get_data_dtype(), count=int(np.prod(shape)), offset=int(hdr['vox_offset'])
    ).reshape(shape, order='F')
    assert nii_data.size == data.size
    if hdr['sform_code'] > 0 or hdr['qform_code'] > 0:
        nii_affine = hdr.get_best_affine()
        vox2vox = np.linalg.solve(affine, nii_affine)
        ijk = np.indices(nii_data.shape).reshape(3, -1)
        src = vox2vox[:3, :3] @ ijk + vox2vox[:3, 3:]
        assert_allclose(src, np.round(src), atol=1e-3)
        src = np.round(src).astype(int)
        assert_allclose(nii_data.reshape(-1), data[tuple(src)], rtol=1e-6, atol=1e-4)
    else:
        # libminc 1.x writes no affine; its axes are xspace, yspace, zspace
        assert_allclose(np.sort(nii_data.ravel()), np.sort(data.ravel()), rtol=1e-6)


def _same_meta(meta_a, meta_b):
    """Info variables and dimension attributes that are kept"""
    assert set(meta_a.info) == set(meta_b.info)
    for name, var in meta_a.info.items():
        for key, value in var.attrs.items():
            if key not in ('parent', 'dimorder'):
                assert meta_b.info[name].attrs[key] == value
    for name, var in meta_a.dimensions.items():
        for key in ('start', 'step', 'direction_cosines', 'units', 'alignment', 'comments'):
            if key in var.attrs:
                assert_array_equal(meta_b.dimensions[name].attrs[key], var.attrs[key])


@pytest.mark.parametrize('src', ['minc1_4d.mnc', 'minc2_1_scale.mnc', 'small.mnc'])
def test_mincconvert(tmp_path, src):
    # mincconvert -2 of our MINC1 files, and mincconvert (to MINC1) of our
    # MINC2 files: nibabel reads them with the same data, affine and
    # metadata
    _need(SIF_TOOLS, 'mincconvert', minc2=True)
    img = load(pjoin(data_path, src))
    for klass, option, back in ((Minc1Image, '-2', 'MINC2'), (Minc2Image, None, 'MINC1')):
        ours = str(tmp_path / 'ours.mnc')
        save(klass.from_image(img), ours)
        converted = str(tmp_path / 'converted.mnc')
        args = ['-clobber'] + ([option] if option else []) + [ours, converted]
        SIF_TOOLS.run('mincconvert', *args)
        assert _magic(converted) == MAGIC[back]
        ours_img, conv_img = load(ours), load(converted)
        assert_array_equal(conv_img.get_fdata(), ours_img.get_fdata())
        assert_allclose(conv_img.affine, ours_img.affine, rtol=0, atol=1e-12)
        _same_meta(ours_img.minc_meta, conv_img.minc_meta)
        assert conv_img.minc_meta.history.startswith(ours_img.minc_meta.history)


@pytest.mark.parametrize('src', ['minc1_4d.mnc', 'tiny.mnc', 'minc2-4d-d.mnc'])
@pytest.mark.parametrize('tool', ['mincconvert', 'minccalc'])
def test_libminc_minc2_round_trip(tmp_path, src, tool):
    # libminc's own MINC2 files load and save with every attribute kept
    _need(SIF_TOOLS, tool, minc2=True)
    ref = str(tmp_path / 'ref.mnc')
    if tool == 'mincconvert':
        SIF_TOOLS.run('mincconvert', '-2', pjoin(data_path, src), ref)
    else:
        SIF_TOOLS.run(
            'minccalc', '-2', '-quiet', '-expression', 'A[0]*2', pjoin(data_path, src), ref
        )
    assert _magic(ref) == MAGIC['MINC2']
    img = load(ref)
    out = str(tmp_path / 'out.mnc')
    save(img, out)
    assert_array_equal(load(out).get_fdata(), img.get_fdata())
    before, after = _walk_minc2(ref), _walk_minc2(out)
    for key in _diff(before, after):
        path, item = key
        if path == 'minc-2.0' and item[1:] in NEW_ATTRS:
            continue
        if _is_scaling(key):
            continue
        raise AssertionError(f'{key}: {before.get(key)!r} -> {after.get(key)!r}')
    # libminc reads the result back
    assert_allclose(_mincstats(SIF_TOOLS, out), _mincstats(SIF_TOOLS, ref), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('src', ['minc1_4d.mnc', 'minc2_1_scale.mnc'])
def test_libminc_convert_model(tmp_path, src):
    # Our conversion and mincconvert's give the same libminc model
    _need(SIF_TOOLS, 'mincconvert', minc2=True)
    img = load(pjoin(data_path, src))
    to_minc2 = isinstance(img, Minc1Image)
    theirs = str(tmp_path / 'theirs.mnc')
    SIF_TOOLS.run('mincconvert', *(['-2'] if to_minc2 else []), pjoin(data_path, src), theirs)
    ours = str(tmp_path / 'ours.mnc')
    save((Minc2Image if to_minc2 else Minc1Image).from_image(img), ours)
    a, b = _walk_model(theirs), _walk_model(ours)
    diffs = {
        key
        for key in _diff(a, b)
        if not (
            _is_scaling(key)
            or key[1] in FORMAT_ATTRS
            or key[0] == 'rootvariable'
            or (key[0] == 'GLOBAL' and key[1][1:] in NEW_ATTRS)
        )
    }
    assert diffs == set()


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
def test_irregular(tmp_path, tools, fmt):
    # libminc reads files with irregular time samples and their widths
    _need_format(tools, fmt, 'mincinfo', 'mincstats', 'mincheader')
    if not have_h5py:
        pytest.skip('Need h5py')
    from .test_minc2_write import _make_irregular

    src = str(tmp_path / 'src.mnc')
    data = np.arange(120, dtype=np.float32).reshape((2, 3, 4, 5))
    write_minc2(data, np.eye(4), src)
    _make_irregular(src, 'time', [0, 7.5], [1, 2])
    fname = str(tmp_path / 'img.mnc')
    save((Minc1Image if fmt == 'MINC1' else Minc2Image).from_image(load(src)), fname)
    assert tools.run('mincinfo', '-attvalue', 'time:spacing', fname).strip() == 'irregular'
    assert_allclose(_mincstats(tools, fname), [0, 119, 59.5])
    header = _mincheader(tools, fname)
    assert header[('time', 'decl')] == 'double(time)'
    assert header[('time-width', 'decl')] == 'double(time)'


def _feature_data(feature):
    """Data, writer options and the mean libminc reads, for a feature"""
    if feature == 'vector':
        data = (np.arange(360).reshape((4, 5, 6, 3)) % 256).astype(np.uint8)
        return data, {'vector_dimension': True}, data.mean()
    if feature == 'gzip':
        return np.arange(60, dtype=np.int16).reshape((3, 4, 5)), {'compression': True}, 29.5
    if feature == 'int8':
        return (np.arange(60) - 20).astype(np.int8).reshape((3, 4, 5)), {}, 9.5
    if feature == 'uint32':
        data = (np.arange(60) + 4_000_000_000).astype(np.uint32).reshape((3, 4, 5))
        return data, {}, data.mean()
    if feature == 'complex':
        real = np.arange(60, dtype=np.float32)
        data = (real + 1j * real[::-1]).astype(np.complex64).reshape((3, 4, 5))
        # libminc reads the real part
        return data, {}, 29.5
    raise ValueError(feature)


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize(
    ('fmt', 'feature'),
    [
        ('MINC1', 'vector'),
        ('MINC2', 'vector'),
        ('MINC2', 'gzip'),
        ('MINC1', 'int8'),
        ('MINC2', 'int8'),
        ('MINC1', 'uint32'),
        ('MINC2', 'uint32'),
        ('MINC2', 'complex'),
    ],
)
def test_features(tmp_path, tools, fmt, feature):
    _need_format(tools, fmt, 'mincstats')
    data, options, mean = _feature_data(feature)
    fname = str(tmp_path / 'img.mnc')
    WRITERS[fmt](data, np.eye(4), fname, **options)
    assert_allclose(_mincstats(tools, fname)[2], mean, rtol=1e-7)


@pytest.mark.parametrize('tools', TOOLS)
def test_labels_minc1_interface(tmp_path, tools):
    # libminc keeps label names in an enumerated HDF5 type, which its MINC2
    # interface (used by ITK) reads, but not its MINC1 interface, which most
    # tools use
    _need_format(tools, 'MINC2', 'mincstats')
    data = (np.arange(60) % 4).astype(np.uint8).reshape((3, 4, 5))
    fname = str(tmp_path / 'img.mnc')
    write_minc2(data, np.eye(4), fname, labels={0: 'a', 1: 'b', 2: 'c', 3: 'd'})
    result = tools.run('mincstats', '-quiet', '-mean', fname, check=False)
    assert 'Unknown type class' in result.stdout + result.stderr


def _walk_any(fname):
    if _magic(fname) == MAGIC['MINC2']:
        return _walk_minc2(fname), 'minc-2.0'
    from .test_minc_roundtrip import _walk_minc1

    return _walk_minc1(fname), 'GLOBAL'


def _check_attribute_round_trip(src, out):
    """Every attribute of `src` is in `out`, but the new file's own"""
    img = load(src)
    save(img, out)
    assert_array_equal(load(out).get_fdata(), img.get_fdata())
    assert_array_equal(load(out).affine, img.affine)
    (before, glob), (after, _) = _walk_any(src), _walk_any(out)
    for key in _diff(before, after):
        if key[0] == glob and key[1][1:] in NEW_ATTRS:
            continue
        if _is_scaling(key):
            continue
        raise AssertionError(f'{key}: {before.get(key)!r} -> {after.get(key)!r}')


@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
@pytest.mark.parametrize(
    'dimorder', ['zfrequency,yfrequency,xfrequency', 'tfrequency,zspace,yspace,xspace']
)
def test_libminc_frequency(tmp_path, fmt, dimorder):
    # rawtominc writes frequency dimensions; nibabel reads them, keeps every
    # attribute through a save, and libminc reads nibabel's files
    _need(SIF_TOOLS, 'rawtominc', 'mincconvert', minc2=True)
    shape = (2, 3, 4, 5)[-len(dimorder.split(',')) :]
    raw = tmp_path / 'raw.bin'
    raw.write_bytes(bytes(range(int(np.prod(shape)))))
    ref = str(tmp_path / 'ref2.mnc')
    SIF_TOOLS.run(
        'rawtominc',
        '-clobber',
        '-byte',
        '-dimorder',
        dimorder,
        '-input',
        raw,
        ref,
        *(str(n) for n in shape),
    )
    if fmt == 'MINC1':
        SIF_TOOLS.run('mincconvert', '-clobber', ref, str(tmp_path / 'ref1.mnc'))
        ref = str(tmp_path / 'ref1.mnc')
    assert _magic(ref) == MAGIC[fmt]
    img = load(ref)
    assert ','.join(img.minc_meta.dimorder) == dimorder
    assert_array_equal(img.get_fdata().ravel(), np.arange(np.prod(shape)))
    out = str(tmp_path / 'out.mnc')
    _check_attribute_round_trip(ref, out)
    assert SIF_TOOLS.run('mincinfo', '-vardims', 'image', out).split() == dimorder.split(',')


@pytest.mark.parametrize('tools', TOOLS)
@pytest.mark.parametrize('fmt', ['MINC1', 'MINC2'])
def test_frequency_mincinfo(tmp_path, tools, fmt):
    # libminc reads the start and step of frequency dimensions we write
    _need_format(tools, fmt, 'mincinfo')
    names = ('zfrequency', 'yfrequency', 'xfrequency')
    affine = np.array([[0, 0, 0.5, -1], [0, 0.25, 0, 2], [2.0, 0, 0, 3], [0, 0, 0, 1]])
    fname = str(tmp_path / 'img.mnc')
    WRITERS[fmt](np.zeros((3, 4, 5), np.int16), affine, fname, dimension_names=names)
    dims = _mincinfo_dims(tools, fname)
    assert [d[0] for d in dims] == list(names)
    assert [(d[2], d[3]) for d in dims] == [(3, 2), (2, 0.25), (-1, 0.5)]


RECORD_CDL = """netcdf rec {
dimensions:
    time = UNLIMITED ;
    zspace = 2 ;
    yspace = 3 ;
    xspace = 4 ;
variables:
    double time(time) ;
        time:varid = "MINC standard variable" ;
        time:vartype = "dimension____" ;
        time:version = "MINC Version    1.0" ;
        time:spacing = "irregular" ;
        time:alignment = "start_" ;
        time:units = "s" ;
    int zspace ;
        zspace:step = 2. ;
        zspace:start = -1. ;
    int yspace ;
        yspace:step = 3. ;
    int xspace ;
        xspace:start = 5. ;
    double image-max(time, zspace) ;
    double image-min(time, zspace) ;
    short image(time, zspace, yspace, xspace) ;
        image:signtype = "signed__" ;
        image:valid_range = 0., 100. ;
        image:image-max = "--->image-max" ;
        image:image-min = "--->image-min" ;
    :history = "made by mincgen\\n" ;
data:
 time = 0, 2.5, 7 ;
 image-max = 100, 100, 50, 50, 10, 10 ;
 image-min = 0, 0, 0, 0, 0, 0 ;
 image = {image} ;
}
""".replace('{image}', ', '.join(str(i % 101) for i in range(72)))


def test_libminc_record_dimension(tmp_path):
    # mincgen of libminc 2.4 writes a MINC1 record dimension (with
    # MINC_FORCE_V2=0); nibabel reads it, keeps every attribute through a
    # save, and libminc reads nibabel's file
    _need(SIF_TOOLS, 'mincgen', 'mincstats')
    cdl = tmp_path / 'rec.cdl'
    cdl.write_text(RECORD_CDL)
    ref = str(tmp_path / 'ref.mnc')
    os.environ['APPTAINERENV_MINC_FORCE_V2'] = '0'
    try:
        SIF_TOOLS.run('mincgen', '-b', '-o', ref, cdl)
    finally:
        del os.environ['APPTAINERENV_MINC_FORCE_V2']
    assert _magic(ref) == MAGIC['MINC1']
    img = load(ref)
    assert img.minc_meta.record_dimension == 'time'
    assert img.shape == (3, 2, 3, 4)
    out = str(tmp_path / 'out.mnc')
    _check_attribute_round_trip(ref, out)
    assert load(out).minc_meta.record_dimension == 'time'
    assert_allclose(_mincstats(SIF_TOOLS, out), _mincstats(SIF_TOOLS, ref), rtol=1e-6)
    # libminc cannot write the record dimension to MINC2 either
    result = SIF_TOOLS.run('mincconvert', '-2', ref, str(tmp_path / 'ref2.mnc'), check=False)
    assert 'H5Pset_chunk' in result.stderr or result.returncode


@pytest.mark.parametrize('tools', TOOLS)
def test_record_dimension_tools(tmp_path, tools):
    _need_format(tools, 'MINC1', 'mincinfo', 'mincstats')
    data = np.arange(3 * 2 * 3 * 4, dtype=np.int16).reshape((3, 2, 3, 4))
    fname = str(tmp_path / 'img.mnc')
    write_minc1(data, np.eye(4), fname, record_dimension=True)
    assert tools.run('mincinfo', '-dimlength', 'time', fname).strip() == '3'
    assert_allclose(_mincstats(tools, fname), [0, 71, 35.5])


MULTIRES_C = r"""
#include <stdlib.h>
#include <minc2.h>

/* Write a 3D MINC2 volume with resolution levels through libminc */
int main(int argc, char **argv)
{
    mihandle_t vol;
    midimhandle_t dims[3];
    mivolumeprops_t props;
    misize_t start[3] = {0, 0, 0}, count[3];
    int sizes[3] = {8, 12, 16}, depth = atoi(argv[2]);
    const char *names[3] = {"zspace", "yspace", "xspace"};
    double steps[3] = {3.0, 2.0, 1.0}, starts[3] = {-10.0, 5.0, 7.0};
    int i, n = sizes[0] * sizes[1] * sizes[2];
    double *data = malloc(n * sizeof(double));
    for (i = 0; i < n; i++) data[i] = (i * 7) % 101;
    for (i = 0; i < 3; i++) {
        micreate_dimension(names[i], MI_DIMCLASS_SPATIAL, MI_DIMATTR_REGULARLY_SAMPLED,
                           sizes[i], &dims[i]);
        miset_dimension_separation(dims[i], steps[i]);
        miset_dimension_start(dims[i], starts[i]);
        count[i] = sizes[i];
    }
    minew_volume_props(&props);
    if (miset_props_multi_resolution(props, 1, depth) < 0) return 1;
    if (micreate_volume(argv[1], 3, dims, MI_TYPE_SHORT, MI_CLASS_REAL, props, &vol) < 0)
        return 2;
    if (micreate_volume_image(vol) < 0) return 3;
    miset_volume_range(vol, 100.0, 0.0);
    miset_volume_valid_range(vol, 32767.0, -32768.0);
    if (miset_real_value_hyperslab(vol, MI_TYPE_DOUBLE, start, count, data) < 0) return 4;
    if (miflush_from_resolution(vol, depth) < 0) return 5;
    miclose_volume(vol);
    return 0;
}
"""


def _libminc_multires(tmp_path, depth=2):
    """A file with resolution levels that libminc writes, or skip"""
    _need(SIF_TOOLS, 'mincinfo', minc2=True)
    source = tmp_path / 'multires.c'
    source.write_text(MULTIRES_C)
    program = str(tmp_path / 'multires')
    result = SIF_TOOLS.run(
        'gcc',
        source,
        '-o',
        program,
        '-I/opt/minc/1.9.18/include',
        '-L/opt/minc/1.9.18/lib',
        '-lminc2',
        '-lhdf5',
        '-lm',
        check=False,
    )
    if result.returncode:
        pytest.skip('Cannot compile against libminc in the minc-toolkit image')
    fname = str(tmp_path / 'multires.mnc')
    os.environ['APPTAINERENV_MINC_COMPRESS'] = '0'
    try:
        SIF_TOOLS.run(program, fname, depth)
    finally:
        del os.environ['APPTAINERENV_MINC_COMPRESS']
    return fname


def test_libminc_resolution_levels(tmp_path):
    # libminc writes lower resolution levels with miset_props_multi_resolution
    # and miflush_from_resolution; nibabel reads each of them
    fname = _libminc_multires(tmp_path)
    img = load(fname)
    assert dict(img.minc_meta.resolution_levels) == {1: (4, 6, 8), 2: (2, 3, 4)}
    full = img.get_fdata()
    # Level 1 is the mean of 2 x 2 x 2 blocks of the full resolution
    level1 = load(fname, resolution_level=1)
    blocks = full.reshape((4, 2, 6, 2, 8, 2)).mean(axis=(1, 3, 5))
    assert_allclose(level1.get_fdata(), blocks, atol=100 / 65535)
    # libminc makes level 2 from the stored values of level 1, which it
    # reads with the scaling of the full resolution rather than that of
    # level 1; nibabel reads the stored level 2 as libminc's level 2 reader
    # does, with the scaling of level 2
    level2 = load(fname, resolution_level=2)
    with h5py.File(fname, 'r') as h5f:
        raw1 = h5f['minc-2.0/image/1/image'][()].astype(float)
    misread = (raw1 + 32768) / 65535 * 100
    assert_allclose(
        level2.get_fdata(), misread.reshape((2, 2, 3, 2, 4, 2)).mean(axis=(1, 3, 5)), atol=1e-3
    )
    # mincinfo shows the full resolution only
    assert SIF_TOOLS.run('mincinfo', '-dimlength', 'xspace', fname).strip() == '16'

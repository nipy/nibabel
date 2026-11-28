# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read images in the NRRD format

NRRD ("nearly raw raster data") is the format of the Teem toolkit, used by 3D
Slicer, ITK and many other tools. The format is defined at:

* http://teem.sourceforge.net/nrrd/format.html
* http://teem.sourceforge.net/nrrd/descformat.html

This module reads NRRD files. Writing is not supported yet.

What is supported:

* Magic versions ``NRRD0001`` to ``NRRD0005``.
* ``raw``, ``gzip`` and ``bzip2`` encodings. Data are read lazily through an
  array proxy. Raw data can be memory mapped; compressed data are decompressed
  only as far as needed for a slice.
* Data in the same file as the header (``.nrrd``) or in one separate data file
  named by a ``.nhdr`` header. A relative data file name is relative to the
  header, not to the working directory.
* ``line skip`` and ``byte skip``, including ``byte skip: -1`` for raw data.
* The spaces ``left-posterior-superior`` (``LPS``), ``right-anterior-superior``
  (``RAS``), ``left-anterior-superior`` (``LAS``) and ``scanner-xyz`` (treated as
  LPS, as ITK and 3D Slicer do). ``3D-right-handed`` files and files with only a
  ``space dimension`` load with no axis flips and a warning, because their axes
  have no anatomical meaning.

Conventions:

* The affine maps voxel indices to RAS+ millimeters, as for other nibabel
  images. NRRD space directions are columns of the affine after conversion to
  RAS+. For an LPS file, the first two rows of the affine are negated.
* Axes are reordered so that the spatial axes come first, then other domain
  axes such as time or list axes, then component axes such as vectors, colors
  or tensors. A 3D vector field over time with on-disk axes (vector, x, y, z,
  time) loads with shape ``(x, y, z, time, 3)``, as in ITK.
  ``img.header.get_axis_order()`` gives the on-disk axis of each loaded axis.
  Size one axes are kept.
* The header fields are available, in on-disk axis order, as
  ``img.header.fields``, and key/value pairs as ``img.header.keyvalues``. The
  measurement frame is never applied to the voxel values;
  ``img.header.get_measurement_frame()`` returns it in RAS+.

The header text is parsed with `pynrrd <https://github.com/mhe/pynrrd>`_. Install
it with ``pip install nibabel[nrrd]``.

Examples
--------
>>> import os
>>> import nibabel as nib
>>> from nibabel.testing import data_path
>>> img = nib.load(os.path.join(data_path, 'nrrd_list_vector.nrrd'))
>>> img.shape
(3, 4, 5, 2, 3)
>>> img.header.fields['kinds']
['vector', 'domain', 'domain', 'domain', 'list']
>>> img.header.get_axis_order()
(1, 2, 3, 4, 0)
>>> nib.aff2axcodes(img.affine)
('L', 'P', 'S')
"""

from __future__ import annotations

import bz2
import gzip
import io
import os
import re
import typing as ty
import warnings
import zlib
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from .arrayproxy import ArrayProxy
from .filebasedimages import ImageFileError
from .fileholders import FileHolder
from .filename_parser import _stringify_path
from .fileslice import canonical_slicers
from .openers import Opener
from .optpkg import optional_package
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage

if ty.TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    import numpy.typing as npt

    from ._typing import Self
    from .filebasedimages import FileSniff
    from .fileholders import FileMap
    from .filename_parser import FileSpec

__doctest_requires__ = {('*',): ['nrrd']}

NRRD_MAGIC = b'NRRD000'
SUPPORTED_VERSIONS = (1, 2, 3, 4, 5)

#: Diagonal of the matrix that maps each supported space to RAS+
SPACE_TO_RAS: dict[str, tuple[int, int, int]] = {
    'left-posterior-superior': (-1, -1, 1),
    'right-anterior-superior': (1, 1, 1),
    'left-anterior-superior': (-1, 1, 1),
    'scanner-xyz': (-1, -1, 1),
}
#: Kinds of axes that can be spatial
DOMAIN_KINDS = frozenset({'domain', 'space'})
#: Kinds of non-spatial axes that are kept before the component axes
LIST_KINDS = frozenset({'list', 'time', '???', 'none'})

#: numpy dtype for each NRRD type name
NRRD_TYPES: dict[str, str] = {
    **dict.fromkeys(('signed char', 'int8', 'int8_t'), 'i1'),
    **dict.fromkeys(('uchar', 'unsigned char', 'uint8', 'uint8_t'), 'u1'),
    **dict.fromkeys(
        ('short', 'short int', 'signed short', 'signed short int', 'int16', 'int16_t'), 'i2'
    ),
    **dict.fromkeys(
        ('ushort', 'unsigned short', 'unsigned short int', 'uint16', 'uint16_t'), 'u2'
    ),
    **dict.fromkeys(('int', 'signed int', 'int32', 'int32_t'), 'i4'),
    **dict.fromkeys(('uint', 'unsigned int', 'uint32', 'uint32_t'), 'u4'),
    **dict.fromkeys(
        (
            'longlong',
            'long long',
            'long long int',
            'signed long long',
            'signed long long int',
            'int64',
            'int64_t',
        ),
        'i8',
    ),
    **dict.fromkeys(
        ('ulonglong', 'unsigned long long', 'unsigned long long int', 'uint64', 'uint64_t'),
        'u8',
    ),
    'float': 'f4',
    'double': 'f8',
}

_REQUIRED_FIELDS = ('type', 'dimension', 'sizes', 'encoding')
_ENCODINGS: dict[str, ty.Literal['gzip', 'bzip2'] | None] = {
    'raw': None,
    'gzip': 'gzip',
    'gz': 'gzip',
    'bzip2': 'bzip2',
    'bz2': 'bzip2',
}
_UNSUPPORTED_ENCODINGS = frozenset({'ascii', 'text', 'txt', 'hex', 'zrl'})
_SPACE_ALIASES = {
    'lps': 'left-posterior-superior',
    'ras': 'right-anterior-superior',
    'las': 'left-anterior-superior',
    'lpst': 'left-posterior-superior-time',
    'rast': 'right-anterior-superior-time',
    'last': 'left-anterior-superior-time',
}
_SPACES_WITHOUT_ANATOMY = frozenset({'3d-right-handed'})
_UNSUPPORTED_SPACES = frozenset(
    {
        '3d-left-handed',
        'left-posterior-superior-time',
        'right-anterior-superior-time',
        'left-anterior-superior-time',
        'scanner-xyz-time',
        '3d-right-handed-time',
        '3d-left-handed-time',
    }
)
# Old spellings of field names
_FIELD_ALIASES = {
    'lineskip': 'line skip',
    'byteskip': 'byte skip',
    'datafile': 'data file',
    'axismins': 'axis mins',
    'axismaxs': 'axis maxs',
    'oldmin': 'old min',
    'oldmax': 'old max',
    'centers': 'centerings',
    'sampleunits': 'sample units',
}
# Fields with one value per axis
_PER_AXIS_FIELDS = (
    'sizes',
    'spacings',
    'thicknesses',
    'axis mins',
    'axis maxs',
    'centerings',
    'labels',
    'units',
    'kinds',
    'space directions',
)
_KEYVALUE_ESCAPES = re.compile(r'\\\\|\\n')
_MAX_MAGIC_LEN = 64
_PYNRRD_MSG = 'NRRD support requires pynrrd; install with `pip install nibabel[nrrd]`'


class NrrdError(Exception):
    """Error reading an NRRD file"""


class NrrdHeaderError(NrrdError, HeaderDataError):
    """NRRD header is malformed or uses a feature that is not supported"""


class NrrdImageError(NrrdError, ImageDataError):
    """NRRD data cannot be read"""


def _read_header_text(fileobj: io.IOBase) -> tuple[dict[str, str], dict[str, str]]:
    """Read the NRRD header from `fileobj` up to and including the blank line

    Parameters
    ----------
    fileobj : file-like
        Binary file positioned at the start of the NRRD magic line.

    Returns
    -------
    fields : dict
        Field values as text, keyed by canonical lower case field name, in file
        order.
    keyvalues : dict
        Key/value pairs, with ``\\n`` and ``\\\\`` escapes decoded.

    Notes
    -----
    `fileobj` is left at the first byte after the blank line that ends the
    header, or at the end of the file for a header with no data.
    """
    magic = fileobj.readline(_MAX_MAGIC_LEN).rstrip()
    if len(magic) != 8 or not magic.startswith(NRRD_MAGIC) or not magic[7:].isdigit():
        raise NrrdHeaderError(
            f'File does not start with the NRRD magic "NRRD000N"; it starts with {magic[:16]!r}'
        )
    version = int(magic[7:])
    if version not in SUPPORTED_VERSIONS:
        raise NrrdHeaderError(
            f'NRRD version {version} ({magic.decode()}) is not supported yet; '
            f'supported versions are {SUPPORTED_VERSIONS[0]} to {SUPPORTED_VERSIONS[-1]}'
        )
    fields: dict[str, str] = {}
    keyvalues: dict[str, str] = {}
    for raw_line in iter(fileobj.readline, b''):
        line = raw_line.rstrip()
        if not line:
            break
        if line.startswith(b'#'):
            continue
        kv_sep = line.find(b':=')
        field_sep = line.find(b': ')
        if kv_sep != -1 and (field_sep == -1 or kv_sep < field_sep):
            key = line[:kv_sep].decode('utf-8', errors='replace').strip()
            value = line[kv_sep + 2 :].decode('utf-8', errors='replace')
            keyvalues[key] = _KEYVALUE_ESCAPES.sub(
                lambda match: '\\' if match.group() == '\\\\' else '\n', value
            )
            continue
        try:
            text = line.decode('ascii')
        except UnicodeDecodeError as err:
            raise NrrdHeaderError(f'NRRD header line {line!r} is not ASCII') from err
        name, colon, value = text.partition(':')
        if not colon:
            raise NrrdHeaderError(f'NRRD header line {text!r} has no colon')
        name = ' '.join(name.lower().split())
        name = _FIELD_ALIASES.get(name, name)
        value = value.strip()
        if name in fields:
            raise NrrdHeaderError(f'NRRD field {name!r} appears more than once')
        if name == 'data file':
            _check_data_file(value)
        fields[name] = value
    return fields, keyvalues


def _check_data_file(value: str) -> None:
    """Raise for a 'data file' value that names more than one file

    These are checked while reading the header, because a detached header with a
    file list continues after the field with one file name per line.
    """
    words = value.split()
    if words[:1] == ['LIST'] or (len(words) >= 4 and '%' in words[0]):
        raise NrrdHeaderError(f'Several data files ("data file: {value}") are not supported yet')


def _parse_fields(text_fields: Mapping[str, str]) -> dict[str, ty.Any]:
    """Parse field values with pynrrd

    This is the only place that uses pynrrd. Each field is parsed on its own, so
    errors name the field, and pynrrd module settings are never changed.
    """
    pynrrd = optional_package('nrrd', trip_msg=_PYNRRD_MSG)[0]
    read_header = pynrrd.read_header
    parse_errors = (pynrrd.NRRDError, ValueError, TypeError, IndexError)
    fields: dict[str, ty.Any] = {}
    for name, value in text_fields.items():
        try:
            fields[name] = read_header(['NRRD0005', f'{name}: {value}'])[name]
        except parse_errors as err:
            raise NrrdHeaderError(f'Could not parse NRRD field "{name}: {value}": {err}') from err
    if 'space directions' in fields:
        fields['space directions'] = _direction_array(fields['space directions'])
    return fields


def _direction_array(directions: ty.Any) -> np.ndarray:
    """Return space directions as a float array with a row of NaN for each ``none``

    pynrrd returns a matrix with NaN rows or a list with ``None`` entries,
    depending on its version and settings.
    """
    if isinstance(directions, np.ndarray):
        return directions.astype(np.float64)
    width = max((len(row) for row in directions if row is not None), default=0)
    try:
        return np.array(
            [np.full(width, np.nan) if row is None else row for row in directions],
            dtype=np.float64,
        )
    except ValueError as err:
        raise NrrdHeaderError(f'NRRD space directions have different lengths: {err}') from err


def _canonical_space(space: str) -> str:
    """Return the full lower case name of NRRD space `space`"""
    name = space.strip().lower()
    return _SPACE_ALIASES.get(name, name)


def _data_dtype(fields: Mapping[str, ty.Any]) -> np.dtype:
    """Return the numpy dtype, with byte order, for NRRD `fields`"""
    nrrd_type = ' '.join(fields['type'].lower().split())
    if nrrd_type == 'block':
        raise NrrdHeaderError('NRRD type "block" is not supported yet')
    if nrrd_type not in NRRD_TYPES:
        raise NrrdHeaderError(f'Unknown NRRD type {fields["type"]!r}')
    dtype = np.dtype(NRRD_TYPES[nrrd_type])
    if dtype.itemsize == 1:
        return dtype
    if 'endian' not in fields:
        raise NrrdHeaderError(f"Required field 'endian' is missing for type {fields['type']!r}")
    endian = fields['endian'].strip().lower()
    if endian not in ('little', 'big'):
        raise NrrdHeaderError(f'Unknown NRRD endian {fields["endian"]!r}')
    return dtype.newbyteorder('<' if endian == 'little' else '>')


def _space_dimension(fields: Mapping[str, ty.Any]) -> int | None:
    """Return the dimension of the world space, or None if there is no space"""
    if 'space' in fields:
        return 3
    return fields.get('space dimension')


def _space_flip(fields: Mapping[str, ty.Any]) -> np.ndarray:
    """Return the matrix that maps vectors in the NRRD space to RAS+"""
    space = _canonical_space(fields['space']) if 'space' in fields else None
    if space in SPACE_TO_RAS:
        return np.diag(SPACE_TO_RAS[space]).astype(np.float64)
    return np.eye(_space_dimension(fields) or 3)


def _group_axes(fields: Mapping[str, ty.Any]) -> tuple[list[int], list[int], list[int]]:
    """Split the on-disk axes into spatial, list and component axes

    Raises
    ------
    NrrdHeaderError
        If kinds and space directions disagree.
    """
    ndim = fields['dimension']
    kinds = [kind.lower() for kind in fields.get('kinds', ['domain'] * ndim)]
    directions = fields.get('space directions')
    spatial: list[int] = []
    listlike: list[int] = []
    component: list[int] = []
    for axis, kind in enumerate(kinds):
        is_component = kind not in DOMAIN_KINDS | LIST_KINDS
        if directions is None:
            if kind in DOMAIN_KINDS or kind in ('???', 'none'):
                spatial.append(axis)
            else:
                (component if is_component else listlike).append(axis)
        elif not np.isnan(directions[axis]).all():
            if kind not in DOMAIN_KINDS and kind not in ('???', 'none'):
                raise NrrdHeaderError(
                    f'NRRD axis {axis} has kind {kind!r} but also a space direction'
                )
            spatial.append(axis)
        elif kind == 'space':
            raise NrrdHeaderError(f"NRRD axis {axis} has kind 'space' but no space direction")
        else:
            (component if is_component else listlike).append(axis)
    sdim = _space_dimension(fields)
    if directions is not None and len(spatial) != sdim:
        raise NrrdHeaderError(
            f'{len(spatial)} NRRD axes have space directions, but the space dimension is '
            f'{sdim}; images with fewer spatial axes than the space dimension are not '
            'supported yet'
        )
    return spatial, listlike, component


def _check_fields(fields: Mapping[str, ty.Any]) -> None:
    """Raise NrrdHeaderError if `fields` are not valid or not supported"""
    for name in _REQUIRED_FIELDS:
        if name not in fields:
            raise NrrdHeaderError(f'Required NRRD field {name!r} is missing')
    ndim = fields['dimension']
    if ndim < 1:
        raise NrrdHeaderError(f'NRRD dimension must be at least 1, not {ndim}')
    for name in _PER_AXIS_FIELDS:
        if name in fields and len(fields[name]) != ndim:
            raise NrrdHeaderError(
                f'NRRD field {name!r} has {len(fields[name])} values for {ndim} axes'
            )
    if any(size < 1 for size in fields['sizes']):
        raise NrrdHeaderError(f'NRRD sizes must be positive, not {list(fields["sizes"])}')
    _data_dtype(fields)
    encoding = fields['encoding'].strip().lower()
    if encoding in _UNSUPPORTED_ENCODINGS:
        raise NrrdHeaderError(f'NRRD encoding {fields["encoding"]!r} is not supported yet')
    if encoding not in _ENCODINGS:
        raise NrrdHeaderError(f'Unknown NRRD encoding {fields["encoding"]!r}')
    if fields.get('line skip', 0) < 0:
        raise NrrdHeaderError(f"NRRD field 'line skip' must be 0 or more: {fields['line skip']}")
    byte_skip = fields.get('byte skip', 0)
    if byte_skip < -1:
        raise NrrdHeaderError(f"NRRD field 'byte skip' must be -1 or more: {byte_skip}")
    if byte_skip == -1 and _ENCODINGS[encoding] is not None:
        raise NrrdHeaderError("NRRD field 'byte skip' can only be -1 for raw encoding")
    # Space
    if 'space' in fields and 'space dimension' in fields:
        raise NrrdHeaderError("NRRD fields 'space' and 'space dimension' cannot both be given")
    if 'space' in fields:
        space = _canonical_space(fields['space'])
        if space in _UNSUPPORTED_SPACES:
            raise NrrdHeaderError(f'NRRD space {fields["space"]!r} is not supported yet')
        if space not in SPACE_TO_RAS and space not in _SPACES_WITHOUT_ANATOMY:
            raise NrrdHeaderError(f'Unknown NRRD space {fields["space"]!r}')
    sdim = _space_dimension(fields)
    if sdim is None:
        for name in ('space directions', 'space origin', 'measurement frame'):
            if name in fields:
                raise NrrdHeaderError(
                    f"NRRD field {name!r} needs a 'space' or 'space dimension' field"
                )
    elif sdim not in (2, 3):
        raise NrrdHeaderError(f'NRRD space dimension {sdim} is not supported')
    else:
        directions = fields.get('space directions')
        if directions is not None:
            if directions.shape[1] != sdim:
                raise NrrdHeaderError(
                    f'NRRD space directions have length {directions.shape[1]}, '
                    f'but the space dimension is {sdim}'
                )
            given = ~np.isnan(directions).all(axis=1)
            if not np.isfinite(directions[given]).all():
                raise NrrdHeaderError(f'NRRD space directions are not finite: {directions}')
        origin = fields.get('space origin')
        if origin is not None and (len(origin) != sdim or not np.isfinite(origin).all()):
            raise NrrdHeaderError(
                f"NRRD field 'space origin' must be {sdim} finite values, not {origin}"
            )
        frame = fields.get('measurement frame')
        if frame is not None and np.shape(frame) != (sdim, sdim):
            raise NrrdHeaderError(
                f"NRRD field 'measurement frame' must be {sdim} vectors of length {sdim}"
            )
    _group_axes(fields)


def _skip_lines(fileobj: io.IOBase, n_lines: int, name: str) -> int:
    """Skip `n_lines` lines of `fileobj` and return the new position"""
    for _ in range(n_lines):
        if not fileobj.readline():
            raise NrrdImageError(f'{name} ended before {n_lines} lines were skipped')
    return fileobj.tell()


class NrrdHeader(SpatialHeader):
    """Header for NRRD images

    The data shape, zooms and affine use the nibabel axis order (see
    :meth:`get_axis_order`). The raw NRRD header is kept in on-disk axis order.

    Attributes
    ----------
    fields : dict
        NRRD header fields, keyed by lower case field name, as parsed by pynrrd.
        Vector values are numpy arrays. ``space directions`` is a float array
        with a row of NaN for each ``none`` axis.
    keyvalues : dict
        NRRD key/value pairs (``key:=value`` lines), as strings.
    """

    def __init__(
        self,
        data_dtype: npt.DTypeLike = np.float32,
        shape: ty.Sequence[int] = (0,),
        zooms: ty.Sequence[float] | None = None,
        *,
        fields: Mapping[str, ty.Any] | None = None,
        keyvalues: Mapping[str, str] | None = None,
    ):
        super().__init__(data_dtype, shape, zooms)
        self.fields: dict[str, ty.Any] = dict(fields or {})
        self.keyvalues: dict[str, str] = dict(keyvalues or {})

    @classmethod
    def from_fileobj(klass, fileobj: io.IOBase) -> NrrdHeader:
        """Read the NRRD header from `fileobj`

        Parameters
        ----------
        fileobj : file-like
            Binary file positioned at the start of the NRRD magic line. It is
            left positioned after the blank line that ends the header.

        Returns
        -------
        header : NrrdHeader

        Raises
        ------
        NrrdHeaderError
            If the header is malformed or uses a feature that is not supported.
        """
        text_fields, keyvalues = _read_header_text(fileobj)
        fields = _parse_fields(text_fields)
        _check_fields(fields)
        hdr = klass(_data_dtype(fields), fields=fields, keyvalues=keyvalues)
        order = sum(_group_axes(fields), [])
        hdr.set_data_shape([fields['sizes'][axis] for axis in order])
        hdr.set_zooms(hdr._zooms_from_fields())
        hdr._warn_no_anatomy()
        return hdr

    def copy(self) -> Self:
        """Return an independent copy of the header"""
        return self.__class__(
            self._dtype,
            self._shape,
            self._zooms,
            fields=deepcopy(self.fields),
            keyvalues=self.keyvalues,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NrrdHeader):
            return super().__eq__(other)
        if super().__eq__(other) is not True or self.keyvalues != other.keyvalues:
            return False
        if self.fields.keys() != other.fields.keys():
            return False
        return all(_values_equal(value, other.fields[name]) for name, value in self.fields.items())

    @classmethod
    def may_contain_header(klass, binaryblock: bytes) -> bool:
        """Return True if `binaryblock` starts with the NRRD magic"""
        return binaryblock[:7] == NRRD_MAGIC and binaryblock[7:8].isdigit()

    def _fields_match_shape(self) -> bool:
        """True if the NRRD fields describe the current data shape"""
        if 'sizes' not in self.fields or len(self.fields['sizes']) != len(self._shape):
            return False
        sizes = self.fields['sizes']
        return tuple(int(sizes[axis]) for axis in self.get_axis_order()) == self._shape

    def get_axis_order(self) -> tuple[int, ...]:
        """Return the on-disk axis for each axis of the data

        Spatial axes come first, then list axes (such as time), then component
        axes (such as vectors, colors or tensors). Each group keeps its on-disk
        order.

        Returns
        -------
        order : tuple of int
            ``order[i]`` is the on-disk axis of axis ``i`` of the image data.
        """
        if 'sizes' not in self.fields or len(self.fields['sizes']) != len(self._shape):
            return tuple(range(len(self._shape)))
        spatial, listlike, component = _group_axes(self.fields)
        return tuple(spatial + listlike + component)

    def get_space(self) -> str | None:
        """Return the full lower case name of the NRRD space, or None

        Short names are expanded, so ``RAS`` gives ``right-anterior-superior``.
        """
        if 'space' not in self.fields:
            return None
        return _canonical_space(self.fields['space'])

    def get_measurement_frame(self) -> np.ndarray | None:
        """Return the measurement frame in RAS+, or None if not given

        Column ``j`` is the RAS+ direction of the ``j``-th coordinate of vector
        and tensor values. The frame is never applied to the data.
        """
        frame = self.fields.get('measurement frame')
        if frame is None:
            return None
        return _space_flip(self.fields) @ np.asarray(frame, dtype=np.float64).T

    def get_best_affine(self) -> np.ndarray:
        """Return the affine from voxel indices to RAS+ millimeters

        Without NRRD fields that match the data shape, this falls back to the
        affine built from the zooms, as for other headers.
        """
        if not self._fields_match_shape():
            return super().get_best_affine()
        fields = self.fields
        spatial = _group_axes(fields)[0][:3]
        affine = np.eye(4)
        directions = fields.get('space directions')
        if directions is None:
            spacings = fields.get('spacings', np.full(fields['dimension'], np.nan))
            for column, axis in enumerate(spatial):
                spacing = spacings[axis]
                affine[column, column] = spacing if np.isfinite(spacing) else 1.0
            return affine
        sdim = directions.shape[1]
        flip = _space_flip(fields)
        affine[:sdim, : len(spatial)] = flip @ directions[spatial].T
        origin = fields.get('space origin')
        if origin is not None:
            affine[:sdim, 3] = flip @ np.asarray(origin, dtype=np.float64)
        return affine

    def _zooms_from_fields(self) -> list[float]:
        fields = self.fields
        spatial = _group_axes(fields)[0]
        directions = fields.get('space directions')
        spacings = fields.get('spacings')
        zooms = []
        for axis in self.get_axis_order():
            if directions is not None and axis in spatial:
                zooms.append(float(np.linalg.norm(directions[axis])))
            elif spacings is not None and np.isfinite(spacings[axis]):
                zooms.append(abs(float(spacings[axis])))
            else:
                zooms.append(1.0)
        return zooms

    def _warn_no_anatomy(self) -> None:
        if 'space directions' not in self.fields:
            msg = 'NRRD file has no space directions, so the affine comes from the spacings'
        elif self.get_space() in SPACE_TO_RAS:
            return
        elif 'space' in self.fields:
            msg = f'NRRD space {self.fields["space"]!r} has no anatomical orientation'
        else:
            msg = f'NRRD file has space dimension {self.fields["space dimension"]} and no space'
        warnings.warn(
            f'{msg}; the affine has no anatomical orientation and no axis flips',
            UserWarning,
            stacklevel=3,
        )


def _values_equal(first: ty.Any, second: ty.Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        first, second = np.asarray(first), np.asarray(second)
        if first.dtype.kind in 'fc' or second.dtype.kind in 'fc':
            return first.shape == second.shape and bool(
                np.array_equal(first, second, equal_nan=True)
            )
        return bool(np.array_equal(first, second))
    return bool(first == second)


class _DataWindow(io.RawIOBase):
    """Read-only view of a binary file that starts at byte `start`

    The underlying file is seeked before every read, so several windows can share
    one file object.
    """

    def __init__(self, fileobj: io.IOBase, start: int):
        self._fileobj = fileobj
        self._start = start
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = pos
        elif whence == io.SEEK_CUR:
            self._pos += pos
        elif whence == io.SEEK_END:
            self._pos = self._fileobj.seek(0, io.SEEK_END) - self._start + pos
        else:
            raise ValueError(f'Invalid whence {whence}')
        return self._pos

    def readinto(self, buffer: ty.Any) -> int:
        view = memoryview(buffer).cast('B')
        self._fileobj.seek(self._start + self._pos)
        data = self._fileobj.read(len(view))
        n_read = len(data)
        view[:n_read] = data
        self._pos += n_read
        return n_read


class _NrrdDataOpener(Opener):
    """Open NRRD data, decompressing according to the NRRD encoding

    Unlike :class:`~nibabel.openers.ImageOpener`, the compression comes from the
    NRRD ``encoding`` field, not from the file extension. Compressed data start
    at byte `data_start` of the file.
    """

    def __init__(self, fileish: str | io.IOBase, compression: str | None, data_start: int):
        if isinstance(fileish, str):
            self._base: io.IOBase = open(fileish, 'rb')
            self._base_is_mine = True
            self._name: str | None = fileish
        else:
            self._base = fileish
            self._base_is_mine = False
            self._name = getattr(fileish, 'name', None)
        # close_if_mine() calls close(), which closes only the objects made here
        self.me_opened = True
        self._stream: io.BufferedReader | None = None
        if compression is None:
            self.fobj = self._base
            return
        self._stream = io.BufferedReader(_DataWindow(self._base, data_start))
        if compression == 'gzip':
            self.fobj = gzip.GzipFile(fileobj=self._stream, mode='rb')
        else:
            self.fobj = bz2.BZ2File(self._stream, mode='rb')

    def close(self) -> None:
        if self._stream is not None:
            self.fobj.close()
            self._stream.close()
        if self._base_is_mine:
            self._base.close()

    @property
    def closed(self) -> bool:
        return self._base.closed if self._base_is_mine else self.fobj.closed


class NrrdArrayProxy(ArrayProxy):
    """Array proxy for NRRD data

    Adds to :class:`~nibabel.arrayproxy.ArrayProxy`:

    * gzip and bzip2 data that start part way through a file, decompressed only
      as far as needed;
    * axis reordering, so the proxy shape is in nibabel order while the data stay
      in on-disk order.
    """

    def __init__(
        self,
        file_like: str | io.IOBase,
        spec: tuple,
        *,
        mmap: bool | str = True,
        order: ty.Literal['F', 'C'] | None = None,
        keep_file_open: bool | None = None,
        compression: ty.Literal['gzip', 'bzip2'] | None = None,
        data_start: int = 0,
        axis_order: ty.Sequence[int] | None = None,
    ):
        """Initialize NRRD array proxy

        Parameters
        ----------
        file_like : str or file-like
            File name or open binary file holding the data.
        spec : tuple
            ``(shape, dtype, offset[, slope, inter])``, with `shape` in on-disk
            axis order. For compressed data, `offset` counts bytes of
            decompressed data.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            Memory map raw data if possible. See
            :class:`~nibabel.arrayproxy.ArrayProxy`.
        order : {None, 'F', 'C'}, optional, keyword only
            Data layout. NRRD data are in Fortran order, the default.
        keep_file_open : {None, True, False}, optional, keyword only
            Keep one file handle (and decompressor) open for the lifetime of
            the proxy. See :class:`~nibabel.arrayproxy.ArrayProxy`.
        compression : {None, 'gzip', 'bzip2'}, optional, keyword only
            Compression of the data.
        data_start : int, optional, keyword only
            Byte position in the file where compressed data start.
        axis_order : sequence of int, optional, keyword only
            On-disk axis for each axis of the proxy. Default is on-disk order.
        """
        super().__init__(file_like, spec, mmap=mmap, order=order, keep_file_open=keep_file_open)
        if compression not in (None, 'gzip', 'bzip2'):
            raise ValueError("compression should be one of {None, 'gzip', 'bzip2'}")
        self._compression = compression
        self._data_start = data_start
        ndim = len(self._shape)
        if axis_order is None:
            axis_order = range(ndim)
        self._axis_order = tuple(int(axis) for axis in axis_order)
        if sorted(self._axis_order) != list(range(ndim)):
            raise ValueError(f'axis_order {axis_order} is not a permutation of {ndim} axes')

    @property
    def shape(self):  # Untyped like ArrayProxy.shape, which overrides an attribute
        return tuple(self._shape[axis] for axis in self._axis_order)

    def copy(self) -> Self:
        """Create a new proxy for the same file and parameters

        If the proxied file is an open file handle, the new proxy shares a lock
        with this one.
        """
        new = self.__class__(
            self.file_like,
            (self._shape, self._dtype, self._offset, self._slope, self._inter),
            mmap=self._mmap,
            order=self.order,
            keep_file_open=self._keep_file_open,
            compression=self._compression,
            data_start=self._data_start,
            axis_order=self._axis_order,
        )
        if self._has_fh():
            new._lock = self._lock
        return new

    def reshape(self, shape: ty.Sequence[int]) -> ty.Any:
        """Return the data reshaped to `shape`

        Like :class:`ArrayProxy`, the reshape uses Fortran order. For data in
        on-disk axis order this is a new proxy. Otherwise the data are read and
        an array is returned.
        """
        if self._axis_order != tuple(range(len(self._shape))):
            return np.reshape(np.asanyarray(self), shape, order=self.order)
        proxy = super().reshape(shape)
        return self.__class__(
            self.file_like,
            (proxy._shape, self._dtype, self._offset, self._slope, self._inter),
            mmap=self._mmap,
            order=self.order,
            keep_file_open=self._keep_file_open,
            compression=self._compression,
            data_start=self._data_start,
        )

    def _should_keep_file_open(self, keep_file_open: bool | None) -> tuple[bool, bool]:
        # indexed_gzip is not used, so only keep the file open when asked
        keep_file_open, _ = super()._should_keep_file_open(keep_file_open)
        return keep_file_open, keep_file_open

    @contextmanager
    def _get_fileobj(self) -> Generator[_NrrdDataOpener, None, None]:
        """Yield an opener for the data, keeping it if `keep_file_open` was True"""
        if self._persist_opener:
            with self._lock:
                if not hasattr(self, '_opener'):
                    self._opener = _NrrdDataOpener(
                        self.file_like, self._compression, self._data_start
                    )
            yield self._opener
        else:
            with _NrrdDataOpener(self.file_like, self._compression, self._data_start) as opener:
                yield opener

    def _get_unscaled(self, slicer: ty.Any) -> np.ndarray:
        try:
            if self._axis_order == tuple(range(len(self._shape))):
                return super()._get_unscaled(slicer)
            return self._get_reordered(slicer)
        except (EOFError, OSError, zlib.error) as err:
            if self._compression is None:
                raise
            raise NrrdImageError(
                f'Could not read {self._compression} NRRD data from '
                f'{_name_of(self.file_like)}: {err}'
            ) from err

    def _get_reordered(self, slicer: ty.Any) -> np.ndarray:
        """Read data for `slicer` given in nibabel axis order"""
        canonical = canonical_slicers(slicer, self.shape)
        real = [item for item in canonical if item is not None]
        disk_slicer: list[ty.Any] = [None] * len(self._shape)
        for item, axis in zip(real, self._axis_order, strict=True):
            disk_slicer[axis] = item
        data = super()._get_unscaled(tuple(disk_slicer))
        # Axes left after integer indexing, in on-disk order and in nibabel order
        kept_disk = [axis for axis, item in enumerate(disk_slicer) if not isinstance(item, int)]
        kept_order = [
            axis
            for axis, item in zip(self._axis_order, real, strict=True)
            if not isinstance(item, int)
        ]
        data = data.transpose([kept_disk.index(axis) for axis in kept_order])
        if None in canonical:
            data = data[
                tuple(
                    None if item is None else slice(None)
                    for item in canonical
                    if not isinstance(item, int)
                )
            ]
        return data


class NrrdImage(SpatialImage):
    """Image in the NRRD format (read only)

    The data are read lazily through :class:`NrrdArrayProxy`. See the module
    documentation for the axis order and affine conventions.
    """

    header_class = NrrdHeader
    header: NrrdHeader
    _header: NrrdHeader
    valid_exts = ('.nrrd', '.nhdr')
    files_types = (('image', '.nrrd'),)
    _compressed_suffixes = ()
    _meta_sniff_len = 8
    makeable = False
    rw = False
    ImageArrayProxy = NrrdArrayProxy

    @classmethod
    def filespec_to_file_map(klass, filespec: FileSpec) -> FileMap:
        """Return a file map with the named ``.nrrd`` or ``.nhdr`` file as ``image``

        A detached data file is found from the header when the image is loaded.
        """
        filespec = _stringify_path(filespec)
        if os.path.splitext(filespec)[1].lower() not in klass.valid_exts:
            raise ImageFileError(
                f'Filespec "{filespec}" does not look right for class {klass}; '
                f'expected an extension in {klass.valid_exts}'
            )
        return {'image': FileHolder(filename=filespec)}

    @classmethod
    def _sniff_meta_for(
        klass,
        filename: FileSpec,
        sniff_nbytes: int,
        sniff: FileSniff | None = None,
    ) -> FileSniff | None:
        meta_fname = _stringify_path(filename)
        if sniff is not None and sniff[1] == meta_fname:
            return sniff
        try:
            with open(meta_fname, 'rb') as fobj:
                binaryblock = fobj.read(sniff_nbytes)
        except OSError:
            return None
        return (binaryblock, meta_fname)

    @classmethod
    def from_file_map(
        klass,
        file_map: FileMap,
        *,
        mmap: bool | ty.Literal['c', 'r'] = True,
        keep_file_open: bool | None = None,
    ) -> Self:
        """Create an image from the ``image`` entry of `file_map`

        Parameters
        ----------
        file_map : dict
            Mapping with an ``image`` file holder for the ``.nrrd`` or
            ``.nhdr`` file.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            Memory map raw data if possible. If False, do not memory map. If one
            of {'c', 'r'}, memory map with ``mode=mmap``. True is the same as
            ``'c'``. Compressed data are never memory mapped.
        keep_file_open : {None, True, False}, optional, keyword only
            If True, keep one file handle (and decompressor) open for the
            lifetime of the proxy. If False, open the file for each read. None
            uses ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT``.

        Returns
        -------
        img : NrrdImage

        Raises
        ------
        NrrdHeaderError
            If the header is malformed or uses a feature that is not supported.
        NrrdImageError
            If the data file cannot be found from the header, or holds too few
            bytes.
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        holder = file_map['image']
        with holder.get_prepare_fileobj('rb') as opener:
            # Opener has no readline, so read through the file it wraps
            fobj = opener.fobj
            header = klass.header_class.from_fileobj(fobj)
            fields = header.fields
            line_skip = fields.get('line skip', 0)
            data_like: str | io.IOBase
            if 'data file' in fields:
                data_like = data_path = klass._data_path(holder, fields['data file'])
                with open(data_path, 'rb') as data_fobj:
                    data_start = _skip_lines(data_fobj, line_skip, data_path)
                    file_size = data_fobj.seek(0, io.SEEK_END)
            else:
                data_like = ty.cast('str | io.IOBase', holder.file_like)
                data_start = _skip_lines(fobj, line_skip, holder.filename or 'NRRD file')
                file_size = fobj.seek(0, io.SEEK_END)
        compression = _ENCODINGS[fields['encoding'].strip().lower()]
        dtype = header.get_data_dtype()
        disk_shape = tuple(int(size) for size in fields['sizes'])
        byte_skip = fields.get('byte skip', 0)
        if compression is None:
            n_bytes = int(np.prod(disk_shape)) * dtype.itemsize
            offset = file_size - n_bytes if byte_skip == -1 else data_start + byte_skip
            if offset < data_start or file_size - offset < n_bytes:
                raise NrrdImageError(
                    f'NRRD data in {_name_of(data_like)} are too short: expected {n_bytes} '
                    f'bytes after byte {data_start}, with byte skip {byte_skip}, but the file '
                    f'has {file_size} bytes'
                )
        else:
            offset = byte_skip
        proxy = klass.ImageArrayProxy(
            data_like,
            (disk_shape, dtype, offset),
            mmap=mmap,
            keep_file_open=keep_file_open,
            compression=compression,
            data_start=data_start,
            axis_order=header.get_axis_order(),
        )
        return klass(proxy, header.get_best_affine(), header, file_map=file_map)

    @staticmethod
    def _data_path(holder: FileHolder, data_file: str) -> str:
        """Return the path of detached data file `data_file`, relative to the header"""
        if os.path.isabs(data_file):
            return data_file
        if holder.filename is None:
            raise NrrdImageError(
                f'NRRD data file {data_file!r} is a relative path, but the header has no '
                'file name to find it from'
            )
        return os.path.join(os.path.dirname(_stringify_path(holder.filename)), data_file)


def _name_of(file_like: str | io.IOBase) -> str:
    return file_like if isinstance(file_like, str) else 'file object'


load = NrrdImage.from_filename

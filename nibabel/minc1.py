# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read and write MINC1 format images

Writing goes through :func:`nibabel.minc1_write.write_minc1`.
"""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np

from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .minc1_write import write_minc1
from .minc_meta import MincMeta
from .spatialimages import SpatialHeader, SpatialImage

_dt_dict = {
    ('b', 'unsigned'): np.uint8,
    ('b', 'signed__'): np.int8,
    ('c', 'unsigned'): 'S1',
    ('h', 'unsigned'): np.uint16,
    ('h', 'signed__'): np.int16,
    ('i', 'unsigned'): np.uint32,
    ('i', 'signed__'): np.int32,
}

# See
# https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#MINC_specific_convenience_functions
_default_dir_cos = {'xspace': [1, 0, 0], 'yspace': [0, 1, 0], 'zspace': [0, 0, 1]}


class MincError(Exception):
    """Error when reading MINC files"""


class _NoVariable:
    """Stand-in for a dimension without variable, with no attributes"""

    data = np.zeros(())


class Minc1File:
    """Class to wrap MINC1 format opened netcdf object

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC file, to pull out useful header
    information, and for the method of reading the data out
    """

    def __init__(self, mincfile):
        self._mincfile = mincfile
        self._image = mincfile.variables['image']
        self._dim_names = self._image.dimensions
        # The code below will error with vector_dimensions.  See:
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#An_Introduction_to_NetCDF
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Image_dimensions
        # The vector dimension (such as RGB components) has no variable
        self._dims = [self._mincfile.variables.get(s, _NoVariable()) for s in self._dim_names]
        # Irregularly spaced dimensions have the coordinate of each sample
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#MINC_specific_convenience_functions
        self._coords = {}
        for name, dim in zip(self._dim_names, self._dims):
            if getattr(dim, 'spacing', b'regular__') == b'irregular' and dim.data.ndim == 1:
                self._coords[name] = np.array(dim.data, dtype=np.float64)
        self._spatial_dims = [name for name in self._dim_names if name.endswith('space')]
        self._check_irregular()
        # the MINC standard appears to allow the following variables to
        # be undefined.
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Image_conversion_variables
        # It wasn't immediately obvious what the defaults were.
        self._image_max = self._mincfile.variables['image-max']
        self._image_min = self._mincfile.variables['image-min']

    def _get_dimensions(self, var):
        # Dimensions for a particular variable
        # Differs for MINC1 and MINC2 - see:
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Associating_HDF5_dataspaces_with_MINC_dimensions
        return var.dimensions

    def get_data_dtype(self):
        typecode = self._image.typecode()
        if typecode == 'f':
            dtt = np.dtype(np.float32)
        elif typecode == 'd':
            dtt = np.dtype(np.float64)
        else:
            signtype = self._image.signtype.decode('latin-1')
            dtt = _dt_dict[(typecode, signtype)]
        return np.dtype(dtt).newbyteorder('>')

    def get_data_shape(self):
        return self._image.data.shape

    def _start_step(self, name, dim):
        """Start and step of dimension `name`, with variable `dim`

        For irregularly spaced dimensions without ``start`` or ``step``
        attributes, these are the first coordinate and the mean spacing.
        """
        coords = self._coords.get(name)
        if hasattr(dim, 'start'):
            start = float(dim.start)
        else:
            start = float(coords[0]) if coords is not None and coords.size else 0.0
        if hasattr(dim, 'step'):
            step = float(dim.step)
        elif coords is not None and coords.size > 1:
            step = float(coords[-1] - coords[0]) / (coords.size - 1)
        else:
            step = 1.0
        return start, step

    def _check_irregular(self):
        """Warn for spatial dimensions that the affine cannot describe"""
        for name in self._spatial_dims:
            coords = self._coords.get(name)
            if coords is None:
                continue
            dim = self._dims[list(self._dim_names).index(name)]
            start, step = self._start_step(name, dim)
            regular = start + step * np.arange(coords.size)
            if not np.allclose(coords, regular, rtol=1e-6, atol=1e-6 * max(abs(step), 1)):
                warnings.warn(
                    f'{name} is irregularly spaced; the affine uses its start and mean '
                    'step, and minc_meta has the coordinates'
                )

    def get_zooms(self):
        """Get real-world sizes of voxels"""
        # zooms must be positive; but steps in MINC can be negative
        return tuple(
            abs(self._start_step(name, dim)[1]) for name, dim in zip(self._dim_names, self._dims)
        )

    def get_affine(self):
        nspatial = len(self._spatial_dims)
        rot_mat = np.eye(nspatial)
        steps = np.zeros((nspatial,))
        starts = np.zeros((nspatial,))
        dim_names = list(self._dim_names)  # for indexing in loop
        for i, name in enumerate(self._spatial_dims):
            dim = self._dims[dim_names.index(name)]
            rot_mat[:, i] = (
                dim.direction_cosines
                if hasattr(dim, 'direction_cosines')
                else _default_dir_cos[name]
            )
            starts[i], steps[i] = self._start_step(name, dim)
        origin = np.dot(rot_mat, starts)
        aff = np.eye(nspatial + 1)
        aff[:nspatial, :nspatial] = rot_mat * steps
        aff[:nspatial, nspatial] = origin
        return aff

    def _get_valid_range(self):
        """Return valid range for image data

        The valid range can come from the image 'valid_range' or
        image 'valid_min' and 'valid_max', or, failing that, from the
        data type range
        """
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.valid_range
        except AttributeError:
            try:
                valid_range = [self._image.valid_min, self._image.valid_max]
            except AttributeError:
                valid_range = [info.min, info.max]
        if valid_range[0] < info.min or valid_range[1] > info.max:
            raise ValueError('Valid range outside input data type range')
        return np.asarray(valid_range, dtype=np.float64)

    def _get_scalar(self, var):
        """Get scalar value from NetCDF scalar"""
        return var.getValue()

    def _get_array(self, var):
        """Get array from NetCDF array"""
        return var.data

    def _normalize(self, data, sliceobj=()):
        """Apply scaling to image data `data` already sliced with `sliceobj`

        https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Pixel_values_and_real_values

        MINC normalization uses "image-min" and "image-max" variables to
        map the data from the valid range of the image to the range
        specified by "image-min" and "image-max".

        The "image-max" and "image-min" are variables that describe the
        "max" and "min" of image over some dimensions of "image".

        The usual case is that "image" has dimensions ["zspace", "yspace",
        "xspace"] and "image-max" has dimensions ["zspace"], but there can be
        up to two dimensions for over which scaling is specified.

        Parameters
        ----------
        data : ndarray
            data after applying `sliceobj` slicing to full image
        sliceobj : tuple, optional
            slice definition. If not specified, assume no slicing has been
            applied to `data`
        """
        ddt = self.get_data_dtype()
        if ddt.kind in 'fc':
            return data
        image_max = self._image_max
        image_min = self._image_min
        mx_dims = self._get_dimensions(image_max)
        mn_dims = self._get_dimensions(image_min)
        if mx_dims != mn_dims:
            raise MincError('"image-max" and "image-min" do not have the same dimensions')
        nscales = len(mx_dims)
        if nscales > 2:
            raise MincError('More than two scaling dimensions')
        if mx_dims != self._dim_names[:nscales]:
            raise MincError('image-max and image dimensions do not match')
        dmin, dmax = self._get_valid_range()
        out_data = np.clip(data, dmin, dmax)
        if nscales == 0:  # scalar values
            imax = self._get_scalar(image_max)
            imin = self._get_scalar(image_min)
        else:  # 1D or 2D array of scaling values
            # We need to get the correct values from image-max and image-min to
            # do the scaling.
            shape = self.get_data_shape()
            sliceobj = canonical_slicers(sliceobj, shape)
            # Indices into sliceobj referring to image axes
            ax_inds = [i for i, obj in enumerate(sliceobj) if obj is not None]
            assert len(ax_inds) == len(shape)
            # Slice imax, imin using same slicer as for data
            nscales_ax = ax_inds[nscales]
            i_slicer = sliceobj[:nscales_ax]
            # Fill slicer to broadcast against sliced data; add length 1 axis
            # for each axis except int axes (which are dropped by slicing)
            broad_part = tuple(
                None for s in sliceobj[ax_inds[nscales] :] if not isinstance(s, Integral)
            )
            i_slicer += broad_part
            imax = self._get_array(image_max)[i_slicer]
            imin = self._get_array(image_min)[i_slicer]
        slope = (imax - imin) / (dmax - dmin)
        inter = imin - dmin * slope
        out_data *= slope
        out_data += inter
        return out_data

    def get_scaled_data(self, sliceobj=()):
        """Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        """
        if sliceobj == ():
            raw_data = self._image.data
        else:
            raw_data = self._image.data[sliceobj]
        dtype = self.get_data_dtype()
        data = np.asarray(raw_data).view(dtype)
        return self._normalize(data, sliceobj)


class MincImageArrayProxy:
    """MINC implementation of array proxy protocol

    The array proxy allows us to freeze the passed fileobj and
    header such that it returns the expected data array.
    """

    def __init__(self, minc_file):
        self.minc_file = minc_file
        self._shape = minc_file.get_data_shape()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_proxy(self):
        return True

    def __array__(self, dtype=None):
        """Read data from file and apply scaling, casting to ``dtype``

        If ``dtype`` is unspecified, the dtype is automatically determined.

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        arr = self.minc_file.get_scaled_data(sliceobj=())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, sliceobj):
        """Read slice `sliceobj` of data from file"""
        return self.minc_file.get_scaled_data(sliceobj)


class MincHeader(SpatialHeader):
    """Class to contain header for MINC formats"""

    # We don't use the data layout - this just in case we do later
    data_layout = 'C'

    def data_to_fileobj(self, data, fileobj, rescale=True):
        """See Header class for an implementation we can't use"""
        raise NotImplementedError

    def data_from_fileobj(self, fileobj):
        """See Header class for an implementation we can't use"""
        raise NotImplementedError


class Minc1Header(MincHeader):
    @classmethod
    def may_contain_header(klass, binaryblock):
        return binaryblock[:4] == b'CDF\x01'


class Minc1Image(SpatialImage):
    """Class for MINC1 format images

    The MINC1 image class uses the default header type, rather than a specific
    MINC header type - and reads the relevant information from the MINC file on
    load.
    """

    header_class: type[MincHeader] = Minc1Header
    header: MincHeader
    _meta_sniff_len: int = 4
    valid_exts: tuple[str, ...] = ('.mnc',)
    files_types: tuple[tuple[str, str], ...] = (('image', '.mnc'),)
    _compressed_suffixes: tuple[str, ...] = ('.gz', '.bz2', '.zst')

    makeable = True
    rw = True

    ImageArrayProxy = MincImageArrayProxy

    #: :class:`~nibabel.minc_meta.MincMeta` snapshot of the metadata of the
    #: file the image was loaded from, or None
    minc_meta: MincMeta | None = None

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        # Note that mmap and keep_file_open are included for proper
        with file_map['image'].get_prepare_fileobj() as fobj:
            ncfile = netcdf_file(fobj)
            minc_file = Minc1File(ncfile)
            minc_meta = MincMeta.from_netcdf(ncfile)
            affine = minc_file.get_affine()
            if affine.shape != (4, 4):
                raise MincError('Image does not have 3 spatial dimensions')
            data_dtype = minc_file.get_data_dtype()
            shape = minc_file.get_data_shape()
            zooms = minc_file.get_zooms()
            header = klass.header_class(data_dtype, shape, zooms)
            data = klass.ImageArrayProxy(minc_file)
        img = klass(data, affine, header, extra=None, file_map=file_map)
        img.minc_meta = minc_meta
        return img

    @classmethod
    def from_image(klass, img):
        """Class instance from `img`, with the metadata of MINC images

        Converting between :class:`Minc1Image` and
        :class:`~nibabel.minc2.Minc2Image` carries the metadata snapshot,
        ``img.minc_meta``, across, so that saving the converted image keeps
        the metadata in the other format.

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``dataobj``, ``affine``, ``header`` and ``extra``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        """
        new = super().from_image(img)
        new.minc_meta = getattr(img, 'minc_meta', None)
        return new

    @staticmethod
    def _write(*args, **kwargs):
        return write_minc1(*args, **kwargs)

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        A :class:`Minc1Image` writes MINC1 and a
        :class:`~nibabel.minc2.Minc2Image` writes MINC2, with
        :func:`~nibabel.minc1_write.write_minc1` and
        :func:`~nibabel.minc2_write.write_minc2`.

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        dtype : dtype-like, optional
           The on-disk data type to coerce the data array.  Default is the
           header data type.

        Notes
        -----
        The data array axes are the MINC dimensions, slowest varying first;
        a 4D array has time on axis 0, as the reader returns it.  If the
        data are still the unmodified proxy of a MINC file, and `dtype` is
        that file's integer type, the file's image-min / image-max scaling
        is kept, so that the stored values are written back unchanged.  See
        :mod:`nibabel.minc2_write` for the scaling chosen otherwise.

        If the image has a metadata snapshot, ``self.minc_meta``, it is
        passed to the writer, which copies the info variables and the
        global, image and dimension attributes, and appends to the history.
        If the data shape or the affine changed since the image was read,
        so that a dimension of the snapshot no longer matches, only the
        geometry of that dimension is dropped from the snapshot: its exact
        start, step and direction cosines, and the coordinates and widths
        of irregular samples.  The dimension is then written regularly
        spaced, with the geometry of the data and the affine, and keeps its
        other attributes.
        """
        if file_map is None:
            file_map = self.file_map
        out_dtype = np.dtype(self.get_data_dtype() if dtype is None else dtype)
        scaling = _proxy_scaling(self.dataobj, out_dtype)
        data = np.asanyarray(self.dataobj)
        holder = file_map['image']
        target = holder.fileobj if holder.fileobj is not None else holder.filename
        self._write(
            data,
            self.affine,
            target,
            dtype=out_dtype.newbyteorder('='),
            scaling=scaling,
            metadata=self.minc_meta,
        )
        self.file_map = file_map


def _proxy_scaling(dataobj, out_dtype):
    """Scaling of the MINC file behind `dataobj`, if any

    Returns
    -------
    scaling : None or tuple
        ``(image_min, image_max, valid_range)`` of the source file, if
        `dataobj` is a MINC array proxy and `out_dtype` is its integer
        storage type; None otherwise.
    """
    if not isinstance(dataobj, MincImageArrayProxy):
        return None
    minc_file = dataobj.minc_file
    in_dtype = minc_file.get_data_dtype()
    if out_dtype.kind in 'fc' or in_dtype.newbyteorder('=') != out_dtype.newbyteorder('='):
        return None
    image_min = minc_file._image_min
    image_max = minc_file._image_max
    if minc_file._get_dimensions(image_min):
        image_min = np.array(minc_file._get_array(image_min), dtype=np.float64)
        image_max = np.array(minc_file._get_array(image_max), dtype=np.float64)
    else:
        image_min = float(minc_file._get_scalar(image_min))
        image_max = float(minc_file._get_scalar(image_max))
    valid_range = tuple(minc_file._get_valid_range())
    return image_min, image_max, valid_range


load = Minc1Image.from_filename

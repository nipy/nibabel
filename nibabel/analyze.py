# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Header and image for the basic Mayo Analyze format

===========================
 The Analyze header format
===========================

This is a binary header format and inherits from ``WrapStruct``

Apart from the attributes and methods of WrapStruct:

Class attributes are::

    .default_x_flip

with methods::

    .get/set_data_shape
    .get/set_data_dtype
    .get/set_zooms
    .get_base_affine()
    .get_best_affine()
    .data_to_fileobj
    .data_from_fileobj

and class methods::

    .from_header(hdr)

More sophisticated headers can add more methods and attributes.

Notes
-----

This - basic - analyze header cannot encode full affines (only
diagonal affines), and cannot do integer scaling.

The inability to store affines means that we have to guess what orientation the
image has.  Most Analyze images are stored on disk in (fastest-changing to
slowest-changing) R->L, P->A and I->S order.  That is, the first voxel is the
rightmost, most posterior and most inferior voxel location in the image, and the
next voxel is one voxel towards the left of the image.

Most people refer to this disk storage format as 'radiological', on the basis
that, if you load up the data as an array ``img_arr`` where the first axis is
the fastest changing, then take a slice in the I->S axis - ``img_arr[:,:,10]`` -
then the right part of the brain will be on the left of your displayed slice.
Radiologists like looking at images where the left of the brain is on the right
side of the image.

Conversely, if the image has the voxels stored with the left voxels first -
L->R, P->A, I->S, then this would be 'neurological' format.  Neurologists like
looking at images where the left side of the brain is on the left of the image.

When we are guessing at an affine for Analyze, this translates to the problem of
whether the affine should consider proceeding within the data down an X line as
being from left to right, or right to left.

By default we assume that the image is stored in R->L format.  We encode this
choice in the ``default_x_flip`` flag that can be True or False.  True means
assume radiological.

If the image is 3D, and the X, Y and Z zooms are x, y, and z, then::

    if default_x_flip is True::
        affine = np.diag((-x,y,z,1))
    else:
        affine = np.diag((x,y,z,1))

In our implementation, there is no way of saving this assumed flip into the
header.  One way of doing this, that we have not used, is to allow negative
zooms, in particular, negative X zooms.  We did not do this because the image
can be loaded with and without a default flip, so the saved zoom will not
constrain the affine.
'''
import sys

import numpy as np

from .volumeutils import (native_code, swapped_code, make_dt_codes, allopen,
                          shape_zoom_affine, array_from_file, seek_tell,
                          apply_read_scaling)
from .arraywriters import make_array_writer, get_slope_inter, WriterError
from .wrapstruct import WrapStruct
from .spatialimages import (HeaderDataError, HeaderTypeError,
                            SpatialImage)
from .fileholders import copy_file_map
from .batteryrunners import Report
from .arrayproxy import ArrayProxy

# Sub-parts of standard analyze header from
# Mayo dbh.h file
header_key_dtd = [
    ('sizeof_hdr', 'i4'),
    ('data_type', 'S10'),
    ('db_name', 'S18'),
    ('extents', 'i4'),
    ('session_error', 'i2'),
    ('regular', 'S1'),
    ('hkey_un0', 'S1')
    ]
image_dimension_dtd = [
    ('dim', 'i2', (8,)),
    ('vox_units', 'S4'),
    ('cal_units', 'S8'),
    ('unused1', 'i2'),
    ('datatype', 'i2'),
    ('bitpix', 'i2'),
    ('dim_un0', 'i2'),
    ('pixdim', 'f4', (8,)),
    ('vox_offset', 'f4'),
    ('funused1', 'f4'),
    ('funused2', 'f4'),
    ('funused3', 'f4'),
    ('cal_max', 'f4'),
    ('cal_min', 'f4'),
    ('compressed', 'i4'),
    ('verified', 'i4'),
    ('glmax', 'i4'),
    ('glmin', 'i4')
    ]
data_history_dtd = [
    ('descrip', 'S80'),
    ('aux_file', 'S24'),
    ('orient', 'S1'),
    ('originator', 'S10'),
    ('generated', 'S10'),
    ('scannum', 'S10'),
    ('patient_id', 'S10'),
    ('exp_date', 'S10'),
    ('exp_time', 'S10'),
    ('hist_un0', 'S3'),
    ('views', 'i4'),
    ('vols_added', 'i4'),
    ('start_field', 'i4'),
    ('field_skip', 'i4'),
    ('omax', 'i4'),
    ('omin', 'i4'),
    ('smax', 'i4'),
    ('smin', 'i4')
    ]

# Full header numpy dtype combined across sub-fields
header_dtype = np.dtype(header_key_dtd + image_dimension_dtd +
                        data_history_dtd)

_dtdefs = ( # code, conversion function, equivalent dtype, aliases
    (0, 'none', np.void),
    (1, 'binary', np.void), # 1 bit per voxel, needs thought
    (2, 'uint8', np.uint8),
    (4, 'int16', np.int16),
    (8, 'int32', np.int32),
    (16, 'float32', np.float32),
    (32, 'complex64', np.complex64), # numpy complex format?
    (64, 'float64', np.float64),
    (128, 'RGB', np.dtype([('R','u1'),
                  ('G', 'u1'),
                  ('B', 'u1')])),
    (255, 'all', np.void))

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)


class AnalyzeHeader(WrapStruct):
    ''' Class for basic analyze header

    Implements zoom-only setting of affine transform, and no image
    scaling
    '''
    # Copies of module-level definitions
    template_dtype = header_dtype
    _data_type_codes = data_type_codes
    # fields with recoders for their values
    _field_recoders = {'datatype': data_type_codes}
    # default x flip
    default_x_flip = True

    # data scaling capabilities
    has_data_slope = False
    has_data_intercept = False

    def __init__(self,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        ''' Initialize header from binary data block

        Parameters
        ----------
        binaryblock : {None, string} optional
            binary block to set into header.  By default, None, in
            which case we insert the default empty header block
        endianness : {None, '<','>', other endian code} string, optional
            endianness of the binaryblock.  If None, guess endianness
            from the data.
        check : bool, optional
            Whether to check content of header in initialization.
            Default is True.

        Examples
        --------
        >>> hdr1 = AnalyzeHeader() # an empty header
        >>> hdr1.endianness == native_code
        True
        >>> hdr1.get_data_shape()
        (0,)
        >>> hdr1.set_data_shape((1,2,3)) # now with some content
        >>> hdr1.get_data_shape()
        (1, 2, 3)

        We can set the binary block directly via this initialization.
        Here we get it from the header we have just made

        >>> binblock2 = hdr1.binaryblock
        >>> hdr2 = AnalyzeHeader(binblock2)
        >>> hdr2.get_data_shape()
        (1, 2, 3)

        Empty headers are native endian by default

        >>> hdr2.endianness == native_code
        True

        You can pass valid opposite endian headers with the
        ``endianness`` parameter. Even empty headers can have
        endianness

        >>> hdr3 = AnalyzeHeader(endianness=swapped_code)
        >>> hdr3.endianness == swapped_code
        True

        If you do not pass an endianness, and you pass some data, we
        will try to guess from the passed data.

        >>> binblock3 = hdr3.binaryblock
        >>> hdr4 = AnalyzeHeader(binblock3)
        >>> hdr4.endianness == swapped_code
        True
        '''
        super(AnalyzeHeader, self).__init__(binaryblock, endianness, check)

    @classmethod
    def guessed_endian(klass, hdr):
        ''' Guess intended endianness from mapping-like ``hdr``

        Parameters
        ----------
        hdr : mapping-like
           hdr for which to guess endianness

        Returns
        -------
        endianness : {'<', '>'}
           Guessed endianness of header

        Examples
        --------
        Zeros header, no information, guess native

        >>> hdr = AnalyzeHeader()
        >>> hdr_data = np.zeros((), dtype=header_dtype)
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True

        A valid native header is guessed native

        >>> hdr_data = hdr.structarr.copy()
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True

        And, when swapped, is guessed as swapped

        >>> sw_hdr_data = hdr_data.byteswap(swapped_code)
        >>> AnalyzeHeader.guessed_endian(sw_hdr_data) == swapped_code
        True

        The algorithm is as follows:

        First, look at the first value in the ``dim`` field; this
        should be between 0 and 7.  If it is between 1 and 7, then
        this must be a native endian header.

        >>> hdr_data = np.zeros((), dtype=header_dtype) # blank binary data
        >>> hdr_data['dim'][0] = 1
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True
        >>> hdr_data['dim'][0] = 6
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True
        >>> hdr_data['dim'][0] = -1
        >>> AnalyzeHeader.guessed_endian(hdr_data) == swapped_code
        True

        If the first ``dim`` value is zeros, we need a tie breaker.
        In that case we check the ``sizeof_hdr`` field.  This should
        be 348.  If it looks like the byteswapped value of 348,
        assumed swapped.  Otherwise assume native.

        >>> hdr_data = np.zeros((), dtype=header_dtype) # blank binary data
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True
        >>> hdr_data['sizeof_hdr'] = 1543569408
        >>> AnalyzeHeader.guessed_endian(hdr_data) == swapped_code
        True
        >>> hdr_data['sizeof_hdr'] = -1
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True

        This is overridden by the ``dim``[0] value though:

        >>> hdr_data['sizeof_hdr'] = 1543569408
        >>> hdr_data['dim'][0] = 1
        >>> AnalyzeHeader.guessed_endian(hdr_data) == native_code
        True
        '''
        dim0 = int(hdr['dim'][0])
        if dim0 == 0:
            if hdr['sizeof_hdr'] == 1543569408:
                return swapped_code
            return native_code
        elif 1 <= dim0 <= 7:
            return native_code
        return swapped_code

    @classmethod
    def default_structarr(klass, endianness=None):
        ''' Return header data for empty header with given endianness
        '''
        hdr_data = super(AnalyzeHeader, klass).default_structarr(endianness)
        hdr_data['sizeof_hdr'] = 348
        hdr_data['dim'] = 1
        hdr_data['dim'][0] = 0
        hdr_data['pixdim'] = 1
        hdr_data['datatype'] = 16 # float32
        hdr_data['bitpix'] = 32
        return hdr_data

    def get_value_label(self, fieldname):
        ''' Returns label for coded field

        A coded field is an int field containing codes that stand for
        discrete values that also have string labels.

        Parameters
        ----------
        fieldname : str
           name of header field to get label for

        Returns
        -------
        label : str
           label for code value in header field `fieldname`

        Raises
        ------
        ValueError : if field is not coded

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.get_value_label('datatype')
        'float32'
        '''
        if not fieldname in self._field_recoders:
            raise ValueError('%s not a coded field' % fieldname)
        code = int(self._structarr[fieldname])
        return self._field_recoders[fieldname].label[code]

    @classmethod
    def from_header(klass, header=None, check=True):
        ''' Class method to create header from another header

        Parameters
        ----------
        header : ``Header`` instance or mapping
           a header of this class, or another class of header for
           conversion to this type
        check : {True, False}
           whether to check header for integrity

        Returns
        -------
        hdr : header instance
           fresh header instance of our own class
        '''
        # own type, return copy
        if type(header) == klass:
            obj = header.copy()
            if check:
                obj.check_fix()
            return obj
        # not own type, make fresh header instance
        obj = klass(check=check)
        if header is None:
            return obj
        try: # check if there is a specific conversion routine
            mapping = header.as_analyze_map()
        except AttributeError:
            # most basic conversion
            obj.set_data_dtype(header.get_data_dtype())
            obj.set_data_shape(header.get_data_shape())
            obj.set_zooms(header.get_zooms())
            return obj
        # header is convertible from a field mapping
        for key, value in mapping.items():
            try:
                obj[key] = value
            except (ValueError, KeyError):
                # the presence of the mapping certifies the fields as
                # being of the same meaning as for Analyze types
                pass
        # set any fields etc that are specific to this format (overriden by
        # sub-classes)
        obj._set_format_specifics()
        # Check for unsupported datatypes
        orig_code = header.get_data_dtype()
        try:
            obj.set_data_dtype(orig_code)
        except HeaderDataError:
            raise HeaderDataError('Input header %s has datatype %s but '
                                  'output header %s does not support it'
                                  % (header.__class__,
                                     header.get_value_label('datatype'),
                                     klass))
        if check:
            obj.check_fix()
        return obj

    def _set_format_specifics(self):
        ''' Utility routine to set format specific header stuff
        '''
        pass

    def raw_data_from_fileobj(self, fileobj):
        ''' Read unscaled data array from `fileobj`

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           unscaled data array
        '''
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset)

    def data_from_fileobj(self, fileobj):
        ''' Read scaled data array from `fileobj`

        Use this routine to get the scaled image data from an image file
        `fileobj`, given a header `self`.  "Scaled" means, with any header
        scaling factors applied to the raw data in the file.  Use
        `raw_data_from_fileobj` to get the raw data.

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           scaled data array

        Notes
        -----
        We use the header to get any scale or intercept values to apply to the
        data.  Raw Analyze files don't have scale factors or intercepts, but
        this routine also works with formats based on Analyze, that do have
        scaling, such as SPM analyze formats and NIfTI.
        '''
        # read unscaled data
        data = self.raw_data_from_fileobj(fileobj)
        # get scalings from header.  Value of None means not present in header
        slope, inter = self.get_slope_inter()
        slope = 1.0 if slope is None else slope
        inter = 0.0 if inter is None else inter
        # Upcast as necessary for big slopes, intercepts
        return apply_read_scaling(data, slope, inter)

    def data_to_fileobj(self, data, fileobj):
        ''' Write `data` to `fileobj`, maybe modifying `self`

        In writing the data, we match the header to the written data, by
        setting the header scaling factors.  Thus we modify `self` in
        the process of writing the data.

        Parameters
        ----------
        data : array-like
           data to write; should match header defined shape
        fileobj : file-like object
           Object with file interface, implementing ``write`` and
           ``seek``

        Examples
        --------
        >>> from nibabel.analyze import AnalyzeHeader
        >>> hdr = AnalyzeHeader()
        >>> hdr.set_data_shape((1, 2, 3))
        >>> hdr.set_data_dtype(np.float64)
        >>> from StringIO import StringIO #23dt : BytesIO
        >>> str_io = StringIO() #23dt : BytesIO
        >>> data = np.arange(6).reshape(1,2,3)
        >>> hdr.data_to_fileobj(data, str_io)
        >>> data.astype(np.float64).tostring('F') == str_io.getvalue()
        True
        '''
        data = np.asanyarray(data)
        shape = self.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        out_dtype = self.get_data_dtype()
        try:
            arr_writer = make_array_writer(data,
                                           out_dtype,
                                           self.has_data_slope,
                                           self.has_data_intercept)
        except WriterError:
            msg = sys.exc_info()[1] # python 2 / 3 compatibility
            raise HeaderTypeError(msg)
        seek_tell(fileobj, self.get_data_offset())
        arr_writer.to_fileobj(fileobj)
        self.set_slope_inter(*get_slope_inter(arr_writer))

    def get_data_dtype(self):
        ''' Get numpy dtype for data

        For examples see ``set_data_dtype``
        '''
        code = int(self._structarr['datatype'])
        dtype = self._data_type_codes.dtype[code]
        return dtype.newbyteorder(self.endianness)

    def set_data_dtype(self, datatype):
        ''' Set numpy dtype for data from code or dtype or type

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.set_data_dtype(np.uint8)
        >>> hdr.get_data_dtype()
        dtype('uint8')
        >>> hdr.set_data_dtype(np.dtype(np.uint8))
        >>> hdr.get_data_dtype()
        dtype('uint8')
        >>> hdr.set_data_dtype('implausible') #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
           ...
        HeaderDataError: data dtype "implausible" not recognized
        >>> hdr.set_data_dtype('none') #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
           ...
        HeaderDataError: data dtype "none" known but not supported
        >>> hdr.set_data_dtype(np.void) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
           ...
        HeaderDataError: data dtype "<type 'numpy.void'>" known but not supported
        '''
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise HeaderDataError(
                'data dtype "%s" not recognized' % datatype)
        dtype = self._data_type_codes.dtype[code]
        # test for void, being careful of user-defined types
        if dtype.type is np.void and not dtype.fields:
            raise HeaderDataError(
                'data dtype "%s" known but not supported' % datatype)
        self._structarr['datatype'] = code
        self._structarr['bitpix'] = dtype.itemsize * 8

    def get_data_shape(self):
        ''' Get shape of data

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.get_data_shape()
        (0,)
        >>> hdr.set_data_shape((1,2,3))
        >>> hdr.get_data_shape()
        (1, 2, 3)

        Expanding number of dimensions gets default zooms

        >>> hdr.get_zooms()
        (1.0, 1.0, 1.0)
        '''
        dims = self._structarr['dim']
        ndims = dims[0]
        if ndims == 0:
            return 0,
        return tuple(int(d) for d in dims[1:ndims+1])

    def set_data_shape(self, shape):
        ''' Set shape of data

        If ``ndims == len(shape)`` then we set zooms for dimensions higher than
        ``ndims`` to 1.0

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        '''
        dims = self._structarr['dim']
        ndims = len(shape)
        dims[:] = 1
        dims[0] = ndims
        dims[1:ndims+1] = shape
        # Check that dimensions fit
        if not np.all(dims[1:ndims+1] == shape):
            raise HeaderDataError('shape %s does not fit in dim datatype' %
                                   (shape,))
        self._structarr['pixdim'][ndims+1:] = 1.0

    def get_base_affine(self):
        ''' Get affine from basic (shared) header fields

        Note that we get the translations from the center of the
        image.

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3, 2, 1))
        >>> hdr.default_x_flip
        True
        >>> hdr.get_base_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        '''
        hdr = self._structarr
        dims = hdr['dim']
        ndim = dims[0]
        return shape_zoom_affine(hdr['dim'][1:ndim+1],
                                 hdr['pixdim'][1:ndim+1],
                                 self.default_x_flip)

    get_best_affine = get_base_affine

    def get_zooms(self):
        ''' Get zooms from header

        Returns
        -------
        z : tuple
           tuple of header zoom values

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.get_zooms()
        (1.0,)
        >>> hdr.set_data_shape((1,2))
        >>> hdr.get_zooms()
        (1.0, 1.0)
        >>> hdr.set_zooms((3, 4))
        >>> hdr.get_zooms()
        (3.0, 4.0)
        '''
        hdr = self._structarr
        dims = hdr['dim']
        ndim = dims[0]
        if ndim == 0:
            return (1.0,)
        pixdims = hdr['pixdim']
        return tuple(pixdims[1:ndim+1])

    def set_zooms(self, zooms):
        ''' Set zooms into header fields

        See docstring for ``get_zooms`` for examples
        '''
        hdr = self._structarr
        dims = hdr['dim']
        ndim = dims[0]
        zooms = np.asarray(zooms)
        if len(zooms) != ndim:
            raise HeaderDataError('Expecting %d zoom values for ndim %d'
                                  % (ndim, ndim))
        if np.any(zooms < 0):
            raise HeaderDataError('zooms must be positive')
        pixdims = hdr['pixdim']
        pixdims[1:ndim+1] = zooms[:]

    def as_analyze_map(self):
        return self

    def get_data_offset(self):
        ''' Return offset into data file to read data

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.get_data_offset()
        0
        >>> hdr['vox_offset'] = 12
        >>> hdr.get_data_offset()
        12
        '''
        return int(self._structarr['vox_offset'])

    def get_slope_inter(self):
        ''' Get scalefactor and intercept

        These are not implemented for basic Analyze
        '''
        return None, None

    def set_slope_inter(self, slope, inter=None):
        ''' Set slope and / or intercept into header

        Set slope and intercept for image data, such that, if the image
        data is ``arr``, then the scaled image data will be ``(arr *
        slope) + inter``

        In this case, for Analyze images, we can't store the slope or the
        intercept, so this method only checks that `slope` is None or 1.0, and
        that `inter` is None or 0.

        Parameters
        ----------
        slope : None or float
            If float, value must be 1.0 or we raise a ``HeaderTypeError``
        inter : None or float, optional
            If float, value must be 0.0 or we raise a ``HeaderTypeError``
        '''
        if (slope is None or slope == 1.0) and (inter is None or inter == 0):
            return
        raise HeaderTypeError('Cannot set slope != 1 or intercept != 0 '
                              'for Analyze headers')

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return (klass._chk_sizeof_hdr,
                klass._chk_datatype,
                klass._chk_bitpix,
                klass._chk_pixdims)

    ''' Check functions in format expected by BatteryRunner class '''

    @staticmethod
    def _chk_sizeof_hdr(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['sizeof_hdr'] == 348:
            return hdr, rep
        rep.problem_level = 30
        rep.problem_msg = 'sizeof_hdr should be 348'
        if fix:
            hdr['sizeof_hdr'] = 348
            rep.fix_msg = 'set sizeof_hdr to 348'
        return hdr, rep

    @classmethod
    def _chk_datatype(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dtype = klass._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 40
            rep.problem_msg = 'data code %d not recognized' % code
        else:
            if dtype.itemsize == 0:
                rep.problem_level = 40
                rep.problem_msg = 'data code %d not supported' % code
            else:
                return hdr, rep
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep

    @classmethod
    def _chk_bitpix(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dt = klass._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 10
            rep.problem_msg = 'no valid datatype to fix bitpix'
            if fix:
                rep.fix_msg = 'no way to fix bitpix'
            return hdr, rep
        bitpix = dt.itemsize * 8
        if bitpix == hdr['bitpix']:
            return hdr, rep
        rep.problem_level = 10
        rep.problem_msg = 'bitpix does not match datatype'
        if fix:
            hdr['bitpix'] = bitpix # inplace modification
            rep.fix_msg = 'setting bitpix to match datatype'
        return hdr, rep

    @staticmethod
    def _chk_pixdims(hdr, fix=False):
        rep = Report(HeaderDataError)
        pixdims = hdr['pixdim']
        spat_dims = pixdims[1:4]
        if not np.any(spat_dims <= 0):
            return hdr, rep
        neg_dims = spat_dims < 0
        zero_dims = spat_dims == 0
        pmsgs = []
        fmsgs = []
        if np.any(zero_dims):
            level = 30
            pmsgs.append('pixdim[1,2,3] should be non-zero')
            if fix:
                spat_dims[zero_dims] = 1
                fmsgs.append('setting 0 dims to 1')
        if np.any(neg_dims):
            level = 35
            pmsgs.append('pixdim[1,2,3] should be positive')
            if fix:
                spat_dims = np.abs(spat_dims)
                fmsgs.append('setting to abs of pixdim values')
        rep.problem_level = level
        rep.problem_msg = ' and '.join(pmsgs)
        if fix:
            pixdims[1:4] = spat_dims
            rep.fix_msg = ' and '.join(fmsgs)
        return hdr, rep


class AnalyzeImage(SpatialImage):
    header_class = AnalyzeHeader
    files_types = (('image','.img'), ('header','.hdr'))
    _compressed_exts = ('.gz', '.bz2')

    ImageArrayProxy = ArrayProxy

    def get_header(self):
        ''' Return header
        '''
        return self._header

    def get_data_dtype(self):
        return self._header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._header.set_data_dtype(dtype)

    @classmethod
    def from_file_map(klass, file_map):
        ''' class method to create image from mapping in `file_map ``
        '''
        hdr_fh, img_fh = klass._get_fileholders(file_map)
        hdrf = hdr_fh.get_prepare_fileobj(mode='rb')
        header = klass.header_class.from_fileobj(hdrf)
        if hdr_fh.fileobj is None: # was filename
            hdrf.close()
        hdr_copy = header.copy()
        imgf = img_fh.fileobj
        if imgf is None:
            imgf = img_fh.filename
        data = klass.ImageArrayProxy(imgf, hdr_copy)
        # Initialize without affine to allow header to pass through unmodified
        img = klass(data, None, header, file_map=file_map)
        # set affine from header though
        img._affine = header.get_best_affine()
        img._load_cache = {'header': hdr_copy,
                           'affine': img._affine.copy(),
                           'file_map': copy_file_map(file_map)}
        return img

    @staticmethod
    def _get_fileholders(file_map):
        """ Return fileholder for header and image

        Allows single-file image types to return one fileholder for both types.
        For Analyze there are two fileholders, one for the header, one for the
        image.
        """
        return file_map['header'], file_map['image']

    def _write_header(self, header_file, header, slope, inter):
        ''' Utility routine to write header

        Parameters
        ----------
        header_file : file-like
           file-like object implementing ``write``, open for writing
        header : header object
        slope : None or float
           slope for data scaling
        inter : None or float
           intercept for data scaling
        '''
        header.set_slope_inter(slope, inter)
        header.write_to(header_file)

    def to_file_map(self, file_map=None):
        ''' Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        '''
        if file_map is None:
            file_map = self.file_map
        data = self.get_data()
        self.update_header()
        hdr = self.get_header()
        out_dtype = self.get_data_dtype()
        arr_writer = make_array_writer(data,
                                       out_dtype,
                                       hdr.has_data_slope,
                                       hdr.has_data_intercept)
        hdr_fh, img_fh = self._get_fileholders(file_map)
        # Check if hdr and img refer to same file; this can happen with odd
        # analyze images but most often this is because it's a single nifti file
        hdr_img_same = hdr_fh.same_file_as(img_fh)
        hdrf = hdr_fh.get_prepare_fileobj(mode='wb')
        if hdr_img_same:
            imgf = hdrf
        else:
            imgf = img_fh.get_prepare_fileobj(mode='wb')
        slope, inter = get_slope_inter(arr_writer)
        self._write_header(hdrf, hdr, slope, inter)
        # Write image
        shape = hdr.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        seek_tell(imgf, hdr.get_data_offset())
        arr_writer.to_fileobj(imgf)
        if hdr_fh.fileobj is None: # was filename
            hdrf.close()
        if not hdr_img_same:
            if img_fh.fileobj is None: # was filename
                imgf.close()
        self._header = hdr
        self.file_map = file_map

    def update_header(self):
        ''' Harmonize header with image data and affine

        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = AnalyzeImage(data, affine)
        >>> hdr = img.get_header()
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_header()
        >>> hdr.get_data_shape() == (2, 3, 4)
        True
        >>> hdr.get_zooms()
        (1.0, 2.0, 3.0)
        '''
        hdr = self._header
        # We need to update the header if the data shape has changed.  It's a
        # bit difficult to change the data shape using the standard API, but
        # maybe it happened
        if not self._data is None and hdr.get_data_shape() != self._data.shape:
            hdr.set_data_shape(self._data.shape)
        # If the affine is not None, and it is different from the main affine in
        # the header, update the heaader
        if self._affine is None:
            return
        if np.allclose(self._affine, hdr.get_best_affine()):
            return
        RZS = self._affine[:3, :3]
        vox = np.sqrt(np.sum(RZS * RZS, axis=0))
        hdr['pixdim'][1:4] = vox


load = AnalyzeImage.load
save = AnalyzeImage.instance_to_filename

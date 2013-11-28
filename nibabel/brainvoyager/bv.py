# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) file formats.

please look at the support site of BrainInnovation for further informations
about the file formats: http://support.brainvoyager.com/

This file implements basic functionality for BV file formats. Look into bv_*.py
files for implementations of the different file formats.

Author: Thomas Emmerling
"""
from __future__ import division
import numpy as np

from ..volumeutils import array_to_file, array_from_file, make_dt_codes
from ..spatialimages import Header, HeaderDataError, SpatialImage
from ..fileholders import copy_file_map
from ..arrayproxy import CArrayProxy
from .. import imageglobals as imageglobals
from ..batteryrunners import BatteryRunner, Report
from struct import pack, unpack, calcsize
from ..externals import OrderedDict

_dtdefs = (  # code, conversion function, equivalent dtype, aliases
    (1, 'int16', np.uint16),
    (2, 'float32', np.float32),
    (3, 'uint8', np.uint8))

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)

# Set example hdr_dict_proto for BV file formats
BV_HDR_DICT_PROTO = (
    ('resolution', 'h', 3),
    ('x_start', 'h', 57),
    ('x_end', 'h', 231),
    ('y_start', 'h', 52),
    ('y_end', 'h', 172),
    ('z_start', 'h', 59),
    ('z_end', 'h', 197),
)


def readCString(f, nStrings=1, bufsize=1000, startPos=None, strip=True,
                rewind=False):
    """Read a zero-terminated string from a file object.

    Read and return a zero-terminated string from a file object.

    Parameters
    ----------
    f : fileobj
       File object to use
    nStrings: int, optional
       Number of strings to search (and return). Default is 1.
    bufsize: int, optional
       Define the buffer size that should be searched for the string.
       Default is 1000 bytes.
    startPos: int, optional
       Define the start file position from which to search. If None then start
       where the file object currently points to. Default is None.
    strip : bool, optional
       Whether to strip the trailing zero from the returned string.
       Default is True.
    rewind: bool, optional
       Whether the fileobj f should be returned to the initial position after
       reading. Default is False.

    Returns
    -------
    str_list : generator of string(s)
    """
    currentPos = f.tell()
    if strip:
        suffix = b''
    else:
        suffix = b'\x00'
    if startPos is not None:
        f.seek(startPos)
    data = f.read(bufsize)
    lines = data.split(b'\x00')
    str_list = []
    if rewind:
        f.seek(currentPos)
    else:
        offset = 0
        for s in range(nStrings):
            offset += len(lines[s]) + 1
        f.seek(currentPos + offset)
    for s in range(nStrings):
        str_list.append(lines[s] + suffix)
    return str_list


def parse_BV_header(hdr_dict_proto, fileobj, parent_hdr_dict=None):
    """Parse the header of a BV file format.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the ``prts`` field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes below.
    fileobj : fileobj
        File object to use. Make sure that the current position is at the
        beginning of the header (e.g. at 0).
    parent_hdr_dict: OrderedDict
        When parse_BV_header() is called recursively the already filled
        (parent) hdr_dict is passed to give access to n_fields_name fields
        outside the current scope (see below).

    Returns
    -------
    hdr_dict : OrderedDict
        An OrderedDict containing all header fields parsed from the file.

    Notes
    -----
    The description of `hdr_dict_proto` below is notated according to
    https://docs.python.org/3/reference/introduction.html#notation

        hdr_dict_proto ::= ((element_proto))*
        element_proto ::= '(' name ',' pack_format ',' default ')'  |
                          '(' name ',' pack_format ',' '(' default ','
                            c_fields_name ',' c_fields_value ')' ')'  |
                          '(' name ',' hdr_dict_proto ',' n_fields_name ')'
        pack_format ::= 'b' | 'h' | 'f' | 'z'
        name ::= str
        n_fields_name ::= str
        c_fields_name ::= str
        c_fields_value ::= int | float | bytes
        default ::= int | float | bytes

    The pack_format codes have meaning::

        b := signed char (1 byte)
        B := unsigned char (1 byte)
        h := signed short integer (2 bytes)
        i := signed integer (4 bytes)
        I := unsigned integer (4 bytes)
        f := float (4 bytes)
        z := zero-terminated string (variable bytes)

    The n_fields_name is used to indicate the name of a header field that
    contains a number for nested header fields loops (e.g. 'NrOfSubMaps' in the
    VMP file header).

    The c_fields_name and c_fields_value parameters are used for header fields
    that are only written depending on the value of another header field (e.g.
    'NrOfLags' in the VMP file header).
    """
    hdr_dict = OrderedDict()
    for name, format, def_or_name in hdr_dict_proto:
        # handle zero-terminated strings
        if format == 'z':
            value = readCString(fileobj)[0]
        # handle array fields
        elif isinstance(format, tuple):
            value = []
            # check the length of the array to expect
            if def_or_name in hdr_dict:
                n_values = hdr_dict[def_or_name]
            else:
                n_values = parent_hdr_dict[def_or_name]
            for i in range(n_values):
                value.append(parse_BV_header(format, fileobj, hdr_dict))
        # handle conditional fields
        elif isinstance(def_or_name, tuple):
            if hdr_dict[def_or_name[1]] == def_or_name[2]:
                bytes = fileobj.read(calcsize(format))
                value = unpack('<' + format, bytes)[0]
            else:  # assign the default value
                value = def_or_name[0]
        else:  # unpack bytes of type format
            bytes = fileobj.read(calcsize(format))
            value = unpack('<' + format, bytes)[0]
        hdr_dict[name] = value
    return hdr_dict


def pack_BV_header(hdr_dict_proto, hdr_dict, parent_hdr_dict=None):
    """Pack the header of a BV file format into a byte string.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the ``prts`` field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdr_dict: OrderedDict
       hdr_dict that contains the fields and values to for the respective
       BV file format.
    parent_hdr_dict: OrderedDict
       When parse_BV_header() is called recursively the already filled
       (parent) hdr_dict is passed to give access to n_fields_name fields
       outside the current scope (see below).

    Returns
    -------
    binaryblock : bytes
        Binary representation of header ready for writing to file.
    """
    binary_parts = []
    for name, format, def_or_name in hdr_dict_proto:
        value = hdr_dict[name]
        # handle zero-terminated strings
        if format == 'z':
            part = value + b'\x00'
        # handle array fields
        elif isinstance(format, tuple):
            # check the length of the array to expect
            if def_or_name in hdr_dict:
                n_values = hdr_dict[def_or_name]
            else:
                n_values = parent_hdr_dict[def_or_name]
            sub_parts = []
            for i in range(n_values):
                sub_parts.append(pack_BV_header(format, value[i], hdr_dict))
            part = b''.join(sub_parts)
        # handle conditional fields
        elif isinstance(def_or_name, tuple):
            if hdr_dict[def_or_name[1]] == def_or_name[2]:
                part = pack('<' + format, value)
            else:
                continue
        else:
            part = pack('<' + format, value)
        binary_parts.append(part)
    return b''.join(binary_parts)


def calc_BV_header_size(hdr_dict_proto, hdr_dict, parent_hdr_dict=None):
    """Calculate the binary size of a hdr_dict for a BV file format header.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the prts field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdr_dict: OrderedDict
       hdr_dict that contains the fields and values to for the respective
       BV file format.
    parent_hdr_dict: OrderedDict
       When parse_BV_header() is called recursively the already filled
       (parent) hdr_dict is passed to give access to n_fields_name fields
       outside the current scope (see below).

    Returns
    -------
    hdr_size : int
        Size of header when packed into bytes ready for writing to file.
    """
    hdr_size = 0
    for name, format, def_or_name in hdr_dict_proto:
        value = hdr_dict[name]
        # handle zero-terminated strings
        if format == 'z':
            hdr_size += len(value) + 1
        # handle array fields
        elif isinstance(format, tuple):
            # check the length of the array to expect
            if def_or_name in hdr_dict:
                n_values = hdr_dict[def_or_name]
            else:
                n_values = parent_hdr_dict[def_or_name]
            for i in range(n_values):
                # recursively iterate through the fields of all items
                # in the array
                hdr_size += calc_BV_header_size(format, value[i], hdr_dict)
        # handle conditional fields
        elif isinstance(def_or_name, tuple):
            if hdr_dict[def_or_name[1]] == def_or_name[2]:
                hdr_size += calcsize(format)
            else:
                continue
        else:
            hdr_size += calcsize(format)
    return hdr_size


def update_BV_header(hdr_dict_proto, hdr_dict_old, hdr_dict_new,
                     parent_old=None, parent_new=None):
    """Update a hdr_dict after changed nested-loops-number or conditional fields.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the prts field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdr_dict_old: OrderedDict
       hdr_dict before any changes.
    hdr_dict_new: OrderedDict
       hdr_dict with changed fields in n_fields_name or c_fields_name fields.
    parent_old: OrderedDict
       When update_BV_header() is called recursively the not yet updated
       (parent) hdr_dict is passed to give access to n_fields_name fields
       outside the current scope (see below).
    parent_new: OrderedDict
       When update_BV_header() is called recursively the not yet updated
       (parent) hdr_dict is passed to give access to n_fields_name fields
       outside the current scope (see below).

    Returns
    -------
    hdr_dict_new : OrderedDict
        An updated version hdr_dict correcting effects of changed nested and
        conditional fields.
    """
    for name, format, def_or_name in hdr_dict_proto:
        # handle nested loop fields
        if isinstance(format, tuple):
            # calculate the change of array length and the new array length
            if def_or_name in hdr_dict_old:
                delta_values = hdr_dict_new[def_or_name] - \
                    hdr_dict_old[def_or_name]
                n_values = hdr_dict_new[def_or_name]
            else:
                delta_values = parent_new[def_or_name] - \
                    parent_old[def_or_name]
                n_values = parent_new[def_or_name]
            if delta_values > 0:  # add nested loops
                for i in range(delta_values):
                    hdr_dict_new[name].append(_proto2default(format, hdr_dict_new))
            elif delta_values < 0:  # remove nested loops
                for i in range(abs(delta_values)):
                    hdr_dict_new[name].pop()
            # loop over nested fields
            for i in range(n_values):
                update_BV_header(format, hdr_dict_old[name][i],
                                 hdr_dict_new[name][i], hdr_dict_old,
                                 hdr_dict_new)
    return hdr_dict_new


def _proto2default(proto, parent_default_hdr=None):
    """Helper for creating a BV header OrderedDict with default parameters.

    Create an OrderedDict that contains keys with the header fields, and
    default values.

    See :func:`parse_BV_header` for description of `proto` format.
    """
    default_hdr = OrderedDict()
    for name, format, def_or_name in proto:
        if isinstance(format, tuple):
            value = []
            # check the length of the array to expect
            if def_or_name in default_hdr:
                n_values = default_hdr[def_or_name]
            else:
                n_values = parent_default_hdr[def_or_name]
            for i in range(n_values):
                value.append(_proto2default(format, default_hdr))
            default_hdr[name] = value
        # handle conditional fields
        elif isinstance(def_or_name, tuple):
            default_hdr[name] = def_or_name[0]
        else:
            default_hdr[name] = def_or_name
    return default_hdr


def combine_st(st_array, inv=False):
    """Combine spatial transformation matrices.

    This recursive function returns the dot product of all spatial
    transformation matrices given in st_array for applying them in one go.
    The order of multiplication follow the order in the given array.

    Parameters
    ----------
    st_array: array
        array filled with transformation matrices of shape (4, 4)

    inv: boolean
        Set to true to invert the transformation matrices before
        multiplication.

    Returns
    -------
    combined_st : array of shape (4, 4)
    """
    if len(st_array) == 1:
        if inv:
            return np.linalg.inv(st_array[0])
        else:
            return st_array[0]
    if inv:
        return np.dot(np.linalg.inv(st_array[0, :, :]),
                      combine_st(st_array[1:, :, :], inv=inv))
    else:
        return np.dot(st_array[0, :, :],
                      combine_st(st_array[1:, :, :], inv=inv))


def parse_st(st_dict):
    """Parse spatial transformation stored in a BV header OrderedDict.

    This function parses a given OrderedDict from a BV header field and returns
    a spatial transformation matrix as a numpy array.

    Parameters
    ----------
    st_dict: OrderedDict
        OrderedDict filled with transformation matrices of shape (4, 4)

    Returns
    -------
    st_array : array of shape (4, 4)
    """
    if st_dict['nr_of_trans_val'] != 16:
        raise BvError('spatial transformation has to be of shape (4, 4)')
    st_array = []
    for v in range(st_dict['nr_of_trans_val']):
        st_array.append(st_dict['trans_val'][v]['value'])
    return np.array(st_array).reshape((4, 4))


class BvError(Exception):
    """Exception for BV format related problems.

    To be raised whenever there is a problem with a BV fileformat.
    """

    pass


class BvFileHeader(Header):
    """Class to hold information from a BV file header."""

    # Copies of module-level definitions
    _data_type_codes = data_type_codes
    _field_recoders = {'datatype': data_type_codes}

    # format defaults
    # BV files are radiological (left-is-right) by default
    # (VTC files have a flag for that, however)
    default_x_flip = True
    default_endianness = '<'  # BV files are always little-endian
    allowed_dtypes = [1, 2, 3]
    default_dtype = 2
    allowed_dimensions = [3]
    data_layout = 'C'
    hdr_dict_proto = BV_HDR_DICT_PROTO

    def __init__(self,
                 hdr_dict=None,
                 endianness=default_endianness,
                 check=True,
                 offset=None):
        """Initialize header from binary data block.

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
        offset : int, optional
            offset of the actual data into to binary file (in bytes)
        """
        if endianness != self.default_endianness:
            raise BvError('BV files are always little-endian')
        self.endianness = self.default_endianness
        if hdr_dict is None:
            hdr_dict = _proto2default(self.hdr_dict_proto)
        self._hdr_dict = hdr_dict
        if offset is None:
            self.set_data_offset(calc_BV_header_size(
                self.hdr_dict_proto, self._hdr_dict))
        if 'framing_cube' in self._hdr_dict:
            self._framing_cube = self._hdr_dict['framing_cube']
        else:
            self._framing_cube = self._guess_framing_cube()
        if check:
            self.check_fix()
        return

    @classmethod
    def from_fileobj(klass, fileobj, endianness=default_endianness,
                     check=True):
        """Return read structure with given or guessed endiancode.

        Parameters
        ----------
        fileobj : file-like object
           Needs to implement ``read`` method
        endianness : None or endian code, optional
           Code specifying endianness of read data

        Returns
        -------
        header : BvFileHeader object
           BvFileHeader object initialized from data in fileobj
        """
        hdr_dict = parse_BV_header(klass.hdr_dict_proto, fileobj)
        offset = fileobj.tell()
        return klass(hdr_dict, endianness, check, offset)

    @classmethod
    def from_header(klass, header=None, check=False):
        """Class method to create header from another header.

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
        """
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
        try:  # check if there is a specific conversion routine
            mapping = header.as_bv_map()
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
                # being of the same meaning as for BV types
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

    def copy(self):
        """Copy object to independent representation.

        The copy should not be affected by any changes to the original
        object.
        """
        return self.__class__(self._hdr_dict)

    def _set_format_specifics(self):
        """Utility routine to set format specific header stuff."""
        pass

    def data_from_fileobj(self, fileobj):
        """Read data array from `fileobj`.

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           data array
        """
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset,
                               order=self.data_layout)

    def get_data_dtype(self):
        """Get numpy dtype for data.

        For examples see ``set_data_dtype``
        """
        if 'datatype' in self._hdr_dict:
            code = self._hdr_dict['datatype']
        else:
            code = self.default_dtype
        dtype = self._data_type_codes.dtype[code]
        return dtype.newbyteorder(self.endianness)

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code or dtype or type."""
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise HeaderDataError(
                'data dtype "%s" not recognized' % datatype)
        if code not in self.allowed_dtypes:
            raise HeaderDataError(
                'data dtype "%s" not supported' % datatype)
        dtype = self._data_type_codes.dtype[code]
        if 'datatype' in self._hdr_dict.keys():
            self._hdr_dict['datatype'] = code
            return
        if dtype.newbyteorder(self.endianness) != self.get_data_dtype():
            raise HeaderDataError(
                'File format does not support setting of header!')

    def get_xflip(self):
        """Get xflip for data."""
        return self.default_x_flip

    def set_xflip(self, xflip):
        """Set xflip for data."""
        if xflip is True:
            return
        else:
            raise BvError('cannot change Left-right convention!')

    def get_data_shape(self):
        """Get shape of data."""
        raise NotImplementedError

    def set_data_shape(self, shape):
        """Set shape of data."""
        raise NotImplementedError

    def get_base_affine(self):
        """Get affine from basic (shared) header fields.

        Note that we get the translations from the center of the
        (guessed) framing cube of the referenced VMR (anatomical) file.

        Internal storage of the image is ZYXT, where (in patient coordiante/
        real world orientations):
        Z := axis increasing from right to left (R to L)
        Y := axis increasing from superior to inferior (S to I)
        X := axis increasing from anterior to posterior (A to P)
        T := volumes (if present in file format)
        """
        zooms = self.get_zooms()
        if not self.get_xflip():
            # make the BV internal Z axis neurological (left-is-left);
            # not default in BV files!
            zooms[0] *= -1

        # compute the rotation
        rot = np.zeros((3, 3))
        # make the flipped BV Z axis the new R axis
        rot[:, 0] = [-zooms[0], 0, 0]
        # make the flipped BV X axis the new A axis
        rot[:, 1] = [0, 0, -zooms[2]]
        # make the flipped BV Y axis the new S axis
        rot[:, 2] = [0, -zooms[1], 0]

        # compute the translation
        fcc = np.array(self.get_framing_cube()) / 2  # center of framing cube
        bbc = np.array(self.get_bbox_center())  # center of bounding box
        tra = np.dot((bbc - fcc), rot)

        # assemble
        M = np.eye(4, 4)
        M[0:3, 0:3] = rot
        M[0:3, 3] = tra.T

        return M

    get_best_affine = get_base_affine

    get_default_affine = get_base_affine

    get_affine = get_base_affine

    def _guess_framing_cube(self):
        """Guess the dimensions of the framing cube.

        Guess the dimensions of the framing cube that constitutes the
        coordinate system boundaries for the bounding box.

        For most BV file formats this need to be guessed from
        x_end, y_end, and z_end in the header.
        """
        # then start guessing...
        hdr = self._hdr_dict
        # get the ends of the bounding box (highest values in each dimension)
        x = hdr['x_end']
        y = hdr['y_end']
        z = hdr['z_end']

        # compare with possible framing cubes
        for fc in [256, 384, 512, 768, 1024]:
            if any([d > fc for d in (x, y, z)]):
                continue
            else:
                return fc, fc, fc

    def get_framing_cube(self):
        """Get the dimensions of the framing cube.

        Get the dimensions of the framing cube that constitutes the
        coordinate system boundaries for the bounding box.
        For most BV file formats this need to be guessed from
        x_end, y_end, and z_end in the header.
        """
        return self._framing_cube

    def set_framing_cube(self, fc):
        """Set the dimensions of the framing cube.

        Set the dimensions of the framing cube that constitutes the
        coordinate system boundaries for the bounding box
        For most BV file formats this need to be guessed from
        x_end, y_end, and z_end in the header.
        Use this if you know about the framing cube for the BV file.
        """
        self._framing_cube = fc

    def get_bbox_center(self):
        """Get the center coordinate of the bounding box.

        Get the center coordinate of the bounding box with respect to the
        framing cube.
        """
        hdr = self._hdr_dict
        x = hdr['x_start'] + \
            ((hdr['x_end'] - hdr['x_start']) / 2)
        y = hdr['y_start'] + \
            ((hdr['y_end'] - hdr['y_start']) / 2)
        z = hdr['z_start'] + \
            ((hdr['z_end'] - hdr['z_start']) / 2)
        return z, y, x

    def get_zooms(self):
        shape = self.get_data_shape()
        return tuple(float(self._hdr_dict['resolution'])
                     for d in shape[0:3])

    def set_zooms(self, zooms):
        if type(zooms) == int:
            self._hdr_dict['resolution'] = zooms
        else:
            if any([zooms[i] != zooms[i + 1] for i in range(len(zooms) - 1)]):
                raise BvError('Zooms for all dimensions must be equal!')
            else:
                self._hdr_dict['resolution'] = int(zooms[0])

    def as_analyze_map(self):
        raise NotImplementedError

    def set_data_offset(self, offset):
        """Set offset into data file to read data."""
        self._data_offset = offset

    def get_data_offset(self):
        """Return offset into data file to read data."""
        self.set_data_offset(calc_BV_header_size(
                             self.hdr_dict_proto, self._hdr_dict))
        return self._data_offset

    def get_slope_inter(self):
        """BV formats do not do scaling."""
        return None, None

    def write_to(self, fileobj):
        """Write header to fileobj.

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method

        Returns
        -------
        None
        """
        binaryblock = pack_BV_header(self.hdr_dict_proto, self._hdr_dict)
        fileobj.write(binaryblock)

    def check_fix(self, logger=None, error_level=None):
        """Check BV header with checks."""
        if logger is None:
            logger = imageglobals.logger
        if error_level is None:
            error_level = imageglobals.error_level
        battrun = BatteryRunner(self.__class__._get_checks())
        self, reports = battrun.check_fix(self)
        for report in reports:
            report.log_raise(logger, error_level)

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class"""
        return (klass._chk_fileversion,)

    ''' Check functions in format expected by BatteryRunner class '''

    @classmethod
    def _chk_fileversion(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if 'version' in hdr._hdr_dict:
            version = hdr._hdr_dict['version']
            if version in klass.supported_fileversions:
                return hdr, rep
            else:
                rep.problem_level = 40
                rep.problem_msg = 'fileversion %d is not supported' % version
                if fix:
                    rep.fix_msg = 'not attempting fix'
                return hdr, rep
        return hdr, rep


class BvFileImage(SpatialImage):
    """Class to hold information from a BV image file."""

    # Set the class of the corresponding header
    header_class = BvFileHeader

    # Set the label ('image') and the extension ('.bv') for a (dummy) BV file
    files_types = (('image', '.bv'),)

    # BV files are not compressed...
    _compressed_exts = ()

    # use the row-major CArrayProxy
    ImageArrayProxy = CArrayProxy

    def update_header(self):
        """Harmonize header with image data and affine.

        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = SpatialImage(data, affine)
        >>> hdr = img.get_header()
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_header()
        >>> hdr.get_data_shape() == (2, 3, 4)
        True
        >>> hdr.get_zooms()
        (1.0, 2.0, 3.0)
        """
        hdr = self._header
        shape = self._dataobj.shape
        # We need to update the header if the data shape has changed.  It's a
        # bit difficult to change the data shape using the standard API, but
        # maybe it happened
        if hdr.get_data_shape() != shape:
            hdr.set_data_shape(shape)

    @classmethod
    def from_file_map(klass, file_map):
        """Load image from `file_map`.

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        bvf = file_map['image'].get_prepare_fileobj('rb')
        header = klass.header_class.from_fileobj(bvf)
        affine = header.get_affine()
        hdr_copy = header.copy()
        # use row-major memory presentation!
        data = klass.ImageArrayProxy(bvf, hdr_copy)
        img = klass(data, affine, header, file_map=file_map)
        img._load_cache = {'header': hdr_copy,
                           'affine': None,
                           'file_map': copy_file_map(file_map)}
        return img

    def _write_header(self, header_file, header):
        """Utility routine to write BV header.

        Parameters
        ----------
        header_file : file-like
           file-like object implementing ``write``, open for writing
        header : header object
        """
        header.write_to(header_file)

    def _write_data(self, bvfile, data, header):
        """Utility routine to write BV image.

        Parameters
        ----------
        bvfile : file-like
           file-like object implementing ``seek`` or ``tell``, and
           ``write``
        data : array-like
           array to write
        header : analyze-type header object
           header
        """
        shape = header.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        offset = header.get_data_offset()
        out_dtype = header.get_data_dtype()
        array_to_file(data, bvfile, out_dtype, offset, order='C')

    def to_file_map(self, file_map=None):
        """Write image to `file_map` or contained ``self.file_map``.

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        if file_map is None:
            file_map = self.file_map
        data = self.get_data()
        with file_map['image'].get_prepare_fileobj('wb') as bvf:
            self._write_header(bvf, self.header)
            self._write_data(bvf, data, self.header)
        self.file_map = file_map

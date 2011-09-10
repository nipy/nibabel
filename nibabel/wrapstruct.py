# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Class to wrap numpy structured array

============
 wrapstruct
============

The ``WrapStruct`` class is a wrapper around a numpy structured array type.

It implements:

* Mappingness from the underlying structured array fields
* ``from_fileobj``, ``write_to`` methods to read and write data to fileobj
* A mechanism for setting checks and fixes to the data on object creation
* A pretty printing mechanism whereby field values can be displayed as
  corresponding strings (see ``get_value_label`` and ``__str_``_
* Endianness guessing, and on-the-fly swapping

Mappingness
-----------

You can access and set fields of the contained structarr using standard
__getitem__ / __setitem__ syntax:

    hdr['field'] = 10

Wrapped structures also implement general mappingness:

    hdr.keys()
    hdr.items()
    hdr.values()

Properties::

    .endianness (read only)
    .binaryblock (read only)
    .structarr (read only)

Methods::

    .as_byteswapped(endianness)
    .check_fix()
    .__str__
    .__eq__
    .__ne__

Class methods::

    .diagnose_binaryblock
    .as_byteswapped(endianness)
    .write_to(fileobj)
    .from_fileobj(fileobj)

More sophisticated headers can add more methods and attributes.

Header checking
---------------

We have a file, and we would like feedback as to whether there are any
problems with this header, and whether they are fixable::

   data = WrapStruct.data_from_fileobj(open('myfile.hdr'))
   res = WrapStruct.diagnose_binaryblock(hdr.binaryblock)

This will run all known checks, with no fixes, returning a string with
diagnostic output.

In creating a header object, we might want to check the header data.  If it
passes the error threshold, it goes through::

   hdr = WrapStruct.from_fileobj(good_fileobj)

whereas::

   hdr = WrapStruct.from_fileobj(bad_fileobj)

would raise some error, with output to logging (see below).

If we want the header, come what may::

   hdr = WrapStruct.from_fileobj(bad_fileobj, check=False)

We set the error level (the level of problem that the ``check=True``
versions will accept as OK) from global defaults::

   import nibabel as nib
   nib.imageglobals.error_level = 30

The same for logging::

   nib.imageglobals.logger = logger
"""
import numpy as np

from .volumeutils import (pretty_mapping, endian_codes, native_code,
                          swapped_code)
from .spatialimages import HeaderDataError
from . import imageglobals as imageglobals
from .batteryrunners import BatteryRunner


class WrapStruct(object):
    _field_recoders = {}
    # placeholder datatype
    dtype_def = [('integer', 'i4')]
    _dtype = np.dtype(dtype_def)

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
        >>> hdr1 = WrapStruct() # an empty header
        >>> hdr1.endianness == native_code
        True
        >>> hdr1['integer']
        array(0)
        >>> hdr1['integer'] = 1
        >>> hdr1['integer']
        array(1)
        '''
        if binaryblock is None:
            self._header_data = self._empty_headerdata(endianness)
            return
        # check size
        if len(binaryblock) != self._dtype.itemsize:
            raise HeaderDataError('Binary block is wrong size')
        hdr = np.ndarray(shape=(),
                         dtype=self._dtype,
                         buffer=binaryblock)
        if endianness is None:
            endianness = self._guessed_endian(hdr)
        else:
            endianness = endian_codes[endianness]
        if endianness != native_code:
            dt = self._dtype.newbyteorder(endianness)
            hdr = np.ndarray(shape=(),
                             dtype=dt,
                             buffer=binaryblock)
        self._header_data = hdr.copy()
        if check:
            self.check_fix()
        return

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        ''' Return read header with given or guessed endiancode

        Parameters
        ----------
        fileobj : file-like object
           Needs to implement ``read`` method
        endianness : None or endian code, optional
           Code specifying endianness of read data

        Returns
        -------
        hdr : WrapStruct object
           WrapStruct object initialized from data in fileobj
        '''
        raw_str = fileobj.read(klass._dtype.itemsize)
        return klass(raw_str, endianness, check)

    @property
    def binaryblock(self):
        ''' binary block of data as string

        Returns
        -------
        binaryblock : string
            string giving binary data block

        Examples
        --------
        >>> # Make default empty header
        >>> hdr = WrapStruct()
        >>> len(hdr.binaryblock)
        4
        '''
        return self._header_data.tostring()

    def write_to(self, fileobj):
        ''' Write header to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = WrapStruct()
        >>> from StringIO import StringIO #23dt : BytesIO
        >>> str_io = StringIO() #23dt : BytesIO
        >>> hdr.write_to(str_io)
        >>> hdr.binaryblock == str_io.getvalue()
        True
        '''
        fileobj.write(self.binaryblock)

    @property
    def endianness(self):
        ''' endian code of binary data

        The endianness code gives the current byte order
        interpretation of the binary data.

        Examples
        --------
        >>> hdr = WrapStruct()
        >>> code = hdr.endianness
        >>> code == native_code
        True

        Notes
        -----
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization, or occasionally byteswapping the
        data - but this is done via the as_byteswapped method
        '''
        if self._header_data.dtype.isnative:
            return native_code
        return swapped_code

    def copy(self):
        ''' Return copy of header

        >>> hdr = WrapStruct()
        >>> hdr['integer'] = 3
        >>> hdr2 = hdr.copy()
        >>> hdr2 is hdr
        False
        >>> hdr2['integer']
        array(3)
        '''
        return self.__class__(
                self.binaryblock,
                self.endianness, check=False)

    def __eq__(self, other):
        ''' equality between two headers defined by binaryblock

        Examples
        --------
        >>> hdr = WrapStruct()
        >>> hdr2 = WrapStruct()
        >>> hdr == hdr2
        True
        >>> hdr3 = WrapStruct(endianness=swapped_code)
        >>> hdr == hdr3
        True
        '''
        this_end = self.endianness
        this_bb = self.binaryblock
        if this_end == other.endianness:
            return this_bb == other.binaryblock
        other_bb = other._header_data.byteswap().tostring()
        return this_bb == other_bb

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, item):
        ''' Return values from header data

        Examples
        --------
        >>> hdr = WrapStruct()
        >>> hdr['integer'] == 0
        True
        '''
        return self._header_data[item]

    def __setitem__(self, item, value):
        ''' Set values in header data

        Examples
        --------
        >>> hdr = WrapStruct()
        >>> hdr['integer'] = 3
        >>> hdr['integer']
        array(3)
        '''
        self._header_data[item] = value

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        ''' Return keys from header data'''
        return list(self._dtype.names)

    def values(self):
        ''' Return values from header data'''
        data = self._header_data
        return [data[key] for key in self._dtype.names]

    def items(self):
        ''' Return items from header data'''
        return zip(self.keys(), self.values())

    def check_fix(self, logger=None, error_level=None):
        ''' Check header data with checks '''
        if logger is None:
            logger = imageglobals.logger
        if error_level is None:
            error_level = imageglobals.error_level
        battrun = BatteryRunner(self.__class__._get_checks())
        self, reports = battrun.check_fix(self)
        for report in reports:
            report.log_raise(logger, error_level)

    @classmethod
    def diagnose_binaryblock(klass, binaryblock, endianness=None):
        ''' Run checks over header binary data, return string '''
        hdr = klass(binaryblock, endianness=endianness, check=False)
        battrun = BatteryRunner(klass._get_checks())
        reports = battrun.check_only(hdr)
        return '\n'.join([report.message
                          for report in reports if report.message])

    def _guessed_endian(self, hdr):
        ''' Guess intended endianness from mapping-like ``hdr``

        Parameters
        ----------
        hdr : mapping-like
           hdr for which to guess endianness

        Returns
        -------
        endianness : {'<', '>'}
           Guessed endianness of header
        '''
        raise NotImplementedError

    def _empty_headerdata(self, endianness=None):
        ''' Return header data for empty header with given endianness
        '''
        dt = self._dtype
        if endianness is not None:
            endianness = endian_codes[endianness]
            dt = dt.newbyteorder(endianness)
        hdr_data = np.zeros((), dtype=dt)
        return hdr_data

    @property
    def structarr(self):
        ''' header data, with data fields

        Examples
        --------
        >>> hdr1 = WrapStruct() # an empty header
        >>> an_int = hdr1.structarr['integer']
        >>> hdr1.structarr = None
        Traceback (most recent call last):
           ...
        AttributeError: can't set attribute
        '''
        return self._header_data

    def __str__(self):
        ''' Return string representation for printing '''
        summary = "%s object, endian='%s'" % (self.__class__,
                                              self.endianness)
        def _getter(obj, key):
            try:
                return obj.get_value_label(key)
            except ValueError:
                return obj[key]

        return '\n'.join(
            [summary,
             pretty_mapping(self, _getter)])

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
        '''
        if not fieldname in self._field_recoders:
            raise ValueError('%s not a coded field' % fieldname)
        code = int(self._header_data[fieldname])
        return self._field_recoders[fieldname].label[code]

    def as_byteswapped(self, endianness=None):
        ''' return new byteswapped header object with given ``endianness``

        Guaranteed to make a copy even if endianness is the same as
        the current endianness.

        Parameters
        ----------
        endianness : None or string, optional
           endian code to which to swap.  None means swap from current
           endianness, and is the default

        Returns
        -------
        hdr : header object
           hdr object with given endianness

        Examples
        --------
        >>> hdr = WrapStruct()
        >>> hdr.endianness == native_code
        True
        >>> bs_hdr = hdr.as_byteswapped()
        >>> bs_hdr.endianness == swapped_code
        True
        >>> bs_hdr = hdr.as_byteswapped(swapped_code)
        >>> bs_hdr.endianness == swapped_code
        True
        >>> bs_hdr is hdr
        False
        >>> bs_hdr == hdr
        True

        If you write to the resulting byteswapped data, it does not
        change the original.

        >>> bs_hdr['integer'] = 3
        >>> bs_hdr == hdr
        False

        If you swap to the same endianness, it returns a copy

        >>> nbs_hdr = hdr.as_byteswapped(native_code)
        >>> nbs_hdr.endianness == native_code
        True
        >>> nbs_hdr is hdr
        False
        '''
        current = self.endianness
        if endianness is None:
            if current == native_code:
                endianness = swapped_code
            else:
                endianness = native_code
        else:
            endianness = endian_codes[endianness]
        if endianness == current:
            return self.copy()
        hdr_data = self._header_data.byteswap()
        return self.__class__(hdr_data.tostring(),
                              endianness,
                              check=False)

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return ()

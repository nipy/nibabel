# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Utility functions for analyze-like formats '''

import sys
import gzip
import bz2

import numpy as np

from .py3k import isfileobj, ZEROB

sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'

endian_codes = (# numpy code, aliases
    ('<', 'little', 'l', 'le', 'L', 'LE'),
    ('>', 'big', 'BIG', 'b', 'be', 'B', 'BE'),
    (native_code, 'native', 'n', 'N', '=', '|', 'i', 'I'),
    (swapped_code, 'swapped', 's', 'S', '!'))
# We'll put these into the Recoder class after we define it

#: default compression level when writing gz and bz2 files
default_compresslevel = 1

#: convenience variables for numpy types
floating_point_types = (np.sctypes['complex'] +
                        np.sctypes['float'])
integer_types = (np.sctypes['int'] + np.sctypes['uint'])
numeric_types = floating_point_types + integer_types


class Recoder(object):
    ''' class to return canonical code(s) from code or aliases

    The concept is a lot easier to read in the implementation and
    tests than it is to explain, so...

    >>> # If you have some codes, and several aliases, like this:
    >>> code1 = 1; aliases1=['one', 'first']
    >>> code2 = 2; aliases2=['two', 'second']
    >>> # You might want to do this:
    >>> codes = [[code1]+aliases1,[code2]+aliases2]
    >>> recodes = Recoder(codes)
    >>> recodes.code['one']
    1
    >>> recodes.code['second']
    2
    >>> recodes.code[2]
    2
    >>> # Or maybe you have a code, a label and some aliases
    >>> codes=((1,'label1','one', 'first'),(2,'label2','two'))
    >>> # you might want to get back the code or the label
    >>> recodes = Recoder(codes, fields=('code','label'))
    >>> recodes.code['first']
    1
    >>> recodes.code['label1']
    1
    >>> recodes.label[2]
    'label2'
    >>> # For convenience, you can get the first entered name by
    >>> # indexing the object directly
    >>> recodes[2]
    2
    '''
    def __init__(self, codes, fields=('code',), map_maker=dict):
        ''' Create recoder object

        ``codes`` give a sequence of code, alias sequences
        ``fields`` are names by which the entries in these sequences can be
        accessed.

        By default ``fields`` gives the first column the name
        "code".  The first column is the vector of first entries
        in each of the sequences found in ``codes``.  Thence you can
        get the equivalent first column value with ob.code[value],
        where value can be a first column value, or a value in any of
        the other columns in that sequence.

        You can give other columns names too, and access them in the
        same way - see the examples in the class docstring.

        Parameters
        ----------
        codes : seqence of sequences
            Each sequence defines values (codes) that are equivalent
        fields : {('code',) string sequence}, optional
            names by which elements in sequences can be accessed
        map_maker: callable, optional
            constructor for dict-like objects used to store key value pairs.
            Default is ``dict``.  ``map_maker()`` generates an empty mapping.
            The mapping need only implement ``__getitem__, __setitem__, keys,
            values``.
        '''
        self.fields = tuple(fields)
        self.field1 = {} # a placeholder for the check below
        for name in fields:
            if name in self.__dict__:
                raise KeyError('Input name %s already in object dict'
                               % name)
            self.__dict__[name] = map_maker()
        self.field1 = self.__dict__[fields[0]]
        self.add_codes(codes)

    def add_codes(self, code_syn_seqs):
        ''' Add codes to object

        Parameters
        ----------
        code_syn_seqs : sequence
            sequence of sequences, where each sequence ``S = code_syn_seqs[n]``
            for n in 0..len(code_syn_seqs), is a sequence giving values in the
            same order as ``self.fields``.  Each S should be at least of the
            same length as ``self.fields``.  After this call, if ``self.fields
            == ['field1', 'field2'], then ``self.field1[S[n]] == S[0]`` for all
            n in 0..len(S) and ``self.field2[S[n]] == S[1]`` for all n in
            0..len(S).

        Examples
        --------
        >>> code_syn_seqs = ((1, 'one'), (2, 'two'))
        >>> rc = Recoder(code_syn_seqs)
        >>> rc.value_set() == set((1,2))
        True
        >>> rc.add_codes(((3, 'three'), (1, 'first')))
        >>> rc.value_set() == set((1,2,3))
        True
        '''
        for code_syns in code_syn_seqs:
            # Add all the aliases
            for alias in code_syns:
                # For all defined fields, make every value in the sequence be an
                # entry to return matching index value.
                for field_ind, field_name in enumerate(self.fields):
                    self.__dict__[field_name][alias] = code_syns[field_ind]

    def __getitem__(self, key):
        ''' Return value from field1 dictionary (first column of values)

        Returns same value as ``obj.field1[key]`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code[key]``

        >>> codes = ((1, 'one'), (2, 'two'))
        >>> Recoder(codes)['two']
        2
        '''
        return self.field1[key]

    def __contains__(self, key):
        """ True if field1 in recoder contains `key`
        """
        try:
            self.field1[key]
        except KeyError:
            return False
        return True

    def keys(self):
        ''' Return all available code and alias values

        Returns same value as ``obj.field1.keys()`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code.keys()``

        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> k = Recoder(codes).keys()
        >>> set(k) == set([1, 2, 'one', 'repeat value', 'two'])
        True
        '''
        return self.field1.keys()

    def value_set(self, name=None):
        ''' Return set of possible returned values for column

        By default, the column is the first column.

        Returns same values as ``set(obj.field1.values())`` and,
        with the default initializing``fields`` argument of
        fields=('code',), this will return the same as
        ``set(obj.code.values())``

        Parameters
        ----------
        name : {None, string}
            Where default of none gives result for first column

        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> vs = Recoder(codes).value_set()
        >>> vs == set([1, 2]) # Sets are not ordered, hence this test
        True
        >>> rc = Recoder(codes, fields=('code', 'label'))
        >>> rc.value_set('label') == set(('one', 'two', 'repeat value'))
        True
        '''
        if name is None:
            d = self.field1
        else:
            d = self.__dict__[name]
        return set(d.values())


# Endian code aliases
endian_codes = Recoder(endian_codes)


class DtypeMapper(object):
    """ Specialized mapper for numpy dtypes

    We pass this mapper into the Recoder class to deal with numpy dtype hashing.

    The hashing problem is that dtypes that compare equal may not have the same
    hash.  This is true for numpys up to the current at time of writing (1.6.0).
    For numpy 1.2.1 at least, even dtypes that look exactly the same in terms of
    fields don't always have the same hash.  This makes dtypes difficult to use
    as keys in a dictionary.

    This class wraps a dictionary in order to implement a __getitem__ to deal
    with dtype hashing. If the key doesn't appear to be in the mapping, and it
    is a dtype, we compare (using ==) all known dtype keys to the input key, and
    return any matching values for the matching key.
    """
    def __init__(self):
        self._dict = {}
        self._dtype_keys = []

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def __setitem__(self, key, value):
        """ Set item into mapping, checking for dtype keys

        Cache dtype keys for comparison test in __getitem__
        """
        self._dict[key] = value
        if hasattr(key, 'subdtype'):
            self._dtype_keys.append(key)

    def __getitem__(self, key):
        """ Get item from mapping, checking for dtype keys

        First do simple hash lookup, then check for a dtype key that has failed
        the hash lookup.  Look then for any known dtype keys that compare equal
        to `key`.
        """
        try:
            return self._dict[key]
        except KeyError:
            pass
        if hasattr(key, 'subdtype'):
            for dt in self._dtype_keys:
                if key == dt:
                    return self._dict[dt]
        raise KeyError(key)


def pretty_mapping(mapping, getterfunc=None):
    ''' Make pretty string from mapping

    Adjusts text column to print values on basis of longest key.
    Probably only sensible if keys are mainly strings.

    You can pass in a callable that does clever things to get the values
    out of the mapping, given the names.  By default, we just use
    ``__getitem__``

    Parameters
    ----------
    mapping : mapping
       implementing iterator returning keys and .items()
    getterfunc : None or callable
       callable taking two arguments, ``obj`` and ``key`` where ``obj``
       is the passed mapping.  If None, just use ``lambda obj, key:
       obj[key]``

    Returns
    -------
    str : string

    Examples
    --------
    >>> d = {'a key': 'a value'}
    >>> print pretty_mapping(d)
    a key  : a value
    >>> class C(object): # to control ordering, show get_ method
    ...     def __iter__(self):
    ...         return iter(('short_field','longer_field'))
    ...     def __getitem__(self, key):
    ...         if key == 'short_field':
    ...             return 0
    ...         if key == 'longer_field':
    ...             return 'str'
    ...     def get_longer_field(self):
    ...         return 'method string'
    >>> def getter(obj, key):
    ...     # Look for any 'get_<name>' methods
    ...     try:
    ...         return obj.__getattribute__('get_' + key)()
    ...     except AttributeError:
    ...         return obj[key]
    >>> print pretty_mapping(C(), getter)
    short_field   : 0
    longer_field  : method string
    '''
    if getterfunc is None:
        getterfunc = lambda obj, key: obj[key]
    lens = [len(str(name)) for name in mapping]
    mxlen = np.max(lens)
    fmt = '%%-%ds  : %%s' % mxlen
    out = []
    for name in mapping:
        value = getterfunc(mapping, name)
        out.append(fmt % (name, value))
    return '\n'.join(out)


def make_dt_codes(codes_seqs):
    ''' Create full dt codes Recoder instance from datatype codes

    Include created numpy dtype (from numpy type) and opposite endian
    numpy dtype

    Parameters
    ----------
    codes_seqs : sequence of sequences
       contained sequences make be length 3 or 4, but must all be the same
       length. Elements are data type code, data type name, and numpy
       type (such as ``np.float32``).  The fourth element is the nifti string
       representation of the code (e.g. "NIFTI_TYPE_FLOAT32")

    Returns
    -------
    rec : ``Recoder`` instance
       Recoder that, by default, returns ``code`` when indexed with any
       of the corresponding code, name, type, dtype, or swapped dtype.
       You can also index with ``niistring`` values if codes_seqs had sequences
       of length 4 instead of 3.
    '''
    fields=['code', 'label', 'type']
    len0 = len(codes_seqs[0])
    if not len0 in (3,4):
        raise ValueError('Sequences must be length 3 or 4')
    if len0 == 4:
        fields.append('niistring')
    dt_codes = []
    intp_dt = np.dtype(np.intp)
    for seq in codes_seqs:
        if len(seq) != len0:
            raise ValueError('Sequences must all have the same length')
        np_type = seq[2]
        this_dt = np.dtype(np_type)
        # Add swapped dtype to synonyms
        code_syns = list(seq) + [this_dt, this_dt.newbyteorder(swapped_code)]
        dt_codes.append(code_syns)
    return Recoder(dt_codes, fields + ['dtype', 'sw_dtype'], DtypeMapper)


def can_cast(in_type, out_type, has_intercept=False, has_slope=False):
    ''' Return True if we can safely cast ``in_type`` to ``out_type``

    Parameters
    ----------
    in_type : numpy type
       type of data we will case from
    out_dtype : numpy type
       type that we want to cast to
    has_intercept : bool, optional
       Whether we can subtract a constant from the data (before scaling)
       before casting to ``out_dtype``.  Default is False
    has_slope : bool, optional
       Whether we can use a scaling factor to adjust slope of
       relationship of data to data in cast array.  Default is False

    Returns
    -------
    tf : bool
       True if we can safely cast, False otherwise

    Examples
    --------
    >>> can_cast(np.float64, np.float32)
    True
    >>> can_cast(np.complex128, np.float32)
    False
    >>> can_cast(np.int64, np.float32)
    True
    >>> can_cast(np.float32, np.int16)
    False
    >>> can_cast(np.float32, np.int16, False, True)
    True
    >>> can_cast(np.int16, np.uint8)
    False
    >>> can_cast(np.int16, np.uint8, False, True)
    False
    >>> can_cast(np.int16, np.uint8, True, True)
    True
    '''
    if np.can_cast(in_type, out_type):
        return True
    if in_type not in numeric_types or out_type not in numeric_types:
        return False
    if out_type in np.sctypes['complex']:
        return True
    if in_type in np.sctypes['complex']:
        return False
    if out_type in np.sctypes['float']:
        return True
    # now we have larger (u)int or float to smaller (u)int
    if not has_slope:
        return False
    if out_type in np.sctypes['uint']:
        if in_type not in np.sctypes['uint']:
            return has_intercept
    return True


def array_from_file(shape, in_dtype, infile, offset=0, order='F'):
    ''' Get array from file with specified shape, dtype and file offset

    Parameters
    ----------
    shape : sequence
        sequence specifying output array shape
    in_dtype : numpy dtype
        fully specified numpy dtype, including correct endianness
    infile : file-like
        open file-like object implementing at least read() and seek()
    offset : int, optional
        offset in bytes into infile to start reading array
        data. Default is 0
    order : {'F', 'C'} string
        order in which to write data.  Default is 'F' (fortran order).

    Returns
    -------
    arr : array-like
        array like object that can be sliced, containing data

    Examples
    --------
    >>> from StringIO import StringIO #23dt : BytesIO
    >>> bio = StringIO() #23dt : BytesIO
    >>> arr = np.arange(6).reshape(1,2,3)
    >>> _ = bio.write(arr.tostring('F')) # outputs int in python3
    >>> arr2 = array_from_file((1,2,3), arr.dtype, bio)
    >>> np.all(arr == arr2)
    True
    >>> bio = StringIO() #23dt : BytesIO
    >>> _ = bio.write(' ' * 10) #23dt : bytes
    >>> _ = bio.write(arr.tostring('F'))
    >>> arr2 = array_from_file((1,2,3), arr.dtype, bio, 10)
    >>> np.all(arr == arr2)
    True
    '''
    in_dtype = np.dtype(in_dtype)
    try: # Try memmapping file on disk
        arr = np.memmap(infile,
                        in_dtype,
                        mode='c',
                        shape=shape,
                        order=order,
                        offset=offset)
        # The error raised by memmap, for different file types, has
        # changed in different incarnations of the numpy routine
    except (AttributeError, TypeError, ValueError): # then read data
        infile.seek(offset)
        if len(shape) == 0:
            return np.array([])
        datasize = int(np.prod(shape) * in_dtype.itemsize)
        if datasize == 0:
            return np.array([])
        data_str = infile.read(datasize)
        if len(data_str) != datasize:
            if hasattr(infile, 'name'):
                file_str = 'file "%s"' % infile.name
            else:
                file_str = 'file object'
            msg = 'Expected %s bytes, got %s bytes from %s\n' \
                  % (datasize, len(data_str), file_str) + \
                  ' - could the file be damaged?'
            raise IOError(msg)
        arr = np.ndarray(shape,
                         in_dtype,
                         buffer=data_str,
                         order=order)
        # for some types, we can write to the string buffer without
        # worrying, but others we can't. 
        if isfileobj(infile) or isinstance(infile, (gzip.GzipFile,
                                                    bz2.BZ2File)):
            arr.flags.writeable = True
        else:
            arr = arr.copy()
    return arr


def array_to_file(data, fileobj, out_dtype=None, offset=0,
                  intercept=0.0, divslope=1.0,
                  mn=None, mx=None, order='F', nan2zero=True):
    ''' Helper function for writing arrays to disk

    Writes arrays as scaled by `intercept` and `divslope`, and clipped
    at (prescaling) `mn` minimum, and `mx` maximum.

    Parameters
    ----------
    data : array
       array to write
    fileobj : file-like
       file-like object implementing ``write`` method.
    out_dtype : None or dtype, optional
       dtype to write array as.  Data array will be coerced to this
       dtype before writing. If None (default) then use input data
       type.
    offset : int, optional
       offset into fileobj at which to start writing data. Default is
       0.
    intercept : scalar, optional
       scalar to subtract from data, before dividing by ``divslope``.
       Default is 0.0
    divslope : None or scalar, optional
       scalefactor to *divide* data by before writing.  Default
       is 1.0. If None, there is no valid data, we write zeros.
    mn : scalar, optional
       minimum threshold in (unscaled) data, such that all data below
       this value are set to this value. Default is None (no threshold)
    mx : scalar, optional
       maximum threshold in (unscaled) data, such that all data above
       this value are set to this value. Default is None (no threshold)
    order : {'F', 'C'}, optional
       memory order to write array.  Default is 'F'
    nan2zero : {True, False}, optional
       Whether to set NaN values to 0 when writing integer output.
       Defaults to True.  If False, NaNs will be represented as numpy
       does when casting, and this can be odd (often the lowest
       available integer value)

    Examples
    --------
    >>> from StringIO import StringIO #23dt : BytesIO
    >>> sio = StringIO() #23dt : BytesIO
    >>> data = np.arange(10, dtype=np.float)
    >>> array_to_file(data, sio, np.float)
    >>> sio.getvalue() == data.tostring('F')
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0) # outputs 0 in python 3
    >>> array_to_file(data, sio, np.int16)
    >>> sio.getvalue() == data.astype(np.int16).tostring()
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)
    >>> array_to_file(data.byteswap(), sio, np.float)
    >>> sio.getvalue() == data.byteswap().tostring('F')
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)
    >>> array_to_file(data, sio, np.float, order='C')
    >>> sio.getvalue() == data.tostring('C')
    True
    '''
    data = np.asarray(data)
    in_dtype = data.dtype
    if out_dtype is None:
        out_dtype = in_dtype
    else:
        out_dtype = np.dtype(out_dtype)
    try:
        fileobj.seek(offset)
    except IOError:
        msg = sys.exc_info()[1] # python 2 / 3 compatibility
        if fileobj.tell() != offset:
            raise IOError(msg)
    if divslope is None: # No valid data
        fileobj.write(ZEROB * (data.size*out_dtype.itemsize))
        return
    nan2zero = (nan2zero and
                data.dtype in floating_point_types and
                out_dtype not in floating_point_types)
    needs_copy = nan2zero or mx or mn or intercept or divslope != 1.0
    if data.ndim < 2: # a little hack to allow 1D arrays in loop below
        data = [data]
    elif order == 'F':
        data = data.T
    elif order != 'C':
        raise ValueError('Order should be one of F or C')
    for dslice in data: # cycle over largest dimension to save memory
        if needs_copy:
            dslice = dslice.copy()
        if nan2zero:
            dslice[np.isnan(dslice)] = 0
        if mx:
            dslice[dslice > mx] = mx
        if mn:
            dslice[dslice < mn] = mn
        if intercept:
            dslice -= intercept
        if divslope != 1.0:
            dslice /= divslope
        if in_dtype == out_dtype:
            fileobj.write(dslice.tostring())
        elif in_dtype == out_dtype.newbyteorder('S'): # just byte swapped
            out_arr = dslice.byteswap()
            fileobj.write(out_arr.tostring())
        else:
            fileobj.write(dslice.astype(out_dtype).tostring())


def calculate_scale(data, out_dtype, allow_intercept):
    ''' Calculate scaling and optional intercept for data

    Parameters
    ----------
    data : array
    out_dtype : dtype
       output data type
    allow_intercept : bool
       If True allow non-zero intercept

    Returns
    -------
    scaling : None or float
       scalefactor to divide into data.  None if no valid data
    intercept : None or float
       intercept to subtract from data.  None if no valid data
    mn : None or float
       minimum of finite value in data or None if this will not
       be used to threshold data
    mx : None or float
       minimum of finite value in data, or None if this will not
       be used to threshold data
    '''
    default_ret = (1.0, 0.0, None, None)
    in_dtype = data.dtype
    if np.can_cast(in_dtype, out_dtype):
        return default_ret
    in_type = in_dtype.type
    out_type = out_dtype.type
    if out_type in floating_point_types:
        return default_ret
    mn, mx = finite_range(data)
    if mn == np.inf: # No valid data
        return None, None, None, None
    info = np.iinfo(out_type)
    type_min = info.min
    type_max = info.max
    if in_type in integer_types:
        # scaling a big int type into a smaller one
        if mx <= type_max and mn >= type_min: # lucky; already in range
            return default_ret
        scaling, intercept = scale_min_max(mn, mx,
                                           out_type,
                                           allow_intercept)
        return scaling, intercept, None, None
    # should now be scaling a fp type to an int type
    if not in_type in np.sctypes['float']:
        raise TypeError('Unexpected input dtype %s' % in_dtype)
    scaling, intercept = scale_min_max(mn, mx, out_type,
                                        allow_intercept)
    return scaling, intercept, mn, mx


def scale_min_max(mn, mx, out_type, allow_intercept):
    ''' Return scaling and intercept min, max of data, given output type

    Returns ``scalefactor`` and ``intercept`` to best fit data with
    given ``mn`` and ``mx`` min and max values into range of data type
    with ``type_min`` and ``type_max`` min and max values for type.

    The calculated scaling is therefore::

        scaled_data = (data-intercept) / scalefactor

    Parameters
    ----------
    mn : scalar
       data minimum value
    mx : scalar
       data maximum value
    out_type : numpy type
       numpy type of output
    allow_intercept : bool
       If true, allow calculation of non-zero intercept.  Otherwise,
       returned intercept is always 0.0

    Returns
    -------
    scalefactor : numpy scalar, dtype=np.maximum_sctype(np.float)
       scalefactor by which to divide data after subtracting intercept
    intercept : numpy scalar, dtype=np.maximum_sctype(np.float)
       value to subtract from data before dividing by scalefactor

    >>> scale_min_max(0, 255, np.uint8, False)
    (1.0, 0.0)
    >>> scale_min_max(-128, 127, np.int8, False)
    (1.0, 0.0)
    >>> scale_min_max(0, 127, np.int8, False)
    (1.0, 0.0)
    >>> scaling, intercept = scale_min_max(0, 127, np.int8,  True)
    >>> np.allclose((0 - intercept) / scaling, -128)
    True
    >>> np.allclose((127 - intercept) / scaling, 127)
    True
    >>> scaling, intercept = scale_min_max(-10, -1, np.int8, True)
    >>> np.allclose((-10 - intercept) / scaling, -128)
    True
    >>> np.allclose((-1 - intercept) / scaling, 127)
    True
    >>> scaling, intercept = scale_min_max(1, 10, np.int8, True)
    >>> np.allclose((1 - intercept) / scaling, -128)
    True
    >>> np.allclose((10 - intercept) / scaling, 127)
    True

    Notes
    -----
    The large integers lead to python long types as max / min for type.
    To contain the rounding error, we need to use the maximum numpy
    float types when casting to float.

    '''
    if mn > mx:
        raise ValueError('min value > max value')
    try:
        info = np.iinfo(out_type)
    except ValueError:
        info = np.finfo(out_type)
    mn, mx, type_min, type_max = np.array(
        [mn, mx, info.min, info.max], np.maximum_sctype(np.float))
    # with intercept
    if allow_intercept:
        data_range = mx-mn
        if data_range == 0:
            return 1.0, mn
        type_range = type_max - type_min
        scaling = data_range / type_range
        intercept = mn - type_min * scaling
        return scaling, intercept
    # without intercept
    if mx == 0 and mn == 0:
        return 1.0, 0.0
    if type_min == 0: # uint
        if mn < 0 and mx > 0:
            raise ValueError('Cannot scale negative and positive '
                             'numbers to uint without intercept')
        if mx < 0:
            scaling = mn / type_max
        else:
            scaling = mx / type_max
    else: # int
        if abs(mx) >= abs(mn):
            scaling = mx / type_max
        else:
            scaling = mn / type_min
    return scaling, 0.0


def finite_range(arr):
    ''' Return range (min, max) of finite values of ``arr``

    Parameters
    ----------
    arr : array

    Returns
    -------
    mn : scalar
       minimum of values in (flattened) array
    mx : scalar
       maximum of values in (flattened) array

    Examples
    --------
    >>> a = np.array([[-1, 0, 1],[np.inf, np.nan, -np.inf]])
    >>> finite_range(a)
    (-1.0, 1.0)
    >>> a = np.array([[np.nan],[np.nan]])
    >>> finite_range(a) == (np.inf, -np.inf)
    True
    >>> a = np.array([[-3, 0, 1],[2,-1,4]], dtype=np.int)
    >>> finite_range(a)
    (-3, 4)
    >>> a = np.array([[1, 0, 1],[2,3,4]], dtype=np.uint)
    >>> finite_range(a)
    (0, 4)
    >>> a = a + 1j
    >>> finite_range(a)
    Traceback (most recent call last):
       ...
    TypeError: Can only handle floats and (u)ints
    '''
    # Resort array to slowest->fastest memory change indices
    stride_order = np.argsort(arr.strides)[::-1]
    sarr = arr.transpose(stride_order)
    typ = sarr.dtype.type
    if typ in integer_types:
        return np.min(sarr), np.max(sarr)
    if typ not in np.sctypes['float']:
        raise TypeError('Can only handle floats and (u)ints')
    # Loop to avoid big isfinite temporary
    mx = -np.inf
    mn = np.inf
    for s in xrange(sarr.shape[0]):
        tmp = sarr[s]
        tmp = tmp[np.isfinite(tmp)]
        if tmp.size:
            mx = max(np.max(tmp), mx)
            mn = min(np.min(tmp), mn)
    return mn, mx


def allopen(fname, *args, **kwargs):
    ''' Generic file-like object open

    If input ``fname`` already looks like a file, pass through.
    If ``fname`` ends with recognizable compressed types, use python
    libraries to open as file-like objects (read or write)
    Otherwise, use standard ``open``.
    '''
    if hasattr(fname, 'write'):
        return fname
    if args:
        mode = args[0]
    elif 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'rb'
        args = (mode,)
    if fname.endswith('.gz'):
        if ('w' in mode and
            len(args) < 2 and
            not 'compresslevel' in kwargs):
            kwargs['compresslevel'] = default_compresslevel
        opener = gzip.open
    elif fname.endswith('.bz2'):
        if ('w' in mode and
            len(args) < 3 and
            not 'compresslevel' in kwargs):
            kwargs['compresslevel'] = default_compresslevel
        opener = bz2.BZ2File
    else:
        opener = open
    return opener(fname, *args, **kwargs)


def shape_zoom_affine(shape, zooms, x_flip=True):
    ''' Get affine implied by given shape and zooms

    We get the translations from the center of the image (implied by
    `shape`).

    Parameters
    ----------
    shape : (N,) array-like
       shape of image data. ``N`` is the number of dimensions
    zooms : (N,) array-like
       zooms (voxel sizes) of the image
    x_flip : {True, False}
       whether to flip the X row of the affine.  Corresponds to
       radiological storage on disk.

    Returns
    -------
    aff : (4,4) array
       affine giving correspondance of voxel coordinates to mm
       coordinates, taking the center of the image as origin

    Examples
    --------
    >>> shape = (3, 5, 7)
    >>> zooms = (3, 2, 1)
    >>> shape_zoom_affine((3, 5, 7), (3, 2, 1))
    array([[-3.,  0.,  0.,  3.],
           [ 0.,  2.,  0., -4.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    >>> shape_zoom_affine((3, 5, 7), (3, 2, 1), False)
    array([[ 3.,  0.,  0., -3.],
           [ 0.,  2.,  0., -4.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    '''
    shape = np.asarray(shape)
    zooms = np.array(zooms) # copy because of flip below
    ndims = len(shape)
    if ndims != len(zooms):
        raise ValueError('Should be same length of zooms and shape')
    if ndims >= 3:
        shape = shape[:3]
        zooms = zooms[:3]
    else:
        full_shape = np.ones((3,))
        full_zooms = np.ones((3,))
        full_shape[:ndims] = shape[:]
        full_zooms[:ndims] = zooms[:]
        shape = full_shape
        zooms = full_zooms
    if x_flip:
        zooms[0] *= -1
    # Get translations from center of image
    origin = (shape-1) / 2.0
    aff = np.eye(4)
    aff[:3, :3] = np.diag(zooms)
    aff[:3, -1] = -origin * zooms
    return aff


def rec2dict(rec):
    ''' Convert recarray to dictionary

    Also converts scalar values to scalars

    Parameters
    ----------
    rec : ndarray
       structured ndarray

    Returns
    -------
    dct : dict
       dict with key, value pairs as for `rec`

    Examples
    --------
    >>> r = np.zeros((), dtype = [('x', 'i4'), ('s', 'S10')])
    >>> d = rec2dict(r)
    >>> d == {'x': 0, 's': ''} #23dt : replace("''", "b''")
    True
    '''
    dct = {}
    for key in rec.dtype.fields:
        val = rec[key]
        try:
            val = np.asscalar(val)
        except ValueError:
            pass
        dct[key] = val
    return dct

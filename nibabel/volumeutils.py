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
import warnings
import gzip
import bz2

import numpy as np

from .py3k import isfileobj, ZEROB
from .casting import (shared_range, type_info, as_int, best_float, OK_FLOATS,
                      able_int_type)

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
    for seq in codes_seqs:
        if len(seq) != len0:
            raise ValueError('Sequences must all have the same length')
        np_type = seq[2]
        this_dt = np.dtype(np_type)
        # Add swapped dtype to synonyms
        code_syns = list(seq) + [this_dt, this_dt.newbyteorder(swapped_code)]
        dt_codes.append(code_syns)
    return Recoder(dt_codes, fields + ['dtype', 'sw_dtype'], DtypeMapper)


@np.deprecate_with_doc('Please use arraywriter classes instead')
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

    Whether we can actually cast int to uint when we don't have an intercept
    depends on the data.  That's why this function isn't very useful. But we
    assume that an integer is using its full range, and check whether scaling
    works in that situation.

    Here we need an intercept to scale the full range of an int to a uint

    >>> can_cast(np.int16, np.uint8, False, True)
    False
    >>> can_cast(np.int16, np.uint8, True, True)
    True
    '''
    in_dtype = np.dtype(in_type)
    # Whether we can cast depends on the data, and we've only got the type.
    # Let's assume integers use all of their range but floats etc not
    if in_dtype.kind in 'iu':
        info = np.iinfo(in_dtype)
        data = np.array([info.min, info.max], dtype=in_dtype)
    else: # Float or complex or something. Any old thing will do
        data = np.ones((1,), in_type)
    from .arraywriters import make_array_writer, WriterError
    try:
        _ = make_array_writer(data, out_type, has_slope, has_intercept)
    except WriterError:
        return False
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
        dtype to write array as.  Data array will be coerced to this dtype
        before writing. If None (default) then use input data type.
    offset : None or int, optional
        offset into fileobj at which to start writing data. Default is 0. None
        means start at current file position
    intercept : scalar, optional
        scalar to subtract from data, before dividing by ``divslope``.  Default
        is 0.0
    divslope : None or scalar, optional
        scalefactor to *divide* data by before writing.  Default is 1.0. If
        None, there is no valid data, we write zeros.
    mn : scalar, optional
        minimum threshold in (unscaled) data, such that all data below this
        value are set to this value. Default is None (no threshold). The typical
        use is to set -np.inf in the data to have this value (which might be the
        minimum non-finite value in the data).
    mx : scalar, optional
        maximum threshold in (unscaled) data, such that all data above this
        value are set to this value. Default is None (no threshold). The typical
        use is to set np.inf in the data to have this value (which might be the
        maximum non-finite value in the data).
    order : {'F', 'C'}, optional
        memory order to write array.  Default is 'F'
    nan2zero : {True, False}, optional
        Whether to set NaN values to 0 when writing integer output.  Defaults to
        True.  If False, NaNs will be represented as numpy does when casting;
        this depends on the underlying C library and is undefined. In practice
        `nan2zero` == False might be a good choice when you completely sure
        there will be no NaNs in the data. This value ignore for float ouptut
        types.

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
    data = np.asanyarray(data)
    in_dtype = data.dtype
    if out_dtype is None:
        out_dtype = in_dtype
    else:
        out_dtype = np.dtype(out_dtype)
    if not offset is None:
        seek_tell(fileobj, offset)
    if (divslope is None or
        (mn, mx) == (0, 0) or
        (None not in (mn, mx) and mx < mn)
       ):
        write_zeros(fileobj, data.size * out_dtype.itemsize)
        return
    if not order in 'FC':
        raise ValueError('Order should be one of F or C')
    # Force upcasting for floats by making atleast_1d
    slope, inter = [np.atleast_1d(v) for v in divslope, intercept]
    # (u)int to (u)int with inter alone - select precision
    if (slope == 1 and inter != 0 and
        in_dtype.kind in 'iu' and out_dtype.kind in 'iu' and
        inter == np.round(inter)): # (u)int to (u)int offset only scaling
        # Does range of in type minus inter fit in out type? If so, use that as
        # working type.  Otherwise use biggest float for max integer precision
        inter = inter.astype(_inter_type(in_dtype, -inter, out_dtype))
    # Do we need float -> int machinery?
    needs_f2i = out_dtype.kind in 'iu' and (
        in_dtype.kind == 'f' or
        slope != 1 or
        (inter != 0 and inter.dtype.kind == 'f'))
    if not needs_f2i:
        # Apply min max thresholding the standard way
        needs_pre_clip = (mn, mx) != (None, None)
        if needs_pre_clip:
            mn, mx = _dt_min_max(in_dtype, mn, mx)
    else: # We do need float to int machinery
        # Replace Nones in (mn, mx) with type min / max if necessary
        dt_mnmx = _dt_min_max(in_dtype, mn, mx)
        # Check what working type we need to cover range
        w_type = working_type(in_dtype, slope, inter)
        assert w_type in np.sctypes['float']
        w_type = best_write_scale_ftype(np.array(dt_mnmx, dtype=in_dtype),
                                        slope, inter, w_type)
        slope = slope.astype(w_type)
        inter = inter.astype(w_type)
        # Apply thresholding after scaling
        needs_pre_clip = False
        # We need to know the result of applying slope and inter to the min and
        # max of the array, in order to clip the output array, after applying
        # the slope and inter.  Otherwise we'd need to clip twice, once before
        # applying (slope, inter), and again after, to ensure we have not hit
        # over- or under-flow. For the same reason we need to know the result of
        # applying slope, inter to 0, in order to fill in the nan output value
        # after scaling etc. We could fill with 0 before scaling, but then we'd
        # have to do an extra copy before filling nans with 0, to avoid
        # overwriting the input array
        # Run min, max, 0 through scaling / rint
        specials = np.array(dt_mnmx + (0,), dtype=in_dtype)
        if inter != 0.0:
            specials = specials - inter
        if slope != 1.0:
            specials = specials / slope
        assert specials.dtype.type == w_type
        post_mn, post_mx, nan_fill = np.rint(specials)
        if post_mn > post_mx: # slope could be negative
            post_mn, post_mx = post_mx, post_mn
        # Ensure safe thresholds applied too
        both_mn, both_mx = shared_range(w_type, out_dtype)
        post_mn = np.max([post_mn, both_mn])
        post_mx = np.min([post_mx, both_mx])
    data = np.atleast_2d(data) # Trick to allow loop below for 1D arrays
    if order == 'F' or (data.ndim == 2 and data.shape[1] == 1):
        data = data.T
    for dslice in data: # cycle over first dimension to save memory
        if needs_pre_clip:
            dslice = np.clip(dslice, mn, mx)
        if inter != 0.0:
            dslice = dslice - inter
        if slope != 1.0:
            dslice = dslice / slope
        if needs_f2i:
            dslice = np.clip(np.rint(dslice), post_mn, post_mx)
            if nan2zero:
                nans = np.isnan(dslice)
                if np.any(nans):
                    dslice[nans] = nan_fill
            dslice = dslice.astype(out_dtype)
        elif dslice.dtype != out_dtype:
            dslice = dslice.astype(out_dtype)
        fileobj.write(dslice.tostring())


def _dt_min_max(dtype_like, mn, mx):
    """ Return ``mx` unless ``mx`` is None, else type max, likewise for ``mn``

    ``mn``, ``mx`` can be None, in which case return the type min / max.
    """
    dt = np.dtype(dtype_like)
    if dt.kind in 'fc':
        mnmx = (-np.inf, np.inf)
    elif dt.kind in 'iu':
        info = np.iinfo(dt)
        mnmx = (info.min, info.max)
    else:
        raise NotImplementedError("unknown dtype")
    return mnmx[0] if mn is None else mn, mnmx[1] if mx is None else mx


def write_zeros(fileobj, count, block_size=8194):
    """ Write `count` zero bytes to `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        with ``write`` method
    count : int
        number of bytes to write
    block_size : int, optional
        largest continuous block to write.
    """
    nblocks = int(count // block_size)
    rem = count % block_size
    blk = ZEROB * block_size
    for bno in range(nblocks):
        fileobj.write(blk)
    fileobj.write(ZEROB * rem)


def seek_tell(fileobj, offset):
    """ Seek in `fileobj` or check we're in the right place already

    Parameters
    ----------
    fileobj : file-like
        object implementing ``seek`` and (if seek raises an IOError) ``tell``
    offset : int
        position in file to which to seek
    """
    try:
        fileobj.seek(offset)
    except IOError:
        msg = sys.exc_info()[1] # python 2 / 3 compatibility
        if fileobj.tell() != offset:
            raise IOError(msg)


def apply_read_scaling(arr, slope = 1.0, inter = 0.0):
    """ Apply scaling in `slope` and `inter` to array `arr`

    This is for loading the array from a file (as opposed to the reverse scaling
    when saving an array to file)

    Return data will be ``arr * slope + inter``. The trick is that we have to
    find a good precision to use for applying the scaling.  The heuristic is
    that the data is always upcast to the higher of the types from `arr,
    `slope`, `inter` if `slope` and / or `inter` are not default values. If the
    dtype of `arr` is an integer, then we assume the data more or less fills the
    integer range, and upcast to a type such that the min, max of ``arr.dtype``
    * scale + inter, will be finite.

    Parameters
    ----------
    arr : array-like
    slope : scalar
    inter : scalar

    Returns
    -------
    ret : array
        array with scaling applied.  Maybe upcast in order to give room for the
        scaling. If scaling is default (1, 0), then `ret` may be `arr` ``ret is
        arr``.
    """
    if (slope, inter) == (1, 0):
        return arr
    shape = arr.shape
    # Force float / float upcasting by promoting to arrays
    arr, slope, inter = [np.atleast_1d(v) for v in arr, slope, inter]
    if arr.dtype.kind in 'iu':
        if (slope, inter) == (1, np.round(inter)):
            # (u)int to (u)int offset-only scaling
            inter = inter.astype(_inter_type(arr.dtype, inter))
        else: # int to float; get enough precision to avoid infs
            # Find floating point type for which scaling does not overflow, starting
            # at given type
            ftype = int_scinter_ftype(arr.dtype, slope, inter, slope.dtype.type)
            slope = slope.astype(ftype)
            inter = inter.astype(ftype)
    if slope != 1.0:
        arr = arr * slope
    if inter != 0.0:
        arr = arr + inter
    return arr.reshape(shape)


def _inter_type(in_type, inter, out_type=None):
    """ Return intercept type for array type `in_type`, starting value `inter`

    When scaling from an (u)int to a (u)int, we can often just use the intercept
    `inter`.  This routine is for that case. It works out if the min and max of
    `in_type`, plus the `inter` can fit into any other integer type, returning
    that type if so.  Otherwise it returns the most capable float.

    Parameters
    ----------
    in_type : numpy type
        Any specifier for a numpy dtype
    inter : scalar
        intercept
    out_type : None or numpy type, optional
        If not None, check any proposed `inter_type` to see whether the
        resulting values will fit within `out_type`; if so return proposed
        `inter_type`, otherwise return highest precision float

    Returns
    -------
    inter_type : numpy type
        Type to which inter should be cast for best integer scaling
    """
    info = np.iinfo(in_type)
    inter = as_int(inter)
    out_mn, out_mx = info.min + inter, info.max + inter
    values = [out_mn, out_mx, info.min, info.max]
    i_type = able_int_type(values + [inter])
    if i_type is None:
        return best_float()
    if out_type is None:
        return i_type
    # The proposal so far is to use an integer type i_type as the working type.
    # However, we might already know the output type to which we will cast.  If
    # the maximum range in the working type will not fit into the known output
    # type, this would require extra casting, so we back off to the best
    # floating point type.
    o_info = np.iinfo(out_type)
    if out_mn >= o_info.min and out_mx <= o_info.max:
        return i_type
    return best_float()


def working_type(in_type, slope=1.0, inter=0.0):
    """ Return array type from applying `slope`, `inter` to array of `in_type`

    Numpy type that results from an array of type `in_type` being combined with
    `slope` and `inter`. It returns something like the dtype type of
    ``((np.zeros((2,), dtype=in_type) - inter) / slope)``, but ignoring the
    actual values of `slope` and `inter`.

    Note that you would not necessarily get the same type by applying slope and
    inter the other way round.  Also, you'll see that the order in which slope
    and inter are applied is the opposite of the order in which they are passed.

    Parameters
    ----------
    in_type : numpy type specifier
        Numpy type of input array.  Any valid input for ``np.dtype()``
    slope : scalar, optional
        slope to apply to array.  If 1.0 (default), ignore this value and its
        type.
    inter : scalar, optional
        intercept to apply to array.  If 0.0 (default), ignore this value and
        its type.

    Returns
    -------
    wtype: numpy type
        Numpy type resulting from applying `inter` and `slope` to array of type
        `in_type`.
    """
    val = np.array([1], dtype=in_type)
    slope = np.array(slope)
    inter = np.array(inter)
    # Don't use real values to avoid overflows.  Promote to 1D to avoid scalar
    # casting rules.  Don't use ones_like, zeros_like because of a bug in numpy
    # <= 1.5.1 in converting complex192 / complex256 scalars.
    if inter != 0:
        val = val + np.array([0], dtype=inter.dtype)
    if slope != 1:
        val = val / np.array([1], dtype=slope.dtype)
    return val.dtype.type


@np.deprecate_with_doc('Please use arraywriter classes instead')
def calculate_scale(data, out_dtype, allow_intercept):
    ''' Calculate scaling and optional intercept for data

    Parameters
    ----------
    data : array
    out_dtype : dtype
       output data type in some form understood by ``np.dtype``
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
    # Code here is a compatibility shell around arraywriters refactor
    in_dtype = data.dtype
    out_dtype = np.dtype(out_dtype)
    if np.can_cast(in_dtype, out_dtype):
        return 1.0, 0.0, None, None
    from .arraywriters import make_array_writer, WriterError, get_slope_inter
    try:
        writer = make_array_writer(data, out_dtype, True, allow_intercept)
    except WriterError:
        msg = sys.exc_info()[1] # python 2 / 3 compatibility
        raise ValueError(msg)
    if out_dtype.kind in 'fc':
        return (1.0, 0.0, None, None)
    mn, mx = writer.finite_range()
    if (mn, mx) == (np.inf, -np.inf): # No valid data
        return (None, None, None, None)
    if not in_dtype.kind in 'fc':
        mn, mx = (None, None)
    return get_slope_inter(writer) + (mn, mx)


@np.deprecate_with_doc('Please use arraywriter classes instead')
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

    Examples
    --------
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
    We don't use this function anywhere in nibabel now, it's here for API
    compatibility only.

    The large integers lead to python long types as max / min for type.
    To contain the rounding error, we need to use the maximum numpy
    float types when casting to float.
    '''
    if mn > mx:
        raise ValueError('min value > max value')
    info = type_info(out_type)
    mn, mx, type_min, type_max = np.array(
        [mn, mx, info['min'], info['max']], np.maximum_sctype(np.float))
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


def int_scinter_ftype(ifmt, slope=1.0, inter=0.0, default=np.float32):
    """ float type containing int type `ifmt` * `slope` + `inter`

    Return float type that can represent the max and the min of the `ifmt` type
    after multiplication with `slope` and addition of `inter` with something
    like ``np.array([imin, imax], dtype=ifmt) * slope + inter``.

    Note that ``slope`` and ``inter`` get promoted to 1D arrays for this purpose
    to avoid the numpy scalar casting rules, which prevent scalars upcasting the
    array.

    Parameters
    ----------
    ifmt : object
        numpy integer type (e.g. np.int32)
    slope : float, optional
        slope, default 1.0
    inter : float, optional
        intercept, default 0.0
    default_out : object, optional
        numpy floating point type, default is ``np.float32``

    Returns
    -------
    ftype : object
        numpy floating point type

    Examples
    --------
    >>> int_scinter_ftype(np.int8, 1.0, 0.0) == np.float32
    True
    >>> int_scinter_ftype(np.int8, 1e38, 0.0) == np.float64
    True

    Notes
    -----
    It is difficult to make floats overflow with just addition because the
    deltas are so large at the extremes of floating point.  For example::

        >>> arr = np.array([np.finfo(np.float32).max], dtype=np.float32)
        >>> res = arr + np.iinfo(np.int16).max
        >>> arr == res
        array([ True], dtype=bool)
    """
    ii = np.iinfo(ifmt)
    tst_arr = np.array([ii.min, ii.max], dtype=ifmt)
    try:
        return _ftype4scaled_finite(tst_arr, slope, inter, 'read', default)
    except ValueError:
        raise ValueError('Overflow using highest floating point type')


def best_write_scale_ftype(arr, slope = 1.0, inter = 0.0, default=np.float32):
    """ Smallest float type to contain range of ``arr`` after scaling

    Scaling that will be applied to ``arr`` is ``(arr - inter) / slope``.

    Note that ``slope`` and ``inter`` get promoted to 1D arrays for this purpose
    to avoid the numpy scalar casting rules, which prevent scalars upcasting the
    array.

    Parameters
    ----------
    arr : array-like
        array that will be scaled
    slope : array-like, optional
        scalar such that output array will be ``(arr - inter) / slope``.
    inter : array-like, optional
        scalar such that output array will be ``(arr - inter) / slope``
    default : numpy type, optional
        minimum float type to return

    Returns
    -------
    ftype : numpy type
        Best floating point type for scaling.  If no floating point type
        prevents overflow, return the top floating point type.  If the input
        array ``arr`` already contains inf values, return the greater of the
        input type and the default type.

    Examples
    --------
    >>> arr = np.array([0, 1, 2], dtype=np.int16)
    >>> best_write_scale_ftype(arr, 1, 0) is np.float32
    True

    Specify higher default return value

    >>> best_write_scale_ftype(arr, 1, 0, default=np.float64) is np.float64
    True

    Even large values that don't overflow don't change output

    >>> arr = np.array([0, np.finfo(np.float32).max], dtype=np.float32)
    >>> best_write_scale_ftype(arr, 1, 0) is np.float32
    True

    Scaling > 1 reduces output values, so no upcast needed

    >>> best_write_scale_ftype(arr, np.float32(2), 0) is np.float32
    True

    Scaling < 1 increases values, so upcast may be needed (and is here)

    >>> best_write_scale_ftype(arr, np.float32(0.5), 0) is np.float64
    True
    """
    default = better_float_of(arr.dtype.type, default)
    if not np.all(np.isfinite(arr)):
        return default
    try:
        return _ftype4scaled_finite(arr, slope, inter, 'write', default)
    except ValueError:
        return OK_FLOATS[-1]


def better_float_of(first, second, default=np.float32):
    """ Return more capable float type of `first` and `second`

    Return `default` if neither of `first` or `second` is a float

    Parameters
    ----------
    first : numpy type specifier
        Any valid input to `np.dtype()``
    second : numpy type specifier
        Any valid input to `np.dtype()``
    default : numpy type specifier, optional
        Any valid input to `np.dtype()``

    Returns
    -------
    better_type : numpy type
        More capable of `first` or `second` if both are floats; if only one is
        a float return that, otherwise return `default`.

    Examples
    --------
    >>> better_float_of(np.float32, np.float64) is np.float64
    True
    >>> better_float_of(np.float32, 'i4') is np.float32
    True
    >>> better_float_of('i2', 'u4') is np.float32
    True
    >>> better_float_of('i2', 'u4', np.float64) is np.float64
    True
    """
    first = np.dtype(first)
    second = np.dtype(second)
    default = np.dtype(default).type
    kinds = (first.kind, second.kind)
    if not 'f' in kinds:
        return default
    if kinds == ('f', 'f'):
        if first.itemsize >= second.itemsize:
            return first.type
        return second.type
    if first.kind == 'f':
        return first.type
    return second.type


def _ftype4scaled_finite(tst_arr, slope, inter, direction='read',
                         default=np.float32):
    """ Smallest float type for scaling of `tst_arr` that does not overflow
    """
    assert direction in ('read', 'write')
    if not default in OK_FLOATS and default is np.longdouble:
        # Omitted longdouble
        return default
    def_ind = OK_FLOATS.index(default)
    # promote to arrays to avoid numpy scalar casting rules
    tst_arr = np.atleast_1d(tst_arr)
    slope = np.atleast_1d(slope)
    inter = np.atleast_1d(inter)
    warnings.filterwarnings('ignore', '.*overflow.*', RuntimeWarning)
    try:
        for ftype in OK_FLOATS[def_ind:]:
            tst_trans = tst_arr.copy()
            slope = slope.astype(ftype)
            inter = inter.astype(ftype)
            if direction == 'read': # as in reading of image from disk
                if slope != 1.0:
                    tst_trans = tst_trans * slope
                if inter != 0.0:
                    tst_trans = tst_trans + inter
            elif direction == 'write':
                if inter != 0.0:
                    tst_trans = tst_trans - inter
                if slope != 1.0:
                    tst_trans = tst_trans / slope
            if np.all(np.isfinite(tst_trans)):
                return ftype
    finally:
        warnings.filters.pop(0)
    raise ValueError('Overflow using highest floating point type')


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
    kind = sarr.dtype.kind
    if kind in 'iu':
        return np.min(sarr), np.max(sarr)
    if kind != 'f':
        raise TypeError('Can only handle floats and (u)ints')
    # Deal with 1D arrays in loop below
    sarr = np.atleast_2d(sarr)
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
    if fname.endswith('.gz') or fname.endswith('.mgz'):
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

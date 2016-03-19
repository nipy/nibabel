# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Header and image reading / writing functions for MGH image format

Author: Krish Subramaniam
'''
from os.path import splitext
import numpy as np

from ..volumeutils import (array_to_file, array_from_file, Recoder)
from ..spatialimages import HeaderDataError, SpatialImage
from ..fileholders import FileHolder, copy_file_map
from ..arrayproxy import ArrayProxy
from ..keywordonly import kw_only_meth
from ..openers import ImageOpener

# mgh header
# See https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
DATA_OFFSET = 284
# Note that mgh data is strictly big endian ( hence the > sign )
header_dtd = [
    ('version', '>i4'),
    ('dims', '>i4', (4,)),
    ('type', '>i4'),
    ('dof', '>i4'),
    ('goodRASFlag', '>i2'),
    ('delta', '>f4', (3,)),
    ('Mdc', '>f4', (3, 3)),
    ('Pxyz_c', '>f4', (3,))
]
# Optional footer. Also has more stuff after this, optionally
footer_dtd = [
    ('mrparms', '>f4', (4,))
]

header_dtype = np.dtype(header_dtd)
footer_dtype = np.dtype(footer_dtd)
hf_dtype = np.dtype(header_dtd + footer_dtd)

# caveat: Note that it's ambiguous to get the code given the bytespervoxel
# caveat 2: Note that the bytespervox you get is in str ( not an int)
_dtdefs = (  # code, conversion function, dtype, bytes per voxel
    (0, 'uint8', '>u1', '1', 'MRI_UCHAR', np.uint8, np.dtype(np.uint8),
     np.dtype(np.uint8).newbyteorder('>')),
    (4, 'int16', '>i2', '2', 'MRI_SHORT', np.int16, np.dtype(np.int16),
     np.dtype(np.int16).newbyteorder('>')),
    (1, 'int32', '>i4', '4', 'MRI_INT', np.int32, np.dtype(np.int32),
     np.dtype(np.int32).newbyteorder('>')),
    (3, 'float', '>f4', '4', 'MRI_FLOAT', np.float32, np.dtype(np.float32),
     np.dtype(np.float32).newbyteorder('>')))

# make full code alias bank, including dtype column
data_type_codes = Recoder(_dtdefs, fields=('code', 'label', 'dtype',
                                           'bytespervox', 'mritype',
                                           'np_dtype1', 'np_dtype2',
                                           'numpy_dtype'))


class MGHError(Exception):
    """Exception for MGH format related problems.

    To be raised whenever MGH is not happy, or we are not happy with
    MGH.
    """


class MGHHeader(object):
    ''' Class for MGH format header

    The header also consists of the footer data which MGH places after the data
    chunk.
    '''
    # Copies of module-level definitions
    template_dtype = hf_dtype
    _hdrdtype = header_dtype
    _ftrdtype = footer_dtype
    _data_type_codes = data_type_codes

    def __init__(self,
                 binaryblock=None,
                 check=True):
        ''' Initialize header from binary data block

        Parameters
        ----------
        binaryblock : {None, string} optional
            binary block to set into header.  By default, None, in
            which case we insert the default empty header block
        check : bool, optional
            Whether to check content of header in initialization.
            Default is True.
        '''
        if binaryblock is None:
            self._header_data = self._empty_headerdata()
            return
        # check size
        if len(binaryblock) != self.template_dtype.itemsize:
            raise HeaderDataError('Binary block is wrong size')
        hdr = np.ndarray(shape=(),
                         dtype=self.template_dtype,
                         buffer=binaryblock)
        # if goodRASFlag, discard delta, Mdc and c_ras stuff
        if int(hdr['goodRASFlag']) < 0:
            hdr = self._set_affine_default(hdr)
        self._header_data = hdr.copy()
        if check:
            self.check_fix()
        return

    def __str__(self):
        ''' Print the MGH header object information
        '''
        txt = []
        txt.append(str(self.__class__))
        txt.append('Dims: ' + str(self.get_data_shape()))
        code = int(self._header_data['type'])
        txt.append('MRI Type: ' + self._data_type_codes.mritype[code])
        txt.append('goodRASFlag: ' + str(self._header_data['goodRASFlag']))
        txt.append('delta: ' + str(self._header_data['delta']))
        txt.append('Mdc: ')
        txt.append(str(self._header_data['Mdc']))
        txt.append('Pxyz_c: ' + str(self._header_data['Pxyz_c']))
        txt.append('mrparms: ' + str(self._header_data['mrparms']))
        return '\n'.join(txt)

    def __getitem__(self, item):
        ''' Return values from header data
        '''
        return self._header_data[item]

    def __setitem__(self, item, value):
        ''' Set values in header data
        '''
        self._header_data[item] = value

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        ''' Return keys from header data'''
        return list(self.template_dtype.names)

    def values(self):
        ''' Return values from header data'''
        data = self._header_data
        return [data[key] for key in self.template_dtype.names]

    def items(self):
        ''' Return items from header data'''
        return zip(self.keys(), self.values())

    @classmethod
    def from_header(klass, header=None, check=True):
        ''' Class method to create MGH header from another MGH header
        '''
        # own type, return copy
        if type(header) == klass:
            obj = header.copy()
            if check:
                obj.check_fix()
            return obj
        # not own type, make fresh header instance
        obj = klass(check=check)
        return obj

    @classmethod
    def from_fileobj(klass, fileobj, check=True):
        '''
        classmethod for loading a MGH fileobject
        '''
        # We need the following hack because MGH data stores header information
        # after the data chunk too. We read the header initially, deduce the
        # dimensions from the header, skip over and then read the footer
        # information
        hdr_str = fileobj.read(klass._hdrdtype.itemsize)
        hdr_str_to_np = np.ndarray(shape=(), dtype=klass._hdrdtype,
                                   buffer=hdr_str)
        if not np.all(hdr_str_to_np['dims']):
            raise MGHError('Dimensions of the data should be non-zero')
        tp = int(hdr_str_to_np['type'])
        fileobj.seek(DATA_OFFSET +
                     int(klass._data_type_codes.bytespervox[tp]) *
                     np.prod(hdr_str_to_np['dims']))
        ftr_str = fileobj.read(klass._ftrdtype.itemsize)
        return klass(hdr_str + ftr_str, check)

    @property
    def binaryblock(self):
        ''' binary block of data as string

        Returns
        -------
        binaryblock : string
            string giving binary data block

        '''
        return self._header_data.tostring()

    def copy(self):
        ''' Return copy of header
        '''
        return self.__class__(self.binaryblock, check=False)

    def __eq__(self, other):
        ''' equality between two MGH format headers

        Examples
        --------
        >>> wstr = MGHHeader()
        >>> wstr2 = MGHHeader()
        >>> wstr == wstr2
        True
        '''
        return self.binaryblock == other.binaryblock

    def __ne__(self, other):
        return not self == other

    def check_fix(self):
        ''' Pass. maybe for now'''

    def get_affine(self):
        ''' Get the affine transform from the header information.
        MGH format doesn't store the transform directly. Instead it's gleaned
        from the zooms ( delta ), direction cosines ( Mdc ), RAS centers (
        Pxyz_c ) and the dimensions.
        '''
        hdr = self._header_data
        d = np.diag(hdr['delta'])
        pcrs_c = hdr['dims'][:3] / 2.0
        Mdc = hdr['Mdc'].T
        pxyz_0 = hdr['Pxyz_c'] - np.dot(Mdc, np.dot(d, pcrs_c))
        M = np.eye(4, 4)
        M[0:3, 0:3] = np.dot(Mdc, d)
        M[0:3, 3] = pxyz_0.T
        return M

    # For compatibility with nifti (multiple affines)
    get_best_affine = get_affine

    def get_vox2ras(self):
        '''return the get_affine()
        '''
        return self.get_affine()

    def get_vox2ras_tkr(self):
        ''' Get the vox2ras-tkr transform. See "Torig" here:
                https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        '''
        ds = np.array(self._header_data['delta'])
        ns = (np.array(self._header_data['dims'][:3]) * ds) / 2.0
        v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                           [0, 0, ds[2], -ns[2]],
                           [0, -ds[1], 0, ns[1]],
                           [0, 0, 0, 1]], dtype=np.float32)
        return v2rtkr

    def get_ras2vox(self):
        '''return the inverse get_affine()
        '''
        return np.linalg.inv(self.get_affine())

    def get_data_dtype(self):
        ''' Get numpy dtype for MGH data

        For examples see ``set_data_dtype``
        '''
        code = int(self._header_data['type'])
        dtype = self._data_type_codes.numpy_dtype[code]
        return dtype

    def set_data_dtype(self, datatype):
        ''' Set numpy dtype for data from code or dtype or type
        '''
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise MGHError('datatype dtype "%s" not recognized' % datatype)
        self._header_data['type'] = code

    def get_zooms(self):
        ''' Get zooms from header

        Returns
        -------
        z : tuple
           tuple of header zoom values
        '''
        hdr = self._header_data
        zooms = hdr['delta']
        return tuple(zooms[:])

    def set_zooms(self, zooms):
        ''' Set zooms into header fields

        See docstring for ``get_zooms`` for examples
        '''
        hdr = self._header_data
        zooms = np.asarray(zooms)
        if len(zooms) != len(hdr['delta']):
            raise HeaderDataError('Expecting %d zoom values for ndim'
                                  % hdr['delta'])
        if np.any(zooms < 0):
            raise HeaderDataError('zooms must be positive')
        delta = hdr['delta']
        delta[:] = zooms[:]

    def get_data_shape(self):
        ''' Get shape of data
        '''
        dims = self._header_data['dims'][:]
        # If last dimension (nframes) is 1, remove it because
        # we want to maintain 3D and it's redundant
        if int(dims[-1]) == 1:
            dims = dims[:-1]
        return tuple(int(d) for d in dims)

    def set_data_shape(self, shape):
        ''' Set shape of data

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        '''
        dims = self._header_data['dims']
        # If len(dims) is 3, add a dimension. MGH header always
        # needs 4 dimensions.
        if len(shape) == 3:
            shape = list(shape)
            shape.append(1)
            shape = tuple(shape)
        dims[:] = shape
        self._header_data['delta'][:] = 1.0

    def get_data_bytespervox(self):
        ''' Get the number of bytes per voxel of the data
        '''
        return int(self._data_type_codes.bytespervox[
            int(self._header_data['type'])])

    def get_data_size(self):
        ''' Get the number of bytes the data chunk occupies.
        '''
        return self.get_data_bytespervox() * np.prod(self._header_data['dims'])

    def get_data_offset(self):
        ''' Return offset into data file to read data
        '''
        return DATA_OFFSET

    def get_footer_offset(self):
        ''' Return offset where the footer resides.
            Occurs immediately after the data chunk.
        '''
        return self.get_data_offset() + self.get_data_size()

    def data_from_fileobj(self, fileobj):
        ''' Read data array from `fileobj`

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           data array
        '''
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset)

    def get_slope_inter(self):
        """ MGH format does not do scaling?
        """
        return None, None

    def _empty_headerdata(self):
        ''' Return header data for empty header
        '''
        dt = self.template_dtype
        hdr_data = np.zeros((), dtype=dt)
        hdr_data['version'] = 1
        hdr_data['dims'][:] = np.array([1, 1, 1, 1])
        hdr_data['type'] = 3
        hdr_data['goodRASFlag'] = 1
        hdr_data['delta'][:] = np.array([1, 1, 1])
        hdr_data['Mdc'][0][:] = np.array([-1, 0, 0])  # x_ras
        hdr_data['Mdc'][1][:] = np.array([0, 0, -1])  # y_ras
        hdr_data['Mdc'][2][:] = np.array([0, 1, 0])   # z_ras
        hdr_data['Pxyz_c'] = np.array([0, 0, 0])  # c_ras
        hdr_data['mrparms'] = np.array([0, 0, 0, 0])
        return hdr_data

    def _set_affine_default(self, hdr):
        ''' If  goodRASFlag is 0, return the default delta, Mdc and Pxyz_c
        '''
        hdr['delta'][:] = np.array([1, 1, 1])
        hdr['Mdc'][0][:] = np.array([-1, 0, 0])  # x_ras
        hdr['Mdc'][1][:] = np.array([0, 0, -1])  # y_ras
        hdr['Mdc'][2][:] = np.array([0, 1, 0])   # z_ras
        hdr['Pxyz_c'][:] = np.array([0, 0, 0])   # c_ras
        return hdr

    def writehdr_to(self, fileobj):
        ''' Write header to fileobj

        Write starts at the beginning.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        '''
        hdr_nofooter = np.ndarray((), dtype=self._hdrdtype,
                                  buffer=self.binaryblock)
        # goto the very beginning of the file-like obj
        fileobj.seek(0)
        fileobj.write(hdr_nofooter.tostring())

    def writeftr_to(self, fileobj):
        ''' Write footer to fileobj

        Footer data is located after the data chunk. So move there and write.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        '''
        ftr_loc_in_hdr = len(self.binaryblock) - self._ftrdtype.itemsize
        ftr_nd = np.ndarray((), dtype=self._ftrdtype,
                            buffer=self.binaryblock, offset=ftr_loc_in_hdr)
        fileobj.seek(self.get_footer_offset())
        fileobj.write(ftr_nd.tostring())


class MGHImage(SpatialImage):
    """ Class for MGH format image
    """
    header_class = MGHHeader
    valid_exts = ('.mgh', '.mgz')
    # Register that .mgz extension signals gzip compression
    ImageOpener.compress_ext_map['.mgz'] = ImageOpener.gz_def
    files_types = (('image', '.mgh'),)
    _compressed_suffixes = ()

    makeable = True
    rw = True

    ImageArrayProxy = ArrayProxy

    @classmethod
    def filespec_to_file_map(klass, filespec):
        """ Check for compressed .mgz format, then .mgh format """
        if splitext(filespec)[1].lower() == '.mgz':
            return dict(image=FileHolder(filename=filespec))
        return super(MGHImage, klass).filespec_to_file_map(filespec)

    @classmethod
    @kw_only_meth(1)
    def from_file_map(klass, file_map, mmap=True):
        '''Load image from `file_map`

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        '''
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        img_fh = file_map['image']
        mghf = img_fh.get_prepare_fileobj('rb')
        header = klass.header_class.from_fileobj(mghf)
        affine = header.get_affine()
        hdr_copy = header.copy()
        # Pass original image fileobj / filename to array proxy
        data = klass.ImageArrayProxy(img_fh.file_like, hdr_copy, mmap=mmap)
        img = klass(data, affine, header, file_map=file_map)
        img._load_cache = {'header': hdr_copy,
                           'affine': affine.copy(),
                           'file_map': copy_file_map(file_map)}
        return img

    @classmethod
    @kw_only_meth(1)
    def from_filename(klass, filename, mmap=True):
        ''' class method to create image from filename `filename`

        Parameters
        ----------
        filename : str
            Filename of image to load
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.

        Returns
        -------
        img : MGHImage instance
        '''
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        file_map = klass.filespec_to_file_map(filename)
        return klass.from_file_map(file_map, mmap=mmap)

    load = from_filename

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
        hdr = self.header
        with file_map['image'].get_prepare_fileobj('wb') as mghf:
            hdr.writehdr_to(mghf)
            self._write_data(mghf, data, hdr)
            self._write_footer(mghf, hdr)
        self._header = hdr
        self.file_map = file_map

    def _write_data(self, mghfile, data, header):
        ''' Utility routine to write image

        Parameters
        ----------
        mghfile : file-like
           file-like object implementing ``seek`` or ``tell``, and
           ``write``
        data : array-like
           array to write
        header : analyze-type header object
           header
        '''
        shape = header.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        offset = header.get_data_offset()
        out_dtype = header.get_data_dtype()
        array_to_file(data, mghfile, out_dtype, offset)

    def _write_footer(self, mghfile, header):
        ''' Utility routine to write header. This write the footer data
        which occurs after the data chunk in mgh file

        Parameters
        ----------
        mghfile : file-like
           file-like object implementing ``write``, open for writing
        header : header object
        '''
        header.writeftr_to(mghfile)

    def _affine2header(self):
        """ Unconditionally set affine into the header """
        hdr = self._header
        shape = self._dataobj.shape
        # for more information, go through save_mgh.m in FreeSurfer dist
        MdcD = self._affine[:3, :3]
        delta = np.sqrt(np.sum(MdcD * MdcD, axis=0))
        Mdc = MdcD / np.tile(delta, (3, 1))
        Pcrs_c = np.array([0, 0, 0, 1], dtype=np.float)
        Pcrs_c[:3] = np.array(shape[:3]) / 2.0
        Pxyz_c = np.dot(self._affine, Pcrs_c)

        hdr['delta'][:] = delta
        hdr['Mdc'][:, :] = Mdc.T
        hdr['Pxyz_c'][:] = Pxyz_c[:3]


load = MGHImage.load
save = MGHImage.instance_to_filename

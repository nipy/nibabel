import io
import struct
import gzip
import numpy as np
from .wrapstruct import LabeledWrapStruct
from .dataobj_images import DataobjImage
from .arrayproxy import ArrayProxy


header_dtd = [
    ('magic', 'S2'),    # 0; 0x5a4d (little endian) == "MZ"
    ('attr', 'u2'),     # 2; Attributes bitfield reporting stored data
    ('nface', 'u4'),    # 4; Number of faces
    ('nvert', 'u4'),    # 8; Number of vertices
    ('nskip', 'u4'),    # 12; Number of bytes to skip (for future header extensions)
]
header_dtype = np.dtype(header_dtd)


class MZ3Header(LabeledWrapStruct):
    template_dtype = header_dtype
    compression = False

    @classmethod
    def from_header(klass, header=None, check=True):
        if type(header) == klass:
            obj = header.copy()
            if check:
                obj.check_fix()
            return obj

    def copy(self):
        ret = super().copy()
        ret.compression = self.compression
        ret._nscalar = self._nscalar
        return ret

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        raw_str = fileobj.read(klass.template_dtype.itemsize)
        compression = raw_str[:2] == b'\x1f\x8b'
        if compression:
            fileobj.seek(0)
            with gzip.open(fileobj, 'rb') as fobj:
                raw_str = fobj.read(klass.template_dtype.itemsize)

        hdr = klass(raw_str, endianness, check)
        hdr.compression = compression
        hdr._nscalar = hdr._calculate_nscalar(fileobj)
        return hdr

    def get_data_offset(self):
        _, attr, nface, nvert, nskip = self._structarr.tolist()

        isface = attr & 1 != 0
        isvert = attr & 2 != 0
        isrgba = attr & 4 != 0
        return 16 + nskip + isface * nface * 12 + isvert * nvert * 12 + isrgba * nvert * 12

    def _calculate_nscalar(self, fileobj):
        _, attr, nface, nvert, nskip = self._structarr.tolist()

        isscalar = attr & 8 != 0
        isdouble = attr & 16 != 0
        base_size = self.get_data_offset()

        nscalar = 0
        if isscalar or isdouble:
            factor = nvert * (4 if isscalar else 8)
            ret = fileobj.tell()
            if self.compression:
                fileobj.seek(-4, 2)
                full_size_mod_4gb = struct.unpack('I', fileobj.read(4))[0]
                full_size = full_size_mod_4gb
                nscalar, remainder = divmod(full_size - base_size, factor)
                for _ in range(5):
                    full_size += (1 << 32)
                    nscalar, remainder = divmod(full_size - base_size, factor)
                    if remainder == 0:
                        break
                else:
                    fileobj.seek(0)
                    with gzip.open(fileobj, 'rb') as fobj:
                        fobj.seek(0, 2)
                        full_size = fobj.tell()
                        nscalar, remainder = divmod(full_size - base_size, factor)
                        if remainder:
                            raise ValueError("Apparent file size failure")
            else:
                fileobj.seek(0, 2)
                full_size = fileobj.tell()
                nscalar, remainder = divmod(full_size - base_size, factor)
                if remainder:
                    raise ValueError("Apparent file size failure")
            fileobj.seek(ret)
        return nscalar

    @classmethod
    def guessed_endian(klass, mapping):
        return '<'

    @classmethod
    def default_structarr(klass, endianness=None):
        if endianness is not None and endian_codes[endianness] != '<':
            raise ValueError('MZ3Header must always be little endian')
        structarr = super().default_structarr(endianness=endianness)
        structarr['magic'] = b"MZ"
        return structarr

    @classmethod
    def may_contain_header(klass, binaryblock):
        if len(binaryblock) < 16:
            return False

        # May be gzipped without changing extension
        if binaryblock[:2] == b'\x1f\x8b':
            with gzip.open(io.BytesIO(binaryblock), 'rb') as fobj:
                binaryblock = fobj.read(16)

        hdr_struct = np.ndarray(shape=(), dtype=klass.template_dtype, buffer=binaryblock[:16])
        return hdr_struct['magic'] == b'MZ'

    def get_data_dtype(self):
        if self._structarr['attr'] & 8:
            return np.dtype('<f4')
        elif self._structarr['attr'] & 16:
            return np.dtype('<f8')

    def set_data_dtype(self, datatype):
        if np.dtype(datatype).byteorder == ">":
            raise ValueError("Cannot set type to big-endian")
        dt = np.dtype(datatype).newbyteorder("<")

        if dt == np.dtype('<f8'):
            self._structarr['attr'] |= 0b00010000
        elif dt == np.dtype('<f4'):
            self._structarr['attr'] &= 0b11101111
        else:
            raise ValueError(f"Cannot set dtype: {datatype}")

    def get_data_shape(self):
        base_shape = (int(self._structarr['nvert']),)
        if self._nscalar == 0:
            return ()
        elif self._nscalar == 1:
            return base_shape
        else:
            return base_shape + (self._nscalar,)


class MZ3Image(DataobjImage):
    header_class = MZ3Header
    valid_exts = ('.mz3',)
    files_types = (('image', '.mz3'),)

    ImageArrayProxy = ArrayProxy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        fh = file_map['image']
        with fh.get_prepare_fileobj(mode='rb') as fileobj:
            header = klass.header_class.from_fileobj(fileobj)
            print(header)

        data_dtype = header.get_data_dtype()
        if data_dtype:
            spec = (header.get_data_shape(), data_dtype, header.get_data_offset())
            dataobj = klass.ImageArrayProxy(fh.filename, spec, mmap=mmap,
                                            keep_file_open=keep_file_open,
                                            compression="gz" if header.compression else None)
        else:
            dataobj = np.array((), dtype="<f4")
        
        img = klass(dataobj, header=header)
        return img

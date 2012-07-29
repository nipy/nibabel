# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Header reading / writing functions for nifti1 image format
'''
import warnings

import numpy as np
import numpy.linalg as npl

from .py3k import ZEROB, ints2bytes, asbytes, asstr
from .volumeutils import Recoder, make_dt_codes, endian_codes
from .spatialimages import HeaderDataError, ImageFileError
from .batteryrunners import Report
from .quaternions import fillpositive, quat2mat, mat2quat
from . import analyze # module import
from .spm99analyze import SpmAnalyzeHeader
from .casting import have_binary128

# Needed for quaternion calculation
FLOAT32_EPS_3 = -np.finfo(np.float32).eps * 3

# nifti1 flat header definition for Analyze-like first 348 bytes
# first number in comments indicates offset in file header in bytes
header_dtd = [
    ('sizeof_hdr', 'i4'), # 0; must be 348
    ('data_type', 'S10'), # 4; unused
    ('db_name', 'S18'),   # 14; unused
    ('extents', 'i4'),    # 32; unused
    ('session_error', 'i2'), # 36; unused
    ('regular', 'S1'),    # 38; unused
    ('dim_info', 'u1'),   # 39; MRI slice ordering code
    ('dim', 'i2', (8,)),     # 40; data array dimensions
    ('intent_p1', 'f4'),  # 56; first intent parameter
    ('intent_p2', 'f4'),  # 60; second intent parameter
    ('intent_p3', 'f4'),  # 64; third intent parameter
    ('intent_code', 'i2'),# 68; NIFTI intent code
    ('datatype', 'i2'),   # 70; it's the datatype
    ('bitpix', 'i2'),     # 72; number of bits per voxel
    ('slice_start', 'i2'),# 74; first slice index
    ('pixdim', 'f4', (8,)),  # 76; grid spacings (units below)
    ('vox_offset', 'f4'), # 108; offset to data in image file
    ('scl_slope', 'f4'),  # 112; data scaling slope
    ('scl_inter', 'f4'),  # 116; data scaling intercept
    ('slice_end', 'i2'),  # 120; last slice index
    ('slice_code', 'u1'), # 122; slice timing order
    ('xyzt_units', 'u1'), # 123; inits of pixdim[1..4]
    ('cal_max', 'f4'),    # 124; max display intensity
    ('cal_min', 'f4'),    # 128; min display intensity
    ('slice_duration', 'f4'), # 132; time for 1 slice
    ('toffset', 'f4'),   # 136; time axis shift
    ('glmax', 'i4'),     # 140; unused
    ('glmin', 'i4'),     # 144; unused
    ('descrip', 'S80'),  # 148; any text
    ('aux_file', 'S24'), # 228; auxiliary filename
    ('qform_code', 'i2'), # 252; xform code
    ('sform_code', 'i2'), # 254; xform code
    ('quatern_b', 'f4'), # 256; quaternion b param
    ('quatern_c', 'f4'), # 260; quaternion c param
    ('quatern_d', 'f4'), # 264; quaternion d param
    ('qoffset_x', 'f4'), # 268; quaternion x shift
    ('qoffset_y', 'f4'), # 272; quaternion y shift
    ('qoffset_z', 'f4'), # 276; quaternion z shift
    ('srow_x', 'f4', (4,)), # 280; 1st row affine transform
    ('srow_y', 'f4', (4,)), # 296; 2nd row affine transform
    ('srow_z', 'f4', (4,)), # 312; 3rd row affine transform
    ('intent_name', 'S16'), # 328; name or meaning of data
    ('magic', 'S4')      # 344; must be 'ni1\0' or 'n+1\0'
    ]

# Full header numpy dtype
header_dtype = np.dtype(header_dtd)

# datatypes not in analyze format, with codes
if have_binary128():
    # Only enable 128 bit floats if we really have IEEE binary 128 longdoubles
    _float128t = np.longdouble
    _complex256t = np.longcomplex
else:
    _float128t = np.void
    _complex256t = np.void

_dtdefs = ( # code, label, dtype definition, niistring
    (0, 'none', np.void, ""),
    (1, 'binary', np.void, ""),
    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),
    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),
    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),
    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),
    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),
    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),
    (128, 'RGB', np.dtype([('R','u1'),
                  ('G', 'u1'),
                  ('B', 'u1')]), "NIFTI_TYPE_RGB24"),
    (255, 'all', np.void, ''),
    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),
    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),
    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),
    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),
    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),
    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),
    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),
    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
    (2304, 'RGBA', np.dtype([('R','u1'),
                    ('G', 'u1'),
                    ('B', 'u1'),
                    ('A', 'u1')]), "NIFTI_TYPE_RGBA32"),
    )

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)

# Transform (qform, sform) codes
xform_codes = Recoder(( # code, label, niistring
                       (0, 'unknown', "NIFTI_XFORM_UNKNOWN"),
                       (1, 'scanner', "NIFTI_XFORM_SCANNER_ANAT"),
                       (2, 'aligned', "NIFTI_XFORM_ALIGNED_ANAT"),
                       (3, 'talairach', "NIFTI_XFORM_TALAIRACH"),
                       (4, 'mni', "NIFTI_XFORM_MNI_152")),
                      fields=('code', 'label', 'niistring'))

# unit codes
unit_codes = Recoder(( # code, label
    (0, 'unknown'),
    (1, 'meter'),
    (2, 'mm'),
    (3, 'micron'),
    (8, 'sec'),
    (16, 'msec'),
    (24, 'usec'),
    (32, 'hz'),
    (40, 'ppm'),
    (48, 'rads')), fields=('code', 'label'))

slice_order_codes = Recoder(( # code, label
    (0, 'unknown'),
    (1, 'sequential increasing', 'seq inc'),
    (2, 'sequential decreasing', 'seq dec'),
    (3, 'alternating increasing', 'alt inc'),
    (4, 'alternating decreasing', 'alt dec'),
    (5, 'alternating increasing 2', 'alt inc 2'),
    (6, 'alternating decreasing 2', 'alt dec 2')),
                            fields=('code', 'label'))

intent_codes = Recoder((
    # code, label, parameters description tuple
    (0, 'none', (), "NIFTI_INTENT_NONE"),
    (2, 'correlation',('p1 = DOF',), "NIFTI_INTENT_CORREL"),
    (3, 't test', ('p1 = DOF',), "NIFTI_INTENT_TTEST"),
    (4, 'f test',
     ('p1 = numerator DOF', 'p2 = denominator DOF'),
     "NIFTI_INTENT_FTEST"),
    (5, 'z score', (), "NIFTI_INTENT_ZSCORE"),
    (6, 'chi2', ('p1 = DOF',), "NIFTI_INTENT_CHISQ"),
    # two parameter beta distribution
    (7, 'beta',
     ('p1=a', 'p2=b'),
     "NIFTI_INTENT_BETA"),
    # Prob(x) = (p1 choose x) * p2^x * (1-p2)^(p1-x), for x=0,1,...,p1
    (8, 'binomial',
     ('p1 = number of trials', 'p2 = probability per trial'),
     "NIFTI_INTENT_BINOM"),
    # 2 parameter gamma
    # Density(x) proportional to # x^(p1-1) * exp(-p2*x)
    (9, 'gamma',
     ('p1 = shape, p2 = scale', 2),
     "NIFTI_INTENT_GAMMA"),
    (10, 'poisson',
     ('p1 = mean',),
     "NIFTI_INTENT_POISSON"),
    (11, 'normal',
     ('p1 = mean', 'p2 = standard deviation',),
     "NIFTI_INTENT_NORMAL"),
    (12, 'non central f test',
     ('p1 = numerator DOF',
      'p2 = denominator DOF',
      'p3 = numerator noncentrality parameter',),
     "NIFTI_INTENT_FTEST_NONC"),
    (13, 'non central chi2',
     ('p1 = DOF', 'p2 = noncentrality parameter',), 
     "NIFTI_INTENT_CHISQ_NONC"),
    (14, 'logistic',
     ('p1 = location', 'p2 = scale',),
     "NIFTI_INTENT_LOGISTIC"),
    (15, 'laplace',
     ('p1 = location', 'p2 = scale'),
     "NIFTI_INTENT_LAPLACE"),
    (16, 'uniform',
     ('p1 = lower end', 'p2 = upper end'),
     "NIFTI_INTENT_UNIFORM"),
    (17, 'non central t test',
     ('p1 = DOF', 'p2 = noncentrality parameter'),
     "NIFTI_INTENT_TTEST_NONC"),
    (18, 'weibull',
     ('p1 = location', 'p2 = scale, p3 = power'),
     "NIFTI_INTENT_WEIBULL"),
    # p1 = 1 = 'half normal' distribution
    # p1 = 2 = Rayleigh distribution
    # p1 = 3 = Maxwell-Boltzmann distribution.
    (19, 'chi', ('p1 = DOF',), "NIFTI_INTENT_CHI"),
    (20, 'inverse gaussian',
     ('pi = mu', 'p2 = lambda'),
     "NIFTI_INTENT_INVGAUSS"),
    (21, 'extreme value 1',
     ('p1 = location', 'p2 = scale'),
     "NIFTI_INTENT_EXTVAL"),
    (22, 'p value', (), "NIFTI_INTENT_PVAL"),
    (23, 'log p value', (), "NIFTI_INTENT_LOGPVAL"),
    (24, 'log10 p value', (), "NIFTI_INTENT_LOG10PVAL"),
    (1001, 'estimate', (), "NIFTI_INTENT_ESTIMATE"),
    (1002, 'label', (), "NIFTI_INTENT_LABEL"),
    (1003, 'neuroname', (), "NIFTI_INTENT_NEURONAME"),
    (1004, 'general matrix',
     ('p1 = M', 'p2 = N'),
     "NIFTI_INTENT_GENMATRIX"),
    (1005, 'symmetric matrix', ('p1 = M',), "NIFTI_INTENT_SYMMATRIX"),
    (1006, 'displacement vector', (), "NIFTI_INTENT_DISPVECT"),
    (1007, 'vector', (), "NIFTI_INTENT_VECTOR"),
    (1008, 'pointset', (), "NIFTI_INTENT_POINTSET"),
    (1009, 'triangle', (), "NIFTI_INTENT_TRIANGLE"),
    (1010, 'quaternion', (), "NIFTI_INTENT_QUATERNION"),
    (1011, 'dimensionless', (), "NIFTI_INTENT_DIMLESS"),
    (2001, 'time series',
     (),
     "NIFTI_INTENT_TIME_SERIES",
     "NIFTI_INTENT_TIMESERIES"), # this mis-spell occurs in the wild
    (2002, 'node index', (), "NIFTI_INTENT_NODE_INDEX"),
    (2003, 'rgb vector', (), "NIFTI_INTENT_RGB_VECTOR"),
    (2004, 'rgba vector', (), "NIFTI_INTENT_RGBA_VECTOR"),
    (2005, 'shape', (), "NIFTI_INTENT_SHAPE")),
    fields=('code', 'label', 'parameters', 'niistring'))


class Nifti1Extension(object):
    """Baseclass for NIfTI1 header extensions.

    This class is sufficient to handle very simple text-based extensions, such
    as `comment`. More sophisticated extensions should/will be supported by
    dedicated subclasses.
    """
    def __init__(self, code, content):
        """
        Parameters
        ----------
        code : int|str
          Canonical extension code as defined in the NIfTI standard, given
          either as integer or corresponding label
          (see :data:`~nibabel.nifti1.extension_codes`)
        content : str
          Extension content as read from the NIfTI file header. This content is
          converted into a runtime representation.
        """
        try:
            self._code = extension_codes.code[code]
        except KeyError:
            # XXX or fail or at least complain?
            self._code = code
        self._content = self._unmangle(content)

    def _unmangle(self, value):
        """Convert the extension content into its runtime representation.

        The default implementation does nothing at all.

        Parameters
        ----------
        value : str
          Extension content as read from file.

        Returns
        -------
        The same object that was passed as `value`.

        Notes
        -----
        Subclasses should reimplement this method to provide the desired
        unmangling procedure and may return any type of object.
        """
        return value

    def _mangle(self, value):
        """Convert the extension content into NIfTI file header representation.

        The default implementation does nothing at all.

        Parameters
        ----------
        value : str
          Extension content in runtime form.

        Returns
        -------
        str

        Notes
        -----
        Subclasses should reimplement this method to provide the desired
        mangling procedure.
        """
        return value

    def get_code(self):
        """Return the canonical extension type code."""
        return self._code

    def get_content(self):
        """Return the extension content in its runtime representation."""
        return self._content

    def get_sizeondisk(self):
        """Return the size of the extension in the NIfTI file.
        """
        # need raw value size plus 8 bytes for esize and ecode
        size = len(self._mangle(self._content))
        size += 8
        # extensions size has to be a multiple of 16 bytes
        size += 16 - (size % 16)
        return size

    def __repr__(self):
        try:
            code = extension_codes.label[self._code]
        except KeyError:
            # deal with unknown codes
            code = self._code

        s = "Nifti1Extension('%s', '%s')" % (code, self._content)
        return s

    def __eq__(self, other):
        return (self._code, self._content) == (other._code, other._content)

    def __ne__(self, other):
        return not self == other

    def write_to(self, fileobj, byteswap):
        ''' Write header extensions to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method
        byteswap : boolean
          Flag if byteswapping the data is required.

        Returns
        -------
        None
        '''
        extstart = fileobj.tell()
        rawsize = self.get_sizeondisk()
        # write esize and ecode first
        extinfo = np.array((rawsize, self._code), dtype=np.int32)
        if byteswap:
            extinfo = extinfo.byteswap()
        fileobj.write(extinfo.tostring())
        # followed by the actual extension content
        # XXX if mangling upon load is implemented, it should be reverted here
        fileobj.write(self._mangle(self._content))
        # be nice and zero out remaining part of the extension till the
        # next 16 byte border
        fileobj.write(ZEROB * (extstart + rawsize - fileobj.tell()))


# NIfTI header extension type codes (ECODE)
# see nifti1_io.h for a complete list of all known extensions and
# references to their description or contacts of the respective
# initiators
extension_codes = Recoder((
    (0, "ignore", Nifti1Extension),
    (2, "dicom", Nifti1Extension),
    (4, "afni", Nifti1Extension),
    (6, "comment", Nifti1Extension),
    (8, "xcede", Nifti1Extension),
    (10, "jimdiminfo", Nifti1Extension),
    (12, "workflow_fwds", Nifti1Extension),
    (14, "freesurfer", Nifti1Extension),
    (16, "pypickle", Nifti1Extension)
    ),
    fields=('code', 'label', 'handler'))


class Nifti1Extensions(list):
    """Simple extension collection, implemented as a list-subclass.
    """
    def count(self, ecode):
        """Returns the number of extensions matching a given *ecode*.

        Parameters
        ----------
          code : int | str
            The ecode can be specified either literal or as numerical value.
        """
        count = 0
        code = extension_codes.code[ecode]
        for e in self:
            if e.get_code() == code:
                count += 1
        return count

    def get_codes(self):
        """Return a list of the extension code of all available extensions"""
        return [e.get_code() for e in self]

    def get_sizeondisk(self):
        """Return the size of the complete header extensions in the NIfTI file.
        """
        return np.sum([e.get_sizeondisk() for e in self])

    def __repr__(self):
        s = "Nifti1Extensions(%s)" \
                % ', '.join([str(e) for e in self])
        return s

    def __cmp__(self, other):
        return cmp(list(self), list(other))

    def write_to(self, fileobj, byteswap):
        ''' Write header extensions to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method
        byteswap : boolean
          Flag if byteswapping the data is required.

        Returns
        -------
        None
        '''
        for e in self:
            e.write_to(fileobj, byteswap)

    @classmethod
    def from_fileobj(klass, fileobj, size, byteswap):
        '''Read header extensions from a fileobj

        Parameters
        ----------
        fileobj : file-like object
            We begin reading the extensions at the current file position
        size : int
            Number of bytes to read. If negative, fileobj will be read till its
            end.
        byteswap : boolean
            Flag if byteswapping the read data is required.

        Returns
        -------
        An extension list. This list might be empty in case not extensions
        were present in fileobj.
        '''
        # make empty extension list
        extensions = klass()
        # assume the file pointer is at the beginning of any extensions.
        # read until the whole header is parsed (each extension is a multiple
        # of 16 bytes) or in case of a separate header file till the end
        # (break inside the body)
        while size >= 16 or size < 0:
            # the next 8 bytes should have esize and ecode
            ext_def = fileobj.read(8)
            # nothing was read and instructed to read till the end
            # -> assume all extensions where parsed and break
            if not len(ext_def) and size < 0:
                break
            # otherwise there should be a full extension header
            if not len(ext_def) == 8:
                raise HeaderDataError('failed to read extension header')
            ext_def = np.fromstring(ext_def, dtype=np.int32)
            if byteswap:
                ext_def = ext_def.byteswap()
            # be extra verbose
            ecode = ext_def[1]
            esize = ext_def[0]
            if esize % 16:
                raise HeaderDataError(
                        'extension size is not a multiple of 16 bytes')
            # read extension itself; esize includes the 8 bytes already read
            evalue = fileobj.read(int(esize - 8))
            if not len(evalue) == esize - 8:
                raise HeaderDataError('failed to read extension content')
            # note that we read a full extension
            size -= esize
            # store raw extension content, but strip trailing NULL chars
            evalue = evalue.rstrip(ZEROB)
            # 'extension_codes' also knows the best implementation to handle
            # a particular extension type
            try:
                ext = extension_codes.handler[ecode](ecode, evalue)
            except KeyError:
                # unknown extension type
                # XXX complain or fail or go with a generic extension
                ext = Nifti1Extension(ecode, evalue)
            extensions.append(ext)
        return extensions


class Nifti1Header(SpmAnalyzeHeader):
    ''' Class for NIFTI1 header

    The NIFTI1 header has many more coded fields than the simpler Analyze
    variants.  Nifti1 headers also have extensions.

    Nifti allows the header to be a separate file, as part of a nifti image /
    header pair, or to precede the data in a single file.  The object needs to
    know which type it is, in order to manage the voxel offset pointing to the
    data, extension reading, and writing the correct magic string.

    This class handles the header-preceding-data case.
    '''
    # Copies of module level definitions
    template_dtype = header_dtype
    _data_type_codes = data_type_codes

    # fields with recoders for their values
    _field_recoders = {'datatype': data_type_codes,
                       'qform_code': xform_codes,
                       'sform_code': xform_codes,
                       'intent_code': intent_codes,
                       'slice_code': slice_order_codes}

    # data scaling capabilities
    has_data_slope = True
    has_data_intercept = True

    # Extension class; should implement __call__ for contruction, and
    # ``from_fileobj`` for reading from file
    exts_klass = Nifti1Extensions

    # Signal whether this is single (header + data) file
    is_single = True

    def __init__(self,
                 binaryblock=None,
                 endianness=None,
                 check=True,
                 extensions=()):
        ''' Initialize header from binary data block and extensions
        '''
        super(Nifti1Header, self).__init__(binaryblock,
                                           endianness,
                                           check)
        self.extensions = self.exts_klass(extensions)

    def copy(self):
        ''' Return copy of header

        Take reference to extensions as well as copy of header contents
        '''
        return self.__class__(
            self.binaryblock,
            self.endianness, 
            False,
            self.extensions)

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        raw_str = fileobj.read(klass.template_dtype.itemsize)
        hdr = klass(raw_str, endianness, check)
        # Read next 4 bytes to see if we have extensions.  The nifti standard
        # has this as a 4 byte string; if the first value is not zero, then we
        # have extensions.  
        extension_status = fileobj.read(4)
        if len(extension_status) < 4 or extension_status[0] == ZEROB:
            return hdr
        # If this is a detached header file read to end
        if not klass.is_single:
            extsize = -1
        else: # otherwise read until the beginning of the data
            extsize = hdr._structarr['vox_offset'] - fileobj.tell()
        byteswap = endian_codes['native'] != hdr.endianness
        hdr.extensions = klass.exts_klass.from_fileobj(fileobj, extsize, byteswap)
        return hdr

    def write_to(self, fileobj):
        # First check that vox offset is large enough
        if self.is_single:
            vox_offset = self._structarr['vox_offset']
            min_vox_offset = 352 + self.extensions.get_sizeondisk()
            if vox_offset < min_vox_offset:
                raise HeaderDataError('vox offset of %d, but need at least %d'
                                      % (vox_offset, min_vox_offset))
        super(Nifti1Header, self).write_to(fileobj)
        if len(self.extensions) == 0:
            # If single file, write required 0 stream to signal no extensions
            if self.is_single:
                fileobj.write(ZEROB * 4)
            return
        # Signal there are extensions that follow
        fileobj.write(ints2bytes([1, 0, 0, 0]))
        byteswap = endian_codes['native'] != self.endianness
        self.extensions.write_to(fileobj, byteswap)

    def get_best_affine(self):
        ''' Select best of available transforms '''
        hdr = self._structarr
        if hdr['sform_code'] != 0:
            return self.get_sform()
        if hdr['qform_code'] != 0:
            return self.get_qform()
        return self.get_base_affine()

    @classmethod
    def default_structarr(klass, endianness=None):
        ''' Create empty header binary block with given endianness '''
        hdr_data = super(Nifti1Header, klass).default_structarr(endianness)
        if klass.is_single:
            hdr_data['magic'] = 'n+1'
            hdr_data['vox_offset'] = 352
        else:
            hdr_data['magic'] = 'ni1'
            hdr_data['vox_offset'] = 0
        return hdr_data

    def get_data_shape(self):
        ''' Get shape of data

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.get_data_shape()
        (0,)
        >>> hdr.set_data_shape((1,2,3))
        >>> hdr.get_data_shape()
        (1, 2, 3)

        Expanding number of dimensions gets default zooms

        >>> hdr.get_zooms()
        (1.0, 1.0, 1.0)

        Notes
        -----
        Allows for freesurfer hack for large vectors described in
        https://github.com/nipy/nibabel/issues/100 and
        http://code.google.com/p/fieldtrip/source/browse/trunk/external/freesurfer/save_nifti.m?spec=svn5022&r=5022#77
        '''
        shape = super(Nifti1Header, self).get_data_shape()
        # Apply freesurfer hack for vector
        if shape != (-1, 1, 1): # Normal case
            return shape
        vec_len = int(self._structarr['glmin'])
        if vec_len == 0:
            raise HeaderDataError('-1 in dim[1] but 0 in glmin; inconsistent '
                                  'freesurfer type header?')
        return (vec_len, 1, 1)

    def set_data_shape(self, shape):
        ''' Set shape of data

        If ``ndims == len(shape)`` then we set zooms for dimensions higher than
        ``ndims`` to 1.0

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape

        Notes
        -----
        Applies freesurfer hack for large vectors described in
        https://github.com/nipy/nibabel/issues/100 and
        http://code.google.com/p/fieldtrip/source/browse/trunk/external/freesurfer/save_nifti.m?spec=svn5022&r=5022#77
        '''
        # Apply freesurfer hack for vector
        hdr = self._structarr
        shape = tuple(shape)
        if (len(shape) == 3 and shape[1:] == (1, 1) and
            shape[0] > np.iinfo(hdr['dim'].dtype.base).max): # Freesurfer case
            try:
                hdr['glmin'] = shape[0]
            except OverflowError:
                overflow = True
            else:
                overflow = hdr['glmin'] != shape[0]
            if overflow:
                raise HeaderDataError('shape[0] %s does not fit in glmax datatype' %
                                      shape[0])
            warnings.warn('Using large vector Freesurfer hack; header will '
                          'not be compatible with SPM or FSL', stacklevel=2)
            shape = (-1, 1, 1)
        super(Nifti1Header, self).set_data_shape(shape)

    def get_qform_quaternion(self):
        ''' Compute quaternion from b, c, d of quaternion

        Fills a value by assuming this is a unit quaternion
        '''
        hdr = self._structarr
        bcd = [hdr['quatern_b'], hdr['quatern_c'], hdr['quatern_d']]
        # Adjust threshold to fact that source data was float32
        return fillpositive(bcd, FLOAT32_EPS_3)

    def get_qform(self, coded=False):
        """ Return 4x4 affine matrix from qform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and qform code.  If False, just
            return affine.  {affine or None} means, return None if qform code ==
            0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine reconstructed from qform
            quaternion.  If `coded` is True, return None if qform code is 0,
            else return the affine.
        code : int
            Qform code. Only returned if `coded` is True.
        """
        hdr = self._structarr
        code = hdr['qform_code']
        if code == 0 and coded:
            return None, 0
        quat = self.get_qform_quaternion()
        R = quat2mat(quat)
        vox = hdr['pixdim'][1:4].copy()
        if np.any(vox) < 0:
            raise HeaderDataError('pixdims[1,2,3] should be positive')
        qfac = hdr['pixdim'][0]
        if qfac not in (-1, 1):
            raise HeaderDataError('qfac (pixdim[0]) should be 1 or -1')
        vox[-1] *= qfac
        S = np.diag(vox)
        M = np.dot(R, S)
        out = np.eye(4)
        out[0:3, 0:3] = M
        out[0:3, 3] = [hdr['qoffset_x'], hdr['qoffset_y'], hdr['qoffset_z']]
        if coded:
            return out, code
        return out

    def set_qform(self, affine, code=None, strip_shears=True):
        ''' Set qform header values from 4x4 affine

        Parameters
        ----------
        hdr : nifti1 header
        affine : None or 4x4 array
            affine transform to write into sform. If None, only set code.
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:
            * If affine is None, `code`-> 0
            * If affine not None and sform code in header == 0, `code`-> 2
              (aligned)
            * If affine not None and sform code in header != 0, `code`-> sform
              code in header
        strip_shears : bool, optional
            Whether to strip shears in `affine`.  If True, shears will be
            silently stripped. If False, the presence of shears will raise a
            ``HeaderDataError``

        Notes
        -----
        The qform transform only encodes translations, rotations and
        zooms. If there are shear components to the `affine` transform, and
        `strip_shears` is True (the default), the written qform gives the
        closest approximation where the rotation matrix is orthogonal. This is
        to allow quaternion representation. The orthogonal representation
        enforces orthogonal axes.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> int(hdr['qform_code']) # gives 0 - unknown
        0
        >>> affine = np.diag([1,2,3,1])
        >>> np.all(hdr.get_qform() == affine)
        False
        >>> hdr.set_qform(affine)
        >>> np.all(hdr.get_qform() == affine)
        True
        >>> int(hdr['qform_code']) # gives 2 - aligned
        2
        >>> hdr.set_qform(affine, code='talairach')
        >>> int(hdr['qform_code'])
        3
        >>> hdr.set_qform(affine, code=None)
        >>> int(hdr['qform_code'])
        3
        >>> hdr.set_qform(affine, code='scanner')
        >>> int(hdr['qform_code'])
        1
        >>> hdr.set_qform(None)
        >>> int(hdr['qform_code'])
        0
        '''
        hdr = self._structarr
        old_code = hdr['qform_code']
        if code is None:
            if affine is None:
                code = 0
            elif old_code == 0:
                code = 2 # aligned
            else:
                code = old_code
        else: # code set
            code = self._field_recoders['qform_code'][code]
        hdr['qform_code'] = code
        if affine is None:
            return
        affine = np.asarray(affine)
        if not affine.shape == (4, 4):
            raise TypeError('Need 4x4 affine as input')
        trans = affine[:3, 3]
        RZS = affine[:3, :3]
        zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
        R = RZS / zooms
        # Set qfac to make R determinant positive
        if npl.det(R) > 0:
            qfac = 1
        else:
            qfac = -1
            R[:, -1] *= -1
        # Make R orthogonal (to allow quaternion representation)
        # The orthogonal representation enforces orthogonal axes
        # (a subtle requirement of the NIFTI format qform transform)
        # Transform below is polar decomposition, returning the closest
        # orthogonal matrix PR, to input R
        P, S, Qs = npl.svd(R)
        PR = np.dot(P, Qs)
        if not strip_shears and not np.allclose(PR, R):
            raise HeaderDataError("Shears in affine and `strip_shears` is "
                                  "False")
        # Convert to quaternion
        quat = mat2quat(PR)
        # Set into header
        hdr['qoffset_x'], hdr['qoffset_y'], hdr['qoffset_z'] = trans
        hdr['pixdim'][0] = qfac
        hdr['pixdim'][1:4] = zooms
        hdr['quatern_b'], hdr['quatern_c'], hdr['quatern_d'] = quat[1:]

    def get_sform(self, coded=False):
        """ Return 4x4 affine matrix from sform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and sform code.  If False, just
            return affine.  {affine or None} means, return None if sform code ==
            0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine from sform fields. If
            `coded` is True, return None if sform code is 0, else return the
            affine.
        code : int
            Sform code. Only returned if `coded` is True.
        """
        hdr = self._structarr
        code = hdr['sform_code']
        if code == 0 and coded:
            return None, 0
        out = np.eye(4)
        out[0, :] = hdr['srow_x'][:]
        out[1, :] = hdr['srow_y'][:]
        out[2, :] = hdr['srow_z'][:]
        if coded:
            return out, code
        return out

    def set_sform(self, affine, code=None):
        ''' Set sform transform from 4x4 affine

        Parameters
        ----------
        hdr : nifti1 header
        affine : None or 4x4 array
            affine transform to write into sform.  If None, only set `code`
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:
            * If affine is None, `code`-> 0
            * If affine not None and sform code in header == 0, `code`-> 2
              (aligned)
            * If affine not None and sform code in header != 0, `code`-> sform
              code in header

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> int(hdr['sform_code']) # gives 0 - unknown
        0
        >>> affine = np.diag([1,2,3,1])
        >>> np.all(hdr.get_sform() == affine)
        False
        >>> hdr.set_sform(affine)
        >>> np.all(hdr.get_sform() == affine)
        True
        >>> int(hdr['sform_code']) # gives 2 - aligned
        2
        >>> hdr.set_sform(affine, code='talairach')
        >>> int(hdr['sform_code'])
        3
        >>> hdr.set_sform(affine, code=None)
        >>> int(hdr['sform_code'])
        3
        >>> hdr.set_sform(affine, code='scanner')
        >>> int(hdr['sform_code'])
        1
        >>> hdr.set_sform(None)
        >>> int(hdr['sform_code'])
        0
        '''
        hdr = self._structarr
        old_code = hdr['sform_code']
        if code is None:
            if affine is None:
                code = 0
            elif old_code == 0:
                code = 2 # aligned
            else:
                code = old_code
        else: # code set
            code = self._field_recoders['sform_code'][code]
        hdr['sform_code'] = code
        if affine is None:
            return
        affine = np.asarray(affine)
        hdr['srow_x'][:] = affine[0, :]
        hdr['srow_y'][:] = affine[1, :]
        hdr['srow_z'][:] = affine[2, :]

    def get_slope_inter(self):
        ''' Get data scaling (slope) and DC offset (intercept) from header data

        Parameters
        ----------
        self : header object
           Should have fields (keys)
           * scl_slope - slope
           * scl_inter - intercept

        Returns
        -------
        slope : None or float
           scaling (slope).  None if there is no valid scaling from these fields
        inter : None or float
           offset (intercept). None if there is no valid scaling or if offset is
           not finite.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.get_slope_inter()
        (1.0, 0.0)
        >>> hdr['scl_slope'] = 0
        >>> hdr.get_slope_inter()
        (None, None)
        >>> hdr['scl_slope'] = np.nan
        >>> hdr.get_slope_inter()
        (None, None)
        >>> hdr['scl_slope'] = 1
        >>> hdr['scl_inter'] = 1
        >>> hdr.get_slope_inter()
        (1.0, 1.0)
        >>> hdr['scl_inter'] = np.inf
        >>> hdr.get_slope_inter()
        (1.0, None)
        '''
        # Note that we are returning float (float64) scalefactors and
        # intercepts, although they are stored as np.float32.
        scale = float(self['scl_slope'])
        dc_offset = float(self['scl_inter'])
        if scale == 0 or not np.isfinite(scale):
            return None, None
        if not np.isfinite(dc_offset):
            dc_offset = None
        return scale, dc_offset

    def set_slope_inter(self, slope, inter=0.0):
        ''' Set slope and / or intercept into header

        Set slope and intercept for image data, such that, if the image
        data is ``arr``, then the scaled image data will be ``(arr *
        slope) + inter``

        Parameters
        ----------
        slope : None or float
           If None, implies `slope`  of 0. When the slope is set to 0 or a
           not-finite value, ``get_slope_inter`` returns (None, None), i.e.
           `inter` is ignored unless there is a valid value for `slope`.
        inter : None or float, optional
           intercept.  None implies inter value of 0.
        '''
        if slope is None:
            slope = 0.0
        if inter is None:
            inter = 0.0
        self._structarr['scl_slope'] = slope
        self._structarr['scl_inter'] = inter

    def get_dim_info(self):
        ''' Gets nifti MRI slice etc dimension information

        Returns
        -------
        freq : {None,0,1,2}
           Which data array axis is freqency encode direction
        phase : {None,0,1,2}
           Which data array axis is phase encode direction
        slice : {None,0,1,2}
           Which data array axis is slice encode direction

        where ``data array`` is the array returned by ``get_data``

        Because nifti1 files are natively Fortran indexed:
          0 is fastest changing in file
          1 is medium changing in file
          2 is slowest changing in file

        ``None`` means the axis appears not to be specified.

        Examples
        --------
        See set_dim_info function

        '''
        hdr = self._structarr
        info = int(hdr['dim_info'])
        freq = info & 3
        phase = (info >> 2) & 3
        slice = (info >> 4) & 3
        return (freq-1 if freq else None,
            phase-1 if phase else None,
            slice-1 if slice else None)

    def set_dim_info(self, freq=None, phase=None, slice=None):
        ''' Sets nifti MRI slice etc dimension information

        Parameters
        ----------
        hdr : nifti1 header
        freq : {None, 0, 1, 2}
            axis of data array refering to freqency encoding
        phase : {None, 0, 1, 2}
            axis of data array refering to phase encoding
        slice : {None, 0, 1, 2}
            axis of data array refering to slice encoding

        ``None`` means the axis is not specified.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(1, 2, 0)
        >>> hdr.get_dim_info()
        (1, 2, 0)
        >>> hdr.set_dim_info(freq=1, phase=2, slice=0)
        >>> hdr.get_dim_info()
        (1, 2, 0)
        >>> hdr.set_dim_info()
        >>> hdr.get_dim_info()
        (None, None, None)
        >>> hdr.set_dim_info(freq=1, phase=None, slice=0)
        >>> hdr.get_dim_info()
        (1, None, 0)

        Notes
        -----
        This is stored in one byte in the header
        '''
        for inp in (freq, phase, slice):
            if inp not in (None, 0, 1, 2):
                raise HeaderDataError('Inputs must be in [None, 0, 1, 2]')
        info = 0
        if not freq is None:
            info = info | ((freq+1) & 3)
        if not phase is None:
            info = info | (((phase+1) & 3) << 2)
        if not slice is None:
            info = info | (((slice+1) & 3) << 4)
        self._structarr['dim_info'] = info

    def get_intent(self, code_repr='label'):
        ''' Get intent code, parameters and name

        Parameters
        ----------
        code_repr : string
           string giving output form of intent code representation.
           Default is 'label'; use 'code' for integer representation.

        Returns
        -------
        code : string or integer
            intent code, or string describing code
        parameters : tuple
            parameters for the intent
        name : string
            intent name

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_intent('t test', (10,), name='some score')
        >>> hdr.get_intent()
        ('t test', (10.0,), 'some score')
        >>> hdr.get_intent('code')
        (3, (10.0,), 'some score')
        '''
        hdr = self._structarr
        recoder = self._field_recoders['intent_code']
        code = int(hdr['intent_code'])
        if code_repr == 'code':
            label = code
        elif code_repr == 'label':
            label = recoder.label[code]
        else:
            raise TypeError('repr can be "label" or "code"')
        n_params = len(recoder.parameters[code])
        params = (float(hdr['intent_p%d' % (i+1)]) for i in range(n_params))
        name = asstr(np.asscalar(hdr['intent_name']))
        return label, tuple(params), name

    def set_intent(self, code, params=(), name=''):
        ''' Set the intent code, parameters and name

        If parameters are not specified, assumed to be all zero. Each
        intent code has a set number of parameters associated. If you
        specify any parameters, then it will need to be the correct number
        (e.g the "f test" intent requires 2).  However, parameters can
        also be set in the file data, so we also allow not setting any
        parameters (empty parameter tuple).

        Parameters
        ----------
        code : integer or string
            code specifying nifti intent
        params : list, tuple of scalars
            parameters relating to intent (see intent_codes)
            defaults to ().  Unspecified parameters are set to 0.0
        name : string
            intent name (description). Defaults to ''

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_intent(0) # unknown code
        >>> hdr.set_intent('z score')
        >>> hdr.get_intent()
        ('z score', (), '')
        >>> hdr.get_intent('code')
        (5, (), '')
        >>> hdr.set_intent('t test', (10,), name='some score')
        >>> hdr.get_intent()
        ('t test', (10.0,), 'some score')
        >>> hdr.set_intent('f test', (2, 10), name='another score')
        >>> hdr.get_intent()
        ('f test', (2.0, 10.0), 'another score')
        >>> hdr.set_intent('f test')
        >>> hdr.get_intent()
        ('f test', (0.0, 0.0), '')
        '''
        hdr = self._structarr
        icode = intent_codes.code[code]
        p_descr = intent_codes.parameters[code]
        if len(params) and len(params) != len(p_descr):
            raise HeaderDataError('Need params of form %s, or empty'
                                  % (p_descr,))
        all_params = [0] * 3
        all_params[:len(params)] = params[:]
        for i, param in enumerate(all_params):
            hdr['intent_p%d' % (i+1)] = param
        hdr['intent_code'] = icode
        hdr['intent_name'] = name

    def get_slice_duration(self):
        ''' Get slice duration

        Returns
        -------
        slice_duration : float
            time to acquire one slice

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_slice_duration(0.3)
        >>> print "%0.1f" % hdr.get_slice_duration()
        0.3

        Notes
        -----
        The Nifti1 spec appears to require the slice dimension to be
        defined for slice_duration to have meaning.
        '''
        _, _, slice_dim = self.get_dim_info()
        if slice_dim is None:
            raise HeaderDataError('Slice dimension must be set '
                                  'for duration to be valid')
        return float(self._structarr['slice_duration'])

    def set_slice_duration(self, duration):
        ''' Set slice duration

        Parameters
        ----------
        duration : scalar
            time to acquire one slice

        Examples
        --------
        See ``get_slice_duration``
        '''
        _, _, slice_dim = self.get_dim_info()
        if slice_dim is None:
            raise HeaderDataError('Slice dimension must be set '
                                  'for duration to be valid')
        self._structarr['slice_duration'] = duration

    def get_n_slices(self):
        ''' Return the number of slices
        '''
        hdr = self._structarr
        _, _, slice_dim = self.get_dim_info()
        if slice_dim is None:
            raise HeaderDataError('Slice dimension not set in header '
                                  'dim_info')
        shape = self.get_data_shape()
        try:
            slice_len = shape[slice_dim]
        except IndexError:
            raise HeaderDataError('Slice dimension index (%s) outside '
                                  'shape tuple (%s)'
                                  % (slice_dim, shape))
        return slice_len

    def get_slice_times(self):
        ''' Get slice times from slice timing information

        Returns
        -------
        slice_times : tuple
            Times of acquisition of slices, where 0 is the beginning of
            the acquisition, ordered by position in file.  nifti allows
            slices at the top and bottom of the volume to be excluded from
            the standard slice timing specification, and calls these
            "padding slices".  We give padding slices ``None`` as a time
            of acquisition

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_data_shape((1, 1, 7))
        >>> hdr.set_slice_duration(0.1)
        >>> hdr['slice_code'] = slice_order_codes['sequential increasing']
        >>> slice_times = hdr.get_slice_times()
        >>> np.allclose(slice_times, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        True
        '''
        hdr = self._structarr
        slice_len = self.get_n_slices()
        duration = self.get_slice_duration()
        slabel = self.get_value_label('slice_code')
        if slabel == 'unknown':
            raise HeaderDataError('Cannot get slice times when '
                                  'Slice code is "unknown"')
        slice_start, slice_end = (int(hdr['slice_start']),
                                  int(hdr['slice_end']))
        if slice_start < 0:
            raise HeaderDataError('slice_start should be >= 0')
        if slice_end == 0:
            slice_end = slice_len-1
        n_timed = slice_end - slice_start + 1
        if n_timed < 1:
            raise HeaderDataError('slice_end should be > slice_start')
        st_order = self._slice_time_order(slabel, n_timed)
        times = st_order * duration
        return ((None,)*slice_start +
                tuple(times) +
                (None,)*(slice_len-slice_end-1))

    def set_slice_times(self, slice_times):
        ''' Set slice times into *hdr*

        Parameters
        ----------
        slice_times : tuple
            tuple of slice times, one value per slice
            tuple can include None to indicate no slice time for that slice

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_data_shape([1, 1, 7])
        >>> hdr.set_slice_duration(0.1)
        >>> times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
        >>> hdr.set_slice_times(times)
        >>> hdr.get_value_label('slice_code')
        'alternating decreasing'
        >>> int(hdr['slice_start'])
        1
        >>> int(hdr['slice_end'])
        5
        '''
        # Check if number of slices matches header
        hdr = self._structarr
        slice_len = self.get_n_slices()
        if slice_len != len(slice_times):
            raise HeaderDataError('Number of slice times does not '
                                  'match number of slices')
        # Extract Nones at beginning and end.  Check for others
        for ind, time in enumerate(slice_times):
            if time is not None:
                slice_start = ind
                break
        else:
            raise HeaderDataError('Not all slice times can be None')
        for ind, time in enumerate(slice_times[::-1]):
            if time is not None:
                slice_end = slice_len-ind-1
                break
        timed = slice_times[slice_start:slice_end+1]
        for time in timed:
            if time is None:
                raise HeaderDataError('Cannot have None in middle '
                                      'of slice time vector')
        # Find slice duration, check times are compatible with single
        # duration
        tdiffs = np.diff(np.sort(timed))
        if not np.allclose(np.diff(tdiffs), 0):
            raise HeaderDataError('Slice times not compatible with '
                                  'single slice duration')
        duration = np.mean(tdiffs)
        # To slice time order
        st_order = np.round(np.array(timed) / duration)
        # Check if slice times fit known schemes
        n_timed = len(timed)
        so_recoder = self._field_recoders['slice_code']
        labels = so_recoder.value_set('label')
        labels.remove('unknown')
        for label in labels:
            if np.all(st_order == self._slice_time_order(
                    label,
                    n_timed)):
                break
        else:
            raise HeaderDataError('slice ordering of %s fits '
                                  'with no known scheme' % st_order)
        # Set values into header
        hdr['slice_start'] = slice_start
        hdr['slice_end'] = slice_end
        hdr['slice_duration'] = duration
        hdr['slice_code'] = slice_order_codes.code[label]

    def _slice_time_order(self, slabel, n_slices):
        ''' Supporting function to give time order of slices from label '''
        if slabel == 'sequential increasing':
            sp_ind_time_order = range(n_slices)
        elif slabel == 'sequential decreasing':
            sp_ind_time_order = range(n_slices)[::-1]
        elif slabel == 'alternating increasing':
            sp_ind_time_order = range(0, n_slices, 2) + range(1, n_slices, 2)
        elif slabel == 'alternating decreasing':
            sp_ind_time_order = range(n_slices - 1, -1, -2) \
                                + range(n_slices -2 , -1, -2)
        elif slabel == 'alternating increasing 2':
            sp_ind_time_order = range(1, n_slices, 2) + range(0, n_slices, 2)
        elif slabel == 'alternating decreasing 2':
            sp_ind_time_order = range(n_slices - 2, -1, -2) \
                                + range(n_slices - 1, -1, -2)
        else:
            raise HeaderDataError('We do not handle slice ordering "%s"'
                                  % slabel)
        return np.argsort(sp_ind_time_order)

    def get_xyzt_units(self):
        xyz_code = self.structarr['xyzt_units'] % 8
        t_code = self.structarr['xyzt_units'] - xyz_code
        return (unit_codes.label[xyz_code],
                unit_codes.label[t_code])

    def set_xyzt_units(self, xyz=None, t=None):
        if xyz is None:
            xyz = 0
        if t is None:
            t = 0
        xyz_code = self.structarr['xyzt_units'] % 8
        t_code = self.structarr['xyzt_units'] - xyz_code
        xyz_code = unit_codes[xyz]
        t_code = unit_codes[t]
        self.structarr['xyzt_units'] = xyz_code + t_code

    def _set_format_specifics(self):
        ''' Utility routine to set format specific header stuff '''
        if self.is_single:
            self._structarr['magic'] = 'n+1'
            if self._structarr['vox_offset'] < 352:
                self._structarr['vox_offset'] = 352
        else:
            self._structarr['magic'] = 'ni1'

    ''' Checks only below here '''

    @classmethod
    def _get_checks(klass):
        # We need to return our own versions of - e.g. chk_datatype, to
        # pick up the Nifti datatypes from our class
        return (klass._chk_sizeof_hdr,
                klass._chk_datatype,
                klass._chk_bitpix,
                klass._chk_pixdims,
                klass._chk_scale_inter,
                klass._chk_qfac,
                klass._chk_magic_offset,
                klass._chk_qform_code,
                klass._chk_sform_code)

    @staticmethod
    def _chk_scale_inter(hdr, fix=False):
        rep = Report(HeaderDataError)
        scale = hdr['scl_slope']
        offset = hdr['scl_inter']
        usable_scale = np.isfinite(scale) and scale !=0
        # Nonzero finite scale, and valid offset
        if usable_scale and np.isfinite(offset) or (offset, scale) == (0, 0):
            return hdr, rep
        # If scale is usable but the intercept is not finite, that's a serious
        # problem
        if usable_scale and not np.isfinite(offset):
            rep.problem_level = 40
            rep.problem_msg = ('"scl_slope" is %s; but "scl_inter" is %s; '
                               '"scl_inter" should be finite'
                               % (scale, offset))
            if fix:
                hdr['scl_inter'] = 0
                rep.fix_msg = 'setting "scl_inter" to 0'
            return hdr, rep
        level = 0
        msgs = []
        fix_msgs = []
        # Non-finite scale is obviously an error.  We still need to check the
        # intercept though
        if not np.isfinite(scale):
            level = 30
            msgs.append('"scl_slope" is %s; should be finite' % scale)
            if fix:
                hdr['scl_slope'] = 0
                fix_msgs.append('setting "scl_slope" to 0 (no scaling)')
        # We've established scale is not usable, so inter will be ignored.  That
        # means we can go a bit easy on bad intercepts
        if offset != 0:
            if level == 0: level = 20
            msgs.append('Unused "scl_inter" is %s; should be 0' % offset)
            if fix:
                hdr['scl_inter'] = 0
                fix_msgs.append('setting "scl_inter" to 0')
        rep.problem_level = level
        rep.problem_msg = '; '.join(msgs)
        rep.fix_msg = '; '.join(fix_msgs)
        return hdr, rep

    @staticmethod
    def _chk_qfac(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['pixdim'][0] in (-1, 1):
            return hdr, rep
        rep.problem_level = 20
        rep.problem_msg = 'pixdim[0] (qfac) should be 1 (default) or -1'
        if fix:
            hdr['pixdim'][0] = 1
            rep.fix_msg = 'setting qfac to 1'
        return hdr, rep

    @staticmethod
    def _chk_magic_offset(hdr, fix=False):
        rep = Report(HeaderDataError)
        # for ease of later string formatting, use scalar of byte string
        magic = np.asscalar(hdr['magic'])
        offset = hdr['vox_offset']
        if magic == asbytes('n+1'): # one file
            if offset >= 352:
                if not offset % 16:
                    return hdr, rep
                else:
                    # SPM uses memory mapping to read the data, and
                    # apparently this has to start on 16 byte boundaries
                    rep.problem_msg = ('vox offset (=%s) not divisible '
                                       'by 16, not SPM compatible' % offset)
                    rep.problem_level = 30
                    if fix:
                        rep.fix_msg = 'leaving at current value'
                    return hdr, rep
            rep.problem_level = 40
            rep.problem_msg = ('vox offset %d too low for '
                               'single file nifti1' % offset)
            if fix:
                hdr['vox_offset'] = 352
                rep.fix_msg = 'setting to minimum value of 352'
        elif magic != asbytes('ni1'): # two files
            # unrecognized nii magic string, oh dear
            rep.problem_msg = ('magic string "%s" is not valid' %
                               asstr(magic))
            rep.problem_level = 45
            if fix:
                rep.fix_msg = 'leaving as is, but future errors are likely'
        return hdr, rep

    @classmethod
    def _chk_qform_code(klass, hdr, fix=False):
        return klass._chk_xform_code('qform_code', hdr, fix)

    @classmethod
    def _chk_sform_code(klass, hdr, fix=False):
        return klass._chk_xform_code('sform_code', hdr, fix)

    @classmethod
    def _chk_xform_code(klass, code_type, hdr, fix):
        # utility method for sform and qform codes
        rep = Report(HeaderDataError)
        code = int(hdr[code_type])
        recoder = klass._field_recoders[code_type]
        if code in recoder.value_set():
            return hdr, rep
        rep.problem_level = 30
        rep.problem_msg = '%s %d not valid' % (code_type, code)
        if fix:
            hdr[code_type] = 0
            rep.fix_msg = 'setting to 0'
        return hdr, rep


class Nifti1PairHeader(Nifti1Header):
    ''' Class for nifti1 pair header '''
    # Signal whether this is single (header + data) file
    is_single = False


class Nifti1Pair(analyze.AnalyzeImage):
    header_class = Nifti1PairHeader

    def _write_header(self, header_file, header, slope, inter):
        super(Nifti1Pair, self)._write_header(header_file,
                                              header,
                                              slope,
                                              inter)

    def update_header(self):
        ''' Harmonize header with image data and affine

        See AnalyzeImage.update_header for more examples

        Examples
        --------
        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = Nifti1Image(data, affine)
        >>> hdr = img.get_header()
        >>> np.all(hdr.get_qform() == affine)
        True
        >>> np.all(hdr.get_sform() == affine)
        True
        '''
        super(Nifti1Pair, self).update_header()
        hdr = self._header
        hdr['magic'] = 'ni1'
        # If the affine is not None, and it is different from the main affine in
        # the header, update the heaader
        if self._affine is None:
            return
        if np.allclose(self._affine, hdr.get_best_affine()):
            return
        # Set affine into sform with default code
        hdr.set_sform(self._affine, code='aligned')
        # Make qform 'unknown'
        hdr.set_qform(self._affine, code='unknown')

    def get_qform(self, coded=False):
        """ Return 4x4 affine matrix from qform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and qform code.  If False, just
            return affine.  {affine or None} means, return None if qform code ==
            0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine reconstructed from qform
            quaternion.  If `coded` is True, return None if qform code is 0,
            else return the affine.
        code : int
            Qform code. Only returned if `coded` is True.

        See also
        --------
        Nifti1Header.set_qform
        """
        return self._header.get_qform(coded)

    def set_qform(self, affine, code=None, strip_shears=True, **kwargs):
        ''' Set qform header values from 4x4 affine

        Parameters
        ----------
        hdr : nifti1 header
        affine : None or 4x4 array
            affine transform to write into sform. If None, only set code.
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:
            * If affine is None, `code`-> 0
            * If affine not None and sform code in header == 0, `code`-> 2
              (aligned)
            * If affine not None and sform code in header != 0, `code`-> sform
              code in header
        strip_shears : bool, optional
            Whether to strip shears in `affine`.  If True, shears will be
            silently stripped. If False, the presence of shears will raise a
            ``HeaderDataError``
        update_affine : bool, optional
            Whether to update the image affine from the header best affine after
            setting the qform. Must be keyword argumemt (because of different
            position in `set_qform`). Default is True

        See also
        --------
        Nifti1Header.set_qform

        Examples
        --------
        >>> data = np.arange(24).reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_qform()
        array([[ 2.,  0.,  0.,  0.],
               [ 0.,  3.,  0.,  0.],
               [ 0.,  0.,  4.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> img.get_qform(coded=True)
        (None, 0)
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_qform(aff2, 'talairach')
        >>> qaff, code = img.get_qform(coded=True)
        >>> np.all(qaff == aff2)
        True
        >>> int(code)
        3
        '''
        update_affine = kwargs.pop('update_affine', True)
        if kwargs:
            raise TypeError('Unexpected keyword argument(s) %s' % kwargs)
        self._header.set_qform(affine, code, strip_shears)
        if update_affine:
            self._affine[:] = self._header.get_best_affine()

    def get_sform(self, coded=False):
        """ Return 4x4 affine matrix from sform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and sform code.  If False, just
            return affine.  {affine or None} means, return None if sform code ==
            0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine from sform fields. If
            `coded` is True, return None if sform code is 0, else return the
            affine.
        code : int
            Sform code. Only returned if `coded` is True.

        See also
        --------
        Nifti1Header.get_sform
        """
        return self._header.get_sform(coded)

    def set_sform(self, affine, code=None, **kwargs):
        ''' Set sform transform from 4x4 affine

        Parameters
        ----------
        hdr : nifti1 header
        affine : None or 4x4 array
            affine transform to write into sform.  If None, only set `code`
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:
            * If affine is None, `code`-> 0
            * If affine not None and sform code in header == 0, `code`-> 2
              (aligned)
            * If affine not None and sform code in header != 0, `code`-> sform
              code in header
        update_affine : bool, optional
            Whether to update the image affine from the header best affine after
            setting the qform.  Must be keyword argumemt (because of different
            position in `set_qform`). Default is True

        See also
        --------
        Nifti1Header.set_sform

        Examples
        --------
        >>> data = np.arange(24).reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_sform()
        array([[ 2.,  0.,  0.,  0.],
               [ 0.,  3.,  0.,  0.],
               [ 0.,  0.,  4.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> saff, code = img.get_sform(coded=True)
        >>> saff
        array([[ 2.,  0.,  0.,  0.],
               [ 0.,  3.,  0.,  0.],
               [ 0.,  0.,  4.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> int(code)
        2
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_sform(aff2, 'talairach')
        >>> saff, code = img.get_sform(coded=True)
        >>> np.all(saff == aff2)
        True
        >>> int(code)
        3
        '''
        update_affine = kwargs.pop('update_affine', True)
        if kwargs:
            raise TypeError('Unexpected keyword argument(s) %s' % kwargs)
        self._header.set_sform(affine, code)
        if update_affine:
            self._affine[:] = self._header.get_best_affine()


class Nifti1Image(Nifti1Pair):
    header_class = Nifti1Header
    files_types = (('image', '.nii'),)

    @staticmethod
    def _get_fileholders(file_map):
        """ Return fileholder for header and image

        For single-file niftis, the fileholder for the header and the image will
        be the same
        """
        return file_map['image'], file_map['image']

    def _write_header(self, header_file, header, slope, inter):
        super(Nifti1Image, self)._write_header(header_file,
                                               header,
                                               slope,
                                               inter)
        # We need to set the header offset ready for writing the image.
        # Streams like bz2 do not allow write seeks, even forward.  We
        # check where to go, and write zeros up until the data part of
        # the file
        offset = header.get_data_offset()
        diff = offset-header_file.tell()
        if diff > 0:
            header_file.write(ZEROB * diff)

    def update_header(self):
        ''' Harmonize header with image data and affine '''
        super(Nifti1Image, self).update_header()
        hdr = self._header
        hdr['magic'] = 'n+1'
        # make sure that there is space for the header.  If any
        # extensions, figure out necessary vox_offset for extensions to
        # fit
        min_vox_offset = 352 + hdr.extensions.get_sizeondisk()
        if hdr['vox_offset'] < min_vox_offset:
            hdr['vox_offset'] = min_vox_offset


def load(filename):
    """ Load nifti1 single or pair from `filename`

    Parameters
    ----------
    filename : str
        filename of image to be loaded

    Returns
    -------
    img : Nifti1Image or Nifti1Pair
        nifti1 single or pair image instance

    Raises
    ------
    ImageFileError: if `filename` doesn't look like nifti1
    IOError : if `filename` does not exist
    """
    try:
        img = Nifti1Image.load(filename)
    except ImageFileError:
        return Nifti1Pair.load(filename)
    return img


def save(img, filename):
    """ Save nifti1 single or pair to `filename`

    Parameters
    ----------
    filename : str
        filename to which to save image
    """
    try:
        Nifti1Image.instance_to_filename(img, filename)
    except ImageFileError:
        Nifti1Pair.instance_to_filename(img, filename)


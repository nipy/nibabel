# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import numpy
from nibabel.volumeutils import Recoder
import numpy as np

xform_codes = Recoder(( # code, label
    (0, 'unknown', "NIFTI_XFORM_UNKNOWN"), # Code for transform unknown or absent
    (1, 'scanner', "NIFTI_XFORM_SCANNER_ANAT"),
    (2, 'aligned', "NIFTI_XFORM_ALIGNED_ANAT"),
    (3, 'talairach', "NIFTI_XFORM_TALAIRACH"),
    (4, 'mni', "NIFTI_XFORM_MNI_152")), fields=('code', 'label', 'niistring'))

try:
    _float128t = np.float128
except AttributeError:
    _float128t = np.void
try:
    _complex256t = np.complex256
except AttributeError:
    _complex256t = np.void
    
dtdefs = ( # code, label, dtype definition, niistring
    (0, 'none', np.void, ""),
    (1, 'binary', np.void, ""), # 1 bit per voxel, needs thought
    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),
    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),
    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),
    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),
    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"), # numpy complex format?
    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),
    (128, 'RGB', np.dtype([('R','u1'),
                  ('G', 'u1'),
                  ('B', 'u1')]), "NIFTI_TYPE_RGB24"),
    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),
    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),
    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),
    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),
    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),
    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"), # Only numpy defined on 64 bit
    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),
    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"), # 64 bit again
    (2304, 'RGBA', np.dtype([('R','u1'),
                    ('G', 'u1'),
                    ('B', 'u1'),
                    ('A', 'u1')]), "NIFTI_TYPE_RGBA32")
    )

# XXX: do i need the extensions provided by volumeutils.make_dt_codes()
data_type_codes = Recoder( dtdefs,  fields=('code', 'label', 'type', 'niistring') )


array_index_order_codes = Recoder((
                                   (1, "RowMajorOrder"),
                                   (2, "ColumnMajorOrder"),
                                   ), fields = ('code', 'label'))

gifti_encoding_codes = Recoder((
                                (0, "undef", "GIFTI_ENCODING_UNDEF", "undef"),
                                (1, "ASCII", "GIFTI_ENCODING_ASCII", "ASCII"),
                                (2, "B64BIN", "GIFTI_ENCODING_B64BIN", "Base64Binary" ),
                                (3, "B64GZ", "GIFTI_ENCODING_B64GZ", "GZipBase64Binary"),
                                (4, "External", "GIFTI_ENCODING_EXTBIN", "ExternalFileBinary"),
                                ), fields = ('code', 'label', 'giistring', 'specs'))




gifti_endian_codes = Recoder((
                            (0, "GIFTI_ENDIAN_UNDEF", "Undef"),
                            (1, "GIFTI_ENDIAN_BIG", "BigEndian"),
                            (2, "GIFTI_ENDIAN_LITTLE", "LittleEndian"),
                              ), fields = ('code', 'giistring', 'specs'))

intent_codes = Recoder((
    # code, label, parameters description tuple
    (0, 'none', (), "NIFTI_INTENT_NONE"),
    (2, 'correlation',('p1 = DOF',), "NIFTI_INTENT_CORREL"),
    (3, 't test', ('p1 = DOF',), "NIFTI_INTENT_TTEST"),
    (4, 'f test', ('p1 = numerator DOF', 'p2 = denominator DOF'), "NIFTI_INTENT_FTEST"),
    (5, 'z score', (), "NIFTI_INTENT_ZSCORE"),
    (6, 'chi2', ('p1 = DOF',), "NIFTI_INTENT_CHISQ"),
    (7, 'beta', ('p1=a', 'p2=b'), "NIFTI_INTENT_BETA"), # two parameter beta distribution
    (8, 'binomial', ('p1 = number of trials', 'p2 = probability per trial'), "NIFTI_INTENT_BINOM"),
    # Prob(x) = (p1 choose x) * p2^x * (1-p2)^(p1-x), for x=0,1,...,p1
    (9, 'gamma', ('p1 = shape, p2 = scale', 2), "NIFTI_INTENT_GAMMA"), # 2 parameter gamma
    (10, 'poisson', ('p1 = mean',), "NIFTI_INTENT_POISSON"), # Density(x) proportional to
                                     # x^(p1-1) * exp(-p2*x)
    (11, 'normal', ('p1 = mean', 'p2 = standard deviation',), "NIFTI_INTENT_NORMAL"),
    (12, 'non central f test', ('p1 = numerator DOF',
                                'p2 = denominator DOF',
                                'p3 = numerator noncentrality parameter',), "NIFTI_INTENT_FTEST_NONC"),
    (13, 'non central chi2', ('p1 = DOF', 'p2 = noncentrality parameter',), "NIFTI_INTENT_CHISQ_NONC"),
    (14, 'logistic', ('p1 = location', 'p2 = scale',), "NIFTI_INTENT_LOGISTIC"),
    (15, 'laplace', ('p1 = location', 'p2 = scale'), "NIFTI_INTENT_LAPLACE"),
    (16, 'uniform', ('p1 = lower end', 'p2 = upper end'), "NIFTI_INTENT_UNIFORM"),
    (17, 'non central t test', ('p1 = DOF', 'p2 = noncentrality parameter'), "NIFTI_INTENT_TTEST_NONC"),
    (18, 'weibull', ('p1 = location', 'p2 = scale, p3 = power'), "NIFTI_INTENT_WEIBULL"),
    (19, 'chi', ('p1 = DOF',), "NIFTI_INTENT_CHI"),
    # p1 = 1 = 'half normal' distribution
    # p1 = 2 = Rayleigh distribution
    # p1 = 3 = Maxwell-Boltzmann distribution.
    (20, 'inverse gaussian', ('pi = mu', 'p2 = lambda'), "NIFTI_INTENT_INVGAUSS"),
    (21, 'extreme value 1', ('p1 = location', 'p2 = scale'), "NIFTI_INTENT_EXTVAL"),
    (22, 'p value', (), "NIFTI_INTENT_PVAL"),
    (23, 'log p value', (), "NIFTI_INTENT_LOGPVAL"),
    (24, 'log10 p value', (), "NIFTI_INTENT_LOG10PVAL"),
    (1001, 'estimate', (), "NIFTI_INTENT_ESTIMATE"),
    (1002, 'label', (), "NIFTI_INTENT_LABEL"),
    (1003, 'neuroname', (), "NIFTI_INTENT_NEURONAME"),
    (1004, 'general matrix', ('p1 = M', 'p2 = N'), "NIFTI_INTENT_GENMATRIX"),
    (1005, 'symmetric matrix', ('p1 = M',), "NIFTI_INTENT_SYMMATRIX"),
    (1006, 'displacement vector', (), "NIFTI_INTENT_DISPVECT"),
    (1007, 'vector', (), "NIFTI_INTENT_VECTOR"),
    (1008, 'poinset', (), "NIFTI_INTENT_POINTSET"),
    (1009, 'triangle', (), "NIFTI_INTENT_TRIANGLE"),
    (1010, 'quaternion', (), "NIFTI_INTENT_QUATERNION"),
    (1011, 'dimensionless', (), "NIFTI_INTENT_DIMLESS"),
    (2001, 'time series', (), "NIFTI_INTENT_TIMESERIES"),
    (2002, 'node index', (), "NIFTI_INTENT_NODE_INDEX"),
    (2003, 'rgb vector', (), "NIFTI_INTENT_RGB_VECTOR"),
    (2004, 'rgba vector', (), "NIFTI_INTENT_RGBA_VECTOR"),
    (2005, 'shape', (), "NIFTI_INTENT_SHAPE")),
                       fields=('code', 'label', 'parameters', 'niistring'))


#  GiftiIntentCode contains the intent codes as defined in nifti1.h
#  Dictionary access as well as direct access to the values in order to
#  provide as many accessor methods as possible.

class GiftiIntentCode(object):
    intents = {}

    NIFTI_INTENT_NONE = 0
    NIFTI_INTENT_TTEST = 3
    NIFTI_INTENT_FTEST = 4
    NIFTI_INTENT_ZSCORE = 5
    NIFTI_INTENT_CHISQ = 6
    NIFTI_INTENT_BETA = 7
    NIFTI_INTENT_BINOM = 8
    NIFTI_INTENT_GAMMA = 9
    NIFTI_INTENT_POISSON = 10
    NIFTI_INTENT_NORMAL = 11
    NIFTI_INTENT_FTEST_NONC = 12
    NIFTI_INTENT_CHISQ_NONC = 13
    NIFTI_INTENT_LOGISTIC = 14
    NIFTI_INTENT_LAPLACE = 15
    NIFTI_INTENT_UNIFORM = 16
    NIFTI_INTENT_TTEST_NONC = 17
    NIFTI_INTENT_WEIBULL = 18
    NIFTI_INTENT_CHI = 19
    NIFTI_INTENT_INVGAUSS = 20
    NIFTI_INTENT_EXTVAL = 21
    NIFTI_INTENT_PVAL = 22
    NIFTI_INTENT_LOGPVAL = 23
    NIFTI_INTENT_LOG10PVAL = 24
    NIFTI_FIRST_STATCODE = 2
    NIFTI_LAST_STATCODE = 24
    NIFTI_INTENT_ESTIMATE = 1001
    NIFTI_INTENT_LABEL = 1002
    NIFTI_INTENT_NEURONAME = 1003
    NIFTI_INTENT_GENMATRIX = 1004
    NIFTI_INTENT_SYMMATRIX = 1005
    NIFTI_INTENT_DISPVECT = 1006
    NIFTI_INTENT_VECTOR = 1007
    NIFTI_INTENT_POINTSET = 1008
    NIFTI_INTENT_TRIANGLE = 1009
    NIFTI_INTENT_QUATERNION = 1010
    NIFTI_INTENT_DIMLESS = 1011
    NIFTI_INTENT_TIMESERIES = 2001
    NIFTI_INTENT_NODE_INDEX = 2002
    NIFTI_INTENT_RGB_VECTOR = 2003
    NIFTI_INTENT_RGBA_VECTOR = 2004
    NIFTI_INTENT_SHAPE = 2005

    intents_inv = {
    0 : "NIFTI_INTENT_NONE",
    3 : "NIFTI_INTENT_TTEST",
    4 : "NIFTI_INTENT_FTEST",
    5 : "NIFTI_INTENT_ZSCORE",
    6 : "NIFTI_INTENT_CHISQ",
    7 : "NIFTI_INTENT_BETA",
    8 : "NIFTI_INTENT_BINOM",
    9 : "NIFTI_INTENT_GAMMA",
    10 : "NIFTI_INTENT_POISSON",
    11 : "NIFTI_INTENT_NORMAL",
    12 : "NIFTI_INTENT_FTEST_NONC",
    13 : "NIFTI_INTENT_CHISQ_NONC",
    14 : "NIFTI_INTENT_LOGISTIC",
    15 : "NIFTI_INTENT_LAPLACE",
    16 : "NIFTI_INTENT_UNIFORM",
    17 : "NIFTI_INTENT_TTEST_NONC",
    18 : "NIFTI_INTENT_WEIBULL",
    19 : "NIFTI_INTENT_CHI",
    20 : "NIFTI_INTENT_INVGAUSS",
    21 : "NIFTI_INTENT_EXTVAL",
    22 : "NIFTI_INTENT_PVAL",
    23 : "NIFTI_INTENT_LOGPVAL",
    24 : "NIFTI_INTENT_LOG10PVAL",
    2 : "NIFTI_FIRST_STATCODE",
    22 : "NIFTI_LAST_STATCODE",
    1001 : "NIFTI_INTENT_ESTIMATE",
    1002 : "NIFTI_INTENT_LABEL",
    1003 : "NIFTI_INTENT_NEURONAME",
    1004 : "NIFTI_INTENT_GENMATRIX",
    1005 : "NIFTI_INTENT_SYMMATRIX",
    1006 : "NIFTI_INTENT_DISPVECT",
    1007 : "NIFTI_INTENT_VECTOR",
    1008 : "NIFTI_INTENT_POINTSET",
    1009 : "NIFTI_INTENT_TRIANGLE",
    1010 : "NIFTI_INTENT_QUATERNION",
    1011 : "NIFTI_INTENT_DIMLESS",
    2001 : "NIFTI_INTENT_TIMESERIES",
    2002 : "NIFTI_INTENT_NODE_INDEX",
    2003 : "NIFTI_INTENT_RGB_VECTOR",
    2004 : "NIFTI_INTENT_RGBA_VECTOR",
    2005 : "NIFTI_INTENT_SHAPE"
    }



class GiftiArrayIndexOrder(object):

    ordering = {}
    ordering["RowMajorOrder"] = 1
    ordering["ColumnMajorOrder"] = 2

    RowMajorOrder = 1
    ColumnMajorOrder = 2

    ordering_inv = {
    1 : "RowMajorOrder",
    2 : "ColumnMajorOrder"
    }




class GiftiEncoding(object):

    encodings = {}
    encodings["GIFTI_ENCODING_UNDEF"]  = 0
    encodings["GIFTI_ENCODING_ASCII"]  = 1
    encodings["GIFTI_ENCODING_B64BIN"] = 2
    encodings["GIFTI_ENCODING_B64GZ"]  = 3
    encodings["GIFTI_ENCODING_EXTBIN"] = 4
    encodings["GIFTI_ENCODING_MAX"]    = 4

    # not specified
    encodings["GZipBase64Binary"] = 3
    encodings["ASCII"] = 1

    GIFTI_ENCODING_UNDEF  = 0
    GIFTI_ENCODING_ASCII  = 1
    GIFTI_ENCODING_B64BIN = 2
    GIFTI_ENCODING_B64GZ  = 3
    GIFTI_ENCODING_EXTBIN = 4
    GIFTI_ENCODING_MAX    = 4

    encodings_inv = {
    0 : "GIFTI_ENCODING_UNDEF",
    1 : "GIFTI_ENCODING_ASCII",
    2 : "GIFTI_ENCODING_B64BIN",
    3 : "GIFTI_ENCODING_B64GZ",
    4 : "GIFTI_ENCODING_EXTBIN"
    }



class GiftiEndian(object):
    endian = {}
    endian["GIFTI_ENDIAN_UNDEF"] = 0
    endian["GIFTI_ENDIAN_BIG"]   = 1
    endian["GIFTI_ENDIAN_LITTLE"]= 2
    endian["GIFTI_ENDIAN_MAX"]   = 2

    # not officially specified
    endian["LittleEndian"]= 2
    endian["BigEndian"]= 1

    GIFTI_ENDIAN_UNDEF = 0
    GIFTI_ENDIAN_BIG   = 1
    GIFTI_ENDIAN_LITTLE= 2
    GIFTI_ENDIAN_MAX   = 2

    endian_inv = {
    0 : "GIFTI_ENDIAN_UNDEF",
    1 : "GIFTI_ENDIAN_BIG",
    2 : "GIFTI_ENDIAN_LITTLE"
    }



class GiftiDataType(object):
    datatypes = {}
    datatypes["NIFTI_TYPE_UINT8"]      = 2
    datatypes["NIFTI_TYPE_INT16"]      = 4
    datatypes["NIFTI_TYPE_INT32"]      = 8
    datatypes["NIFTI_TYPE_FLOAT32"]    = 16
    datatypes["NIFTI_TYPE_COMPLEX64"]  = 32
    datatypes["NIFTI_TYPE_FLOAT64"]    = 64
    datatypes["NIFTI_TYPE_RGB24"]      = 128
    datatypes["NIFTI_TYPE_INT8"]       = 256
    datatypes["NIFTI_TYPE_UINT16"]     = 512
    datatypes["NIFTI_TYPE_UINT32"]     = 768
    datatypes["NIFTI_TYPE_INT64"]      = 1024
    datatypes["NIFTI_TYPE_UINT64"]     = 1280
    datatypes["NIFTI_TYPE_FLOAT128"]   = 1536   #  Python cannot handle 128-bit floats
    datatypes["NIFTI_TYPE_COMPLEX128"] = 1792
    datatypes["NIFTI_TYPE_COMPLEX256"] = 2048   #  Python cannot handle 128-bit floats
    datatypes["NIFTI_TYPE_RGBA32"]     = 2304

    NIFTI_TYPE_UINT8      = 2
    NIFTI_TYPE_INT16      = 4
    NIFTI_TYPE_INT32      = 8
    NIFTI_TYPE_FLOAT32    = 16
    NIFTI_TYPE_COMPLEX64  = 32
    NIFTI_TYPE_FLOAT64    = 64
    NIFTI_TYPE_RGB24      = 128
    NIFTI_TYPE_INT8       = 256
    NIFTI_TYPE_UINT16     = 512
    NIFTI_TYPE_UINT32     = 768
    NIFTI_TYPE_INT64      = 1024
    NIFTI_TYPE_UINT64     = 1280
    NIFTI_TYPE_FLOAT128   = 1536   #  Python cannot handle 128-bit floats
    NIFTI_TYPE_COMPLEX128 = 1792
    NIFTI_TYPE_COMPLEX256 = 2048   #  Python cannot handle 128-bit floats
    NIFTI_TYPE_RGBA32     = 2304

    datatypes_inv =  {
    2 : "NIFTI_TYPE_UINT8",
    4 : "NIFTI_TYPE_INT16",
    8 : "NIFTI_TYPE_INT32",
    16 : "NIFTI_TYPE_FLOAT32",
    32 : "NIFTI_TYPE_COMPLEX64",
    64 : "NIFTI_TYPE_FLOAT64",
    128 : "NIFTI_TYPE_RGB24",
    256 : "NIFTI_TYPE_INT8",
    512 : "NIFTI_TYPE_UINT16",
    768 : "NIFTI_TYPE_UINT32",
    1024 : "NIFTI_TYPE_INT64",
    1280 : "NIFTI_TYPE_UINT64",
    1536 : "NIFTI_TYPE_FLOAT128",
    1792 : "NIFTI_TYPE_COMPLEX128",
    2048 : "NIFTI_TYPE_COMPLEX256",
    2304 : "NIFTI_TYPE_RGBA32"
    }

#  Some helper functions
def get_endianness():
    import array
    end =  ord(array.array("i",[1]).tostring()[0])

    if end:
        # this is little endian
        return GiftiEndian.GIFTI_ENDIAN_LITTLE
    else:
        # this is big endian
        return GiftiEndian.GIFTI_ENDIAN_BIG

GiftiType2npyType = { \
 GiftiDataType.NIFTI_TYPE_COMPLEX128 : numpy.dtype('complex128'),
 GiftiDataType.NIFTI_TYPE_COMPLEX64 : numpy.dtype('complex64'),
 GiftiDataType.NIFTI_TYPE_FLOAT32 : numpy.dtype('float32'),
 GiftiDataType.NIFTI_TYPE_FLOAT64 : numpy.dtype('float64'),
 GiftiDataType.NIFTI_TYPE_INT16 : numpy.dtype('int16'),
 GiftiDataType.NIFTI_TYPE_INT32 : numpy.dtype('int32'),
 GiftiDataType.NIFTI_TYPE_INT64 : numpy.dtype('int64'),
 GiftiDataType.NIFTI_TYPE_INT8 : numpy.dtype('int8'),
 GiftiDataType.NIFTI_TYPE_UINT16 : numpy.dtype('uint16'),
 GiftiDataType.NIFTI_TYPE_UINT32 : numpy.dtype('uint32'),
 GiftiDataType.NIFTI_TYPE_UINT64 : numpy.dtype('uint64'),
 GiftiDataType.NIFTI_TYPE_UINT8 : numpy.dtype('uint8'),
 #GiftiDataType.NIFTI_TYPE_RGB24 : rgb.fields_rgb, # See below
 #GiftiDataType.NIFTI_TYPE_RGBA32 : rgb.fields_rgba,
 2048 : numpy.dtype('complex128'),
 32 : numpy.dtype('complex64'),
 16 : numpy.dtype('float32'),
 64 : numpy.dtype('float64'),
 4 : numpy.dtype('int16'),
 8 : numpy.dtype('int32'),
 1024 : numpy.dtype('int64'),
 256 : numpy.dtype('int8'),
 512 : numpy.dtype('uint16'),
 768 : numpy.dtype('uint32'),
 1280 : numpy.dtype('uint64'),
 2 : numpy.dtype('uint8'),
 }

# what about rgbs? XXX

#from numpy import uint8
#class rgb:
#   """ mini class to handle access to NIFTI RGB* typed arrays. You can
# switch from fields-based (1D) indexation to shape-based (2D) indexation
# with as1D and as2D"""
#   uint8s_rgb = numpy.dtype('(3,)uint8')
#   uint8s_rgba = numpy.dtype('(4,)uint8')
#   fields_rgb = numpy.dtype([('r',uint8), ('g',uint8), ('b',uint8) ])
#   fields_rgba = numpy.dtype([('r',uint8),('g',uint8),('b',uint8),('a',uint8)])
#
#_as2D = lambda a : numpy.frombuffer(a, (rgb.uint8s_rgb, rgb.uint8s_rgba)[(a.itemsize==1) and (a.shape[-1] - 3) or (a.itemsize - 3)])
#_as1D = lambda a : numpy.frombuffer(a, (rgb.fields_rgb, rgb.fields_rgba)[(a.itemsize==1) and (a.shape[-1] - 3) or (a.itemsize - 3)])
#rgb.as2D = staticmethod(_as2D)
#rgb.as1D = staticmethod(_as1D)

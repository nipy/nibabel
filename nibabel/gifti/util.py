# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import numpy

#  GiftiIntentCode contains the intent codes as defined in nifti1.h
#  Dictionary access as well as direct access to the values in order to
#  provide as many accessor methods as possible.

class GiftiIntentCode(object):
    intents = {}
    intents["NIFTI_INTENT_NONE"] = 0
    intents["NIFTI_INTENT_TTEST"] = 3
    intents["NIFTI_INTENT_FTEST"] = 4
    intents["NIFTI_INTENT_ZSCORE"] = 5
    intents["NIFTI_INTENT_CHISQ"] = 6
    intents["NIFTI_INTENT_BETA"] = 7
    intents["NIFTI_INTENT_BINOM"] = 8
    intents["NIFTI_INTENT_GAMMA"] = 9
    intents["NIFTI_INTENT_POISSON"] = 10
    intents["NIFTI_INTENT_NORMAL"] = 11
    intents["NIFTI_INTENT_FTEST_NONC"] = 12
    intents["NIFTI_INTENT_CHISQ_NONC"] = 13
    intents["NIFTI_INTENT_LOGISTIC"] = 14
    intents["NIFTI_INTENT_LAPLACE"] = 15
    intents["NIFTI_INTENT_UNIFORM"] = 16
    intents["NIFTI_INTENT_TTEST_NONC"] = 17
    intents["NIFTI_INTENT_WEIBULL"] = 18
    intents["NIFTI_INTENT_CHI"] = 19
    intents["NIFTI_INTENT_INVGAUSS"] = 20
    intents["NIFTI_INTENT_EXTVAL"] = 21
    intents["NIFTI_INTENT_PVAL"] = 22
    intents["NIFTI_INTENT_LOGPVAL"] = 23
    intents["NIFTI_INTENT_LOG10PVAL"] = 24
    intents["NIFTI_FIRST_STATCODE"] = 2
    intents["NIFTI_LAST_STATCODE"] = 24
    intents["NIFTI_INTENT_ESTIMATE"] = 1001
    intents["NIFTI_INTENT_LABEL"] = 1002
    intents["NIFTI_INTENT_NEURONAME"] = 1003
    intents["NIFTI_INTENT_GENMATRIX"] = 1004
    intents["NIFTI_INTENT_SYMMATRIX"] = 1005
    intents["NIFTI_INTENT_DISPVECT"] = 1006
    intents["NIFTI_INTENT_VECTOR"] = 1007
    intents["NIFTI_INTENT_POINTSET"] = 1008
    intents["NIFTI_INTENT_TRIANGLE"] = 1009
    intents["NIFTI_INTENT_QUATERNION"] = 1010
    intents["NIFTI_INTENT_DIMLESS"] = 1011
    intents["NIFTI_INTENT_TIMESERIES"] = 2001
    intents["NIFTI_INTENT_NODE_INDEX"] = 2002
    intents["NIFTI_INTENT_RGB_VECTOR"] = 2003
    intents["NIFTI_INTENT_RGBA_VECTOR"] = 2004
    intents["NIFTI_INTENT_SHAPE"] = 2005

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

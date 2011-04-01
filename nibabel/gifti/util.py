# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

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
    
dtdefs = ( # code, label, dtype definition, niistring, XXX: format for store in txt
    (0, 'none', np.void, "", ""),
    (1, 'binary', np.void, "", ""), # 1 bit per voxel, needs thought
    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8", "%i"),
    (4, 'int16', np.int16, "NIFTI_TYPE_INT16", "%i"),
    (8, 'int32', np.int32, "NIFTI_TYPE_INT32", "%i"),
    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32", "%10.6f"),
    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64", "%10.6f"), # numpy complex format?
    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64", "%10.6f"),
    (128, 'RGB', np.dtype([('R','u1'),
                  ('G', 'u1'),
                  ('B', 'u1')]), "NIFTI_TYPE_RGB24", ""),
    (256, 'int8', np.int8, "NIFTI_TYPE_INT8", "%i"),
    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16", "%i"),
    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32", "%i"),
    (1024,'int64', np.int64, "NIFTI_TYPE_INT64", "%i"),
    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64", "%i"),
    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128", "%10.6f"), # Only numpy defined on 64 bit
    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128", "%10.6f"),
    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256", "%10.6f"), # 64 bit again
    (2304, 'RGBA', np.dtype([('R','u1'),
                    ('G', 'u1'),
                    ('B', 'u1'),
                    ('A', 'u1')]), "NIFTI_TYPE_RGBA32", ""),
    )


# Translate dtype.kind char codes to XML text output strings
KIND2FMT = {
    'i': '%i',
    'u': '%i',
    'f': '%10.6f',
    'c': '%10.6f',
    'V': ''}


# XXX: do i need the extensions provided by volumeutils.make_dt_codes()
data_type_codes = Recoder( dtdefs,  fields=('code', 'label', 'type', 'niistring', 'fmt') )


array_index_order_codes = Recoder((
    (1, "RowMajorOrder", 'C'),
    (2, "ColumnMajorOrder", 'F')),
    fields = ('code', 'label', 'npcode'))

gifti_encoding_codes = Recoder((
    (0, "undef", "GIFTI_ENCODING_UNDEF", "undef"),
    (1, "ASCII", "GIFTI_ENCODING_ASCII", "ASCII"),
    (2, "B64BIN", "GIFTI_ENCODING_B64BIN", "Base64Binary" ),
    (3, "B64GZ", "GIFTI_ENCODING_B64GZ", "GZipBase64Binary"),
    (4, "External", "GIFTI_ENCODING_EXTBIN", "ExternalFileBinary")),
    fields = ('code', 'label', 'giistring', 'specs'))

gifti_endian_codes = Recoder((
    (0, "GIFTI_ENDIAN_UNDEF", "Undef", "undef"),
    (1, "GIFTI_ENDIAN_BIG", "BigEndian", "big"),
    (2, "GIFTI_ENDIAN_LITTLE", "LittleEndian", "little")),
    fields = ('code', 'giistring', 'specs', 'byteorder'))

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
    (1008, 'pointset', (), "NIFTI_INTENT_POINTSET"),
    (1009, 'triangle', (), "NIFTI_INTENT_TRIANGLE"),
    (1010, 'quaternion', (), "NIFTI_INTENT_QUATERNION"),
    (1011, 'dimensionless', (), "NIFTI_INTENT_DIMLESS"),
    (2001, 'time series', (), "NIFTI_INTENT_TIMESERIES"),
    (2002, 'node index', (), "NIFTI_INTENT_NODE_INDEX"),
    (2003, 'rgb vector', (), "NIFTI_INTENT_RGB_VECTOR"),
    (2004, 'rgba vector', (), "NIFTI_INTENT_RGBA_VECTOR"),
    (2005, 'shape', (), "NIFTI_INTENT_SHAPE")),
                       fields=('code', 'label', 'parameters', 'niistring'))


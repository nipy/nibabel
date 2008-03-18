#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Some little helper for dtype conversion."""

__docformat__ = 'restructuredtext'


# the swig wrapper if the NIfTI C library
import nifti.nifticlib as nifticlib
import numpy as N


filetypes = [ 'ANALYZE', 'NIFTI', 'NIFTI_PAIR', 'ANALYZE_GZ', 'NIFTI_GZ',
              'NIFTI_PAIR_GZ' ]
"""Typecodes of all supported NIfTI image formats."""

N2nifti_dtype_map = { N.uint8: nifticlib.NIFTI_TYPE_UINT8,
                      N.int8 : nifticlib.NIFTI_TYPE_INT8,
                      N.uint16: nifticlib.NIFTI_TYPE_UINT16,
                      N.int16 : nifticlib.NIFTI_TYPE_INT16,
                      N.uint32: nifticlib.NIFTI_TYPE_UINT32,
                      N.int32 : nifticlib.NIFTI_TYPE_INT32,
                      N.uint64: nifticlib.NIFTI_TYPE_UINT64,
                      N.int64 : nifticlib.NIFTI_TYPE_INT64,
                      N.float32: nifticlib.NIFTI_TYPE_FLOAT32,
                      N.float64: nifticlib.NIFTI_TYPE_FLOAT64,
                      N.complex128: nifticlib.NIFTI_TYPE_COMPLEX128
                    }
"""Mapping of NumPy datatypes to NIfTI datatypes."""

nifti2numpy_dtype_map = \
    { nifticlib.NIFTI_TYPE_UINT8: 'u1',
      nifticlib.NIFTI_TYPE_INT8: 'i1',
      nifticlib.NIFTI_TYPE_UINT16: 'u2',
      nifticlib.NIFTI_TYPE_INT16: 'i2',
      nifticlib.NIFTI_TYPE_UINT32: 'u4',
      nifticlib.NIFTI_TYPE_INT32: 'i4',
      nifticlib.NIFTI_TYPE_UINT64: 'u8',
      nifticlib.NIFTI_TYPE_INT64: 'i8',
      nifticlib.NIFTI_TYPE_FLOAT32: 'f4',
      nifticlib.NIFTI_TYPE_FLOAT64: 'f8',
      nifticlib.NIFTI_TYPE_COMPLEX128: 'c16'
    }
"""Mapping of NIfTI to NumPy datatypes (necessary for handling memory
mapped array with proper byte-order handling."""


def Ndtype2niftidtype(array):
    """Return the NIfTI datatype id for a corresponding NumPy datatype.
    """
    # get the real datatype from N type dictionary
    dtype = N.typeDict[str(array.dtype)]

    if not N2nifti_dtype_map.has_key(dtype):
        raise ValueError, "Unsupported datatype '%s'" % str(array.dtype)

    return N2nifti_dtype_map[dtype]


def nhdr2dict(nhdr):
    """Convert a NIfTI header struct into a python dictionary.

    While most elements of the header struct will be translated
    1:1 some (e.g. sform matrix) are converted into more convenient
    datatypes (i.e. 4x4 matrix instead of 16 separate values).

    :Parameters:
        nhdr: nifti_1_header

    :Returns:
        dict
    """
    h = {}

    # the following header elements are converted in a simple loop
    # as they do not need special handling
    auto_convert = [ 'session_error', 'extents', 'sizeof_hdr',
                     'slice_duration', 'slice_start', 'xyzt_units',
                     'cal_max', 'intent_p1', 'intent_p2', 'intent_p3',
                     'intent_code', 'sform_code', 'cal_min', 'scl_slope',
                     'slice_code', 'bitpix', 'descrip', 'glmin', 'dim_info',
                     'glmax', 'data_type', 'aux_file', 'intent_name',
                     'vox_offset', 'db_name', 'scl_inter', 'magic',
                     'datatype', 'regular', 'slice_end', 'qform_code',
                     'toffset' ]


    # now just dump all attributes into a dict
    for attr in auto_convert:
        h[attr] = eval('nhdr.' + attr)

    # handle a few special cases
    # handle 'pixdim': carray -> list
    pixdim = nifticlib.floatArray_frompointer(nhdr.pixdim)
    h['pixdim'] = [ pixdim[i] for i in range(8) ]

    # handle dim: carray -> list
    dim = nifticlib.shortArray_frompointer(nhdr.dim)
    h['dim'] = [ dim[i] for i in range(8) ]

    # handle sform: carrays -> (4x4) ndarray
    srow_x = nifticlib.floatArray_frompointer( nhdr.srow_x )
    srow_y = nifticlib.floatArray_frompointer( nhdr.srow_y )
    srow_z = nifticlib.floatArray_frompointer( nhdr.srow_z )

    h['sform'] = N.array( [ [ srow_x[i] for i in range(4) ],
                                [ srow_y[i] for i in range(4) ],
                                [ srow_z[i] for i in range(4) ],
                                [ 0.0, 0.0, 0.0, 1.0 ] ] )

    # handle qform stuff: 3 numbers -> list
    h['quatern'] = [ nhdr.quatern_b, nhdr.quatern_c, nhdr.quatern_d ]
    h['qoffset'] = [ nhdr.qoffset_x, nhdr.qoffset_y, nhdr.qoffset_z ]

    return h


def updateNiftiHeaderFromDict(nhdr, hdrdict):
    """Update a NIfTI header struct with data from a dictionary.

    The supplied dictionary might contain additonal data elements
    that do not match any nifti header element. These are silently ignored.

    Several checks are performed to ensure validity of the resulting
    nifti header struct. If any check fails a ValueError exception will be
    thrown. However, some tests are still missing.

    :Parameters:
        nhdr: nifti_1_header
            To be updated NIfTI header struct (in-place update).
        hdrdict: dict
            Dictionary containing information intented to be merged into
            the NIfTI header struct.
    """
    # this function is still incomplete. add more checks

    if hdrdict.has_key('data_type'):
        if len(hdrdict['data_type']) > 9:
            raise ValueError, \
                  "Nifti header property 'data_type' must not be longer " \
                  + "than 9 characters."
        nhdr.data_type = hdrdict['data_type']
    if hdrdict.has_key('db_name'):
        if len(hdrdict['db_name']) > 79:
            raise ValueError, "Nifti header property 'db_name' must " \
                              + "not be longer than 17 characters."
        nhdr.db_name = hdrdict['db_name']

    if hdrdict.has_key('extents'):
        nhdr.extents = hdrdict['extents']
    if hdrdict.has_key('session_error'):
        nhdr.session_error = hdrdict['session_error']

    if hdrdict.has_key('regular'):
        if len(hdrdict['regular']) > 1:
            raise ValueError, \
                  "Nifti header property 'regular' has to be a single " \
                  + "character."
        nhdr.regular = hdrdict['regular']
    if hdrdict.has_key('dim_info'):
        if len(hdrdict['dim_info']) > 1:
            raise ValueError, \
                  "Nifti header property 'dim_info' has to be a " \
                  + "single character."
        nhdr.dim_info = hdrdict['dim_info']

    if hdrdict.has_key('dim'):
        dim = nifticlib.shortArray_frompointer(nhdr.dim)
        for i in range(8):
            dim[i] = hdrdict['dim'][i]
    if hdrdict.has_key('intent_p1'):
        nhdr.intent_p1 = hdrdict['intent_p1']
    if hdrdict.has_key('intent_p2'):
        nhdr.intent_p2 = hdrdict['intent_p2']
    if hdrdict.has_key('intent_p3'):
        nhdr.intent_p3 = hdrdict['intent_p3']
    if hdrdict.has_key('intent_code'):
        nhdr.intent_code = hdrdict['intent_code']
    if hdrdict.has_key('datatype'):
        nhdr.datatype = hdrdict['datatype']
    if hdrdict.has_key('bitpix'):
        nhdr.bitpix = hdrdict['bitpix']
    if hdrdict.has_key('slice_start'):
        nhdr.slice_start = hdrdict['slice_start']
    if hdrdict.has_key('pixdim'):
        pixdim = nifticlib.floatArray_frompointer(nhdr.pixdim)
        for i in range(8):
            pixdim[i] = hdrdict['pixdim'][i]
    if hdrdict.has_key('vox_offset'):
        nhdr.vox_offset = hdrdict['vox_offset']
    if hdrdict.has_key('scl_slope'):
        nhdr.scl_slope = hdrdict['scl_slope']
    if hdrdict.has_key('scl_inter'):
        nhdr.scl_inter = hdrdict['scl_inter']
    if hdrdict.has_key('slice_end'):
        nhdr.slice_end = hdrdict['slice_end']
    if hdrdict.has_key('slice_code'):
        nhdr.slice_code = hdrdict['slice_code']
    if hdrdict.has_key('xyzt_units'):
        nhdr.xyzt_units = hdrdict['xyzt_units']
    if hdrdict.has_key('cal_max'):
        nhdr.cal_max = hdrdict['cal_max']
    if hdrdict.has_key('cal_min'):
        nhdr.cal_min = hdrdict['cal_min']
    if hdrdict.has_key('slice_duration'):
        nhdr.slice_duration = hdrdict['slice_duration']
    if hdrdict.has_key('toffset'):
        nhdr.toffset = hdrdict['toffset']
    if hdrdict.has_key('glmax'):
        nhdr.glmax = hdrdict['glmax']
    if hdrdict.has_key('glmin'):
        nhdr.glmin = hdrdict['glmin']

    if hdrdict.has_key('descrip'):
        if len(hdrdict['descrip']) > 79:
            raise ValueError, \
                  "Nifti header property 'descrip' must not be longer " \
                  + "than 79 characters."
        nhdr.descrip = hdrdict['descrip']
    if hdrdict.has_key('aux_file'):
        if len(hdrdict['aux_file']) > 23:
            raise ValueError, \
                  "Nifti header property 'aux_file' must not be longer " \
                  + "than 23 characters."
        nhdr.aux_file = hdrdict['aux_file']

    if hdrdict.has_key('qform_code'):
        nhdr.qform_code = hdrdict['qform_code']

    if hdrdict.has_key('sform_code'):
        nhdr.sform_code = hdrdict['sform_code']

    if hdrdict.has_key('quatern'):
        if not len(hdrdict['quatern']) == 3:
            raise ValueError, \
                  "Nifti header property 'quatern' must be float 3-tuple."

        nhdr.quatern_b = hdrdict['quatern'][0]
        nhdr.quatern_c = hdrdict['quatern'][1]
        nhdr.quatern_d = hdrdict['quatern'][2]

    if hdrdict.has_key('qoffset'):
        if not len(hdrdict['qoffset']) == 3:
            raise ValueError, \
                  "Nifti header property 'qoffset' must be float 3-tuple."

        nhdr.qoffset_x = hdrdict['qoffset'][0]
        nhdr.qoffset_y = hdrdict['qoffset'][1]
        nhdr.qoffset_z = hdrdict['qoffset'][2]

    if hdrdict.has_key('sform'):
        if not hdrdict['sform'].shape == (4, 4):
            raise ValueError, \
                  "Nifti header property 'sform' must be 4x4 matrix."

        srow_x = nifticlib.floatArray_frompointer(nhdr.srow_x)
        for i in range(4):
            srow_x[i] = hdrdict['sform'][0][i]
        srow_y = nifticlib.floatArray_frompointer(nhdr.srow_y)
        for i in range(4):
            srow_y[i] = hdrdict['sform'][1][i]
        srow_z = nifticlib.floatArray_frompointer(nhdr.srow_z)
        for i in range(4):
            srow_z[i] = hdrdict['sform'][2][i]

    if hdrdict.has_key('intent_name'):
        if len(hdrdict['intent_name']) > 15:
            raise ValueError, \
                  "Nifti header property 'intent_name' must not be " \
                  + "longer than 15 characters."
        nhdr.intent_name = hdrdict['intent_name']

    if hdrdict.has_key('magic'):
        if hdrdict['magic'] != 'ni1' and hdrdict['magic'] != 'n+1':
            raise ValueError, \
                  "Nifti header property 'magic' must be 'ni1' or 'n+1'."
        nhdr.magic = hdrdict['magic']

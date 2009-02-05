#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utility functions for PyNifti"""

__docformat__ = 'restructuredtext'


import numpy as N
import nifti.clib as ncl

#
# type maps, codes and other NIfTI header stuff
#
filetypes = [ 'ANALYZE', 'NIFTI', 'NIFTI_PAIR', 'ANALYZE_GZ', 'NIFTI_GZ',
              'NIFTI_PAIR_GZ' ]
"""Typecodes of all supported NIfTI image formats."""

N2nifti_dtype_map = { N.uint8: ncl.NIFTI_TYPE_UINT8,
                      N.int8 : ncl.NIFTI_TYPE_INT8,
                      N.uint16: ncl.NIFTI_TYPE_UINT16,
                      N.int16 : ncl.NIFTI_TYPE_INT16,
                      N.uint32: ncl.NIFTI_TYPE_UINT32,
                      N.int32 : ncl.NIFTI_TYPE_INT32,
                      N.uint64: ncl.NIFTI_TYPE_UINT64,
                      N.int64 : ncl.NIFTI_TYPE_INT64,
                      N.float32: ncl.NIFTI_TYPE_FLOAT32,
                      N.float64: ncl.NIFTI_TYPE_FLOAT64,
                      N.complex128: ncl.NIFTI_TYPE_COMPLEX128
                    }
"""Mapping of NumPy datatypes to NIfTI datatypes."""


nifti2numpy_dtype_map = \
    { ncl.NIFTI_TYPE_UINT8: 'u1',
      ncl.NIFTI_TYPE_INT8: 'i1',
      ncl.NIFTI_TYPE_UINT16: 'u2',
      ncl.NIFTI_TYPE_INT16: 'i2',
      ncl.NIFTI_TYPE_UINT32: 'u4',
      ncl.NIFTI_TYPE_INT32: 'i4',
      ncl.NIFTI_TYPE_UINT64: 'u8',
      ncl.NIFTI_TYPE_INT64: 'i8',
      ncl.NIFTI_TYPE_FLOAT32: 'f4',
      ncl.NIFTI_TYPE_FLOAT64: 'f8',
      ncl.NIFTI_TYPE_COMPLEX128: 'c16'
    }
"""Mapping of NIfTI to NumPy datatypes (necessary for handling memory
mapped array with proper byte-order handling."""


nifti_units_map = \
    {"unknown": ncl.NIFTI_UNITS_UNKNOWN,
     "m": ncl.NIFTI_UNITS_METER,
     "mm": ncl.NIFTI_UNITS_MM,
     "um": ncl.NIFTI_UNITS_MICRON,
     "s": ncl.NIFTI_UNITS_SEC,
     "ms": ncl.NIFTI_UNITS_MSEC,
     "us": ncl.NIFTI_UNITS_USEC,
     "Hz": ncl.NIFTI_UNITS_HZ,
     "ppm": ncl.NIFTI_UNITS_PPM,
     "rad/s": ncl.NIFTI_UNITS_RADS,
    }


# encode bits of NIfTI1 standard
valid_xyz_unit_codes = range(8)
valid_time_unit_codes = range(0, 64, 8)

def Ndtype2niftidtype(array):
    """Return the NIfTI datatype id for a corresponding NumPy datatype.
    """
    # get the real datatype from N type dictionary
    dtype = N.typeDict[str(array.dtype)]

    if not N2nifti_dtype_map.has_key(dtype):
        raise ValueError, "Unsupported datatype '%s'" % str(array.dtype)

    return N2nifti_dtype_map[dtype]


nifti_xform_map = \
    { 'unknown': ncl.NIFTI_XFORM_UNKNOWN,
      'scanner': ncl.NIFTI_XFORM_SCANNER_ANAT,
      'aligned': ncl.NIFTI_XFORM_ALIGNED_ANAT,
      'talairach': ncl.NIFTI_XFORM_TALAIRACH,
      'mni152': ncl.NIFTI_XFORM_MNI_152,
    }
nifti_xform_inv_map = dict([(v, k) for k, v in nifti_xform_map.iteritems()])


#
# utility functions
#

def time2vol( t, tr, lag=0.0, decimals=0 ):
    """ Translates a time 't' into a volume number. By default function returns
    the volume number that is closest in time. Volumes are assumed to be
    recorded exactly (and completely) after tr/2, e.g. if 'tr' is 2 secs the
    first volume is recorded at exactly one second.

    't' might be a single value, a sequence or an array.

    The repetition 'tr' might be specified directly, but can also be a 
    NiftiImage object. In the latter case the value of 'tr' is determined from
    the 'rtime' property of the NiftiImage object.

    't' and 'tr' can be given in an arbitrary unit (but both have to be in the
    same unit).

    The 'lag' argument can be used to shift the times by constant offset.

    Please note that numpy.round() is used to round to interger value (rounds
    to even numbers). The 'decimals' argument will be passed to numpy.round().
    """
    # transform to numpy array for easy handling
    tmp = N.array(t)

    # Use rtime from tr if it exists.  This will be true for NiftiImage objects.
    if hasattr(tr, 'rtime'):
        tr = tr.rtime

    vol = N.round( ( tmp + lag + tr/2 ) / tr, decimals )

    return vol


def applyFxToVolumes( ts, vols, fx, **kwargs ):
    """ Apply a function on selected volumes of a timeseries.

    'ts' is a 4d timeseries. It can be a NiftiImage or a ndarray.
    In case of a ndarray one has to make sure that the time is on the
    first axis. 'ts' can actually be of any dimensionality, but datasets aka
    volumes are assumed to be along the first axis.

    'vols' is either a sequence of sequences or a 2d array indicating which 
    volumes fx should be applied to. Each row defines a set of volumes.

    'fx' is a callable function to get an array of the selected volumes as
    argument. Additonal arguments may be specified as keyword arguments and
    are passed to 'fx'.

    The output will be a 4d array with one computed volume per row in the 'vols'
    array.
    """
    # get data array from nifti image or assume data array is
    # already present
    if hasattr(ts, 'data') and isinstance(ts.data, N.ndarray):
        data = ts.data
    else:
        data = ts

    out = []

    for vol in vols:
        out.append( fx( data[ N.array( vol ) ], **kwargs ) )

    return N.array( out )


def getPeristimulusTimeseries( ts, onsetvols, nvols = 10, fx = N.mean ):
    """ Returns 4d array with peristimulus timeseries.

    :Parameters:
      ts:
        source 4d timeseries
      onsetvols:
        sequence of onsetvolumes to be averaged over
      nvols:
        length of the peristimulus timeseries in volumes (starting from
        onsetvol)
      fx:
        function to be applied to the list of corresponding volumes. Typically
        this will be mean(), so it is default, but it could also be var() or
        something different. The supplied function is to be able to handle an
        'axis=0' argument similiar to NumPy's mean(), var(), ...
    """
    selected = [ [ o + offset for o in onsetvols ] \
                    for offset in range( nvols ) ]

    if fx == tuple:
        return applyFxToVolumes( ts, selected, fx )
    else:
        return applyFxToVolumes( ts, selected, fx, axis=0 )


#
# little helpers
#
def nhdr2dict(nhdr, extensions=None):
    """Convert a NIfTI header struct into a python dictionary.

    While most elements of the header struct will be translated
    1:1 some (e.g. sform matrix) are converted into more convenient
    datatypes (i.e. 4x4 matrix instead of 16 separate values).

    :Parameters:
        nhdr: nifti_1_header
        extensions: NiftiExtensions instance
          All extensions will be merged into the returned dictionary
          under the special `extensions` key.

    :Returns:
        dict
    """
    h = {}

    # the following header elements are converted in a simple loop
    # as they do not need special handling
    auto_convert = [ 'session_error', 'extents', 'sizeof_hdr',
                     'slice_duration', 'slice_start',
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
    pixdim = ncl.floatArray_frompointer(nhdr.pixdim)
    h['pixdim'] = [ pixdim[i] for i in range(8) ]

    # handle dim: carray -> list
    dim = ncl.shortArray_frompointer(nhdr.dim)
    h['dim'] = [ dim[i] for i in range(8) ]

    # handle sform: carrays -> (4x4) ndarray
    srow_x = ncl.floatArray_frompointer( nhdr.srow_x )
    srow_y = ncl.floatArray_frompointer( nhdr.srow_y )
    srow_z = ncl.floatArray_frompointer( nhdr.srow_z )

    h['sform'] = N.array( [ [ srow_x[i] for i in range(4) ],
                                [ srow_y[i] for i in range(4) ],
                                [ srow_z[i] for i in range(4) ],
                                [ 0.0, 0.0, 0.0, 1.0 ] ] )

    # handle qform stuff: 3 numbers -> list
    h['quatern'] = [ nhdr.quatern_b, nhdr.quatern_c, nhdr.quatern_d ]
    h['qoffset'] = [ nhdr.qoffset_x, nhdr.qoffset_y, nhdr.qoffset_z ]

    # some more postprocessing
    # expand units
    h['xyz_unit'] = ncl.xyzt2space(nhdr.xyzt_units)
    h['time_unit'] = ncl.xyzt2time(nhdr.xyzt_units)

    if not extensions:
        return h

    #
    # handle extensions
    #
    # simply store a tuple of code (i.e. extension type) and extension data
    h['extensions'] = [e for e in extensions.iteritems()]

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
        dim = ncl.shortArray_frompointer(nhdr.dim)
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
        pixdim = ncl.floatArray_frompointer(nhdr.pixdim)
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
    if hdrdict.has_key('xyz_unit') \
       or hdrdict.has_key('time_unit'):
        # precharge units from current header, in case only one of them is to be
        # updated
        tu = ncl.xyzt2space(nhdr.xyzt_units)
        su = ncl.xyzt2time(nhdr.xyzt_units)

        # overwrite unit if present
        if hdrdict.has_key('xyz_unit'):
            su = _checkUnit(hdrdict['xyz_unit'], valid_xyz_unit_codes)
        if hdrdict.has_key('time_unit'):
            tu = _checkUnit(hdrdict['time_unit'], valid_time_unit_codes)
        # compress both units into hdr format
        nhdr.xyzt_units = ncl.spacetime2xyzt(su, tu)

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

        srow_x = ncl.floatArray_frompointer(nhdr.srow_x)
        for i in range(4):
            srow_x[i] = hdrdict['sform'][0][i]
        srow_y = ncl.floatArray_frompointer(nhdr.srow_y)
        for i in range(4):
            srow_y[i] = hdrdict['sform'][1][i]
        srow_z = ncl.floatArray_frompointer(nhdr.srow_z)
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


def splitFilename(filename):
    """Split a NIfTI filename into basename and extension.

    :Parameters:
        filename: str
            Filename to be split.

    :Returns:
      tuple: (basename, extension)
        The function returns a tuple of basename and extension. If no valid
        NIfTI filename extension is found, the whole string is returned as
        basename and the extension string will be empty.
    """
    parts = filename.split('.')

    if parts[-1] == 'gz':
        if not parts[-2] in [ 'nii', 'hdr', 'img' ]:
            return filename, ''
        else:
            return '.'.join(parts[:-2]), '.'.join(parts[-2:])
    else:
        if not parts[-1] in [ 'nii', 'hdr', 'img' ]:
            return filename, ''
        else:
            return '.'.join(parts[:-1]), parts[-1]


def _checkUnit(value, valid_codes):
    """Internal helper function to check axis units and convert into codes.

    :Parameter:
      value: int | str
      valid_codes: list

    :Returns:
      int

    .. warning::
      This is an internal method. Neither its availability nor its API is
      guarenteed.
    """
    # is it a code?
    if isinstance(value, int):
        if value not in valid_codes:
            raise ValueError, \
                  "'%i' is not a valid NIfTI units code. Should be one of" \
                  " %s." % (value,
                            str([c for c in nifti_units_map.values()
                                    if c in valid_codes]))
        # value seems to be valid
        return value

    # must be plain unit then
    else:
        if value not in nifti_units_map.keys() \
           or nifti_units_map[value] not in valid_codes:
            raise ValueError, \
                  "'%s' is not a valid NIfTI unit. Should be one of" \
                  " %s." % (value,
                            str([c for c in nifti_units_map.keys()
                                    if nifti_units_map[c] in valid_codes]))

        # value seems to be valid
        return nifti_units_map[value]

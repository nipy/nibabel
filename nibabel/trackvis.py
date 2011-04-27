""" Read and write trackvis files
"""
import warnings
import struct
import itertools

import numpy as np
import numpy.linalg as npl

from .py3k import asbytes
from .volumeutils import (native_code, swapped_code, endian_codes,
                          allopen, rec2dict)
from .orientations import aff2axcodes

# Definition of trackvis header structure.
# See http://www.trackvis.org/docs/?subsect=fileformat
# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_1_dtd = [
    ('id_string', 'S6'),
    ('dim', 'h', 3),
    ('voxel_size', 'f4', 3),
    ('origin', 'f4', 3),
    ('n_scalars', 'h'),
    ('scalar_name', 'S20', 10),
    ('n_properties', 'h'),
    ('property_name', 'S20', 10),
    ('reserved', 'S508'),
    ('voxel_order', 'S4'),
    ('pad2', 'S4'),
    ('image_orientation_patient', 'f4', 6),
    ('pad1', 'S2'),
    ('invert_x', 'S1'),
    ('invert_y', 'S1'),
    ('invert_z', 'S1'),
    ('swap_xy', 'S1'),
    ('swap_yz', 'S1'),
    ('swap_zx', 'S1'),
    ('n_count', 'i4'),
    ('version', 'i4'),
    ('hdr_size', 'i4'),
    ]

# Version 2 adds a 4x4 matrix giving the affine transformtation going
# from voxel coordinates in the referenced 3D voxel matrix, to xyz
# coordinates (axes L->R, P->A, I->S).  IF (0 based) value [3, 3] from
# this matrix is 0, this means the matrix is not recorded.
header_2_dtd = [
    ('id_string', 'S6'),
    ('dim', 'h', 3),
    ('voxel_size', 'f4', 3),
    ('origin', 'f4', 3),
    ('n_scalars', 'h'),
    ('scalar_name', 'S20', 10),
    ('n_properties', 'h'),
    ('property_name', 'S20', 10),
    ('vox_to_ras', 'f4', (4,4)), # new field for version 2
    ('reserved', 'S444'),
    ('voxel_order', 'S4'),
    ('pad2', 'S4'),
    ('image_orientation_patient', 'f4', 6),
    ('pad1', 'S2'),
    ('invert_x', 'S1'),
    ('invert_y', 'S1'),
    ('invert_z', 'S1'),
    ('swap_xy', 'S1'),
    ('swap_yz', 'S1'),
    ('swap_zx', 'S1'),
    ('n_count', 'i4'),
    ('version', 'i4'),
    ('hdr_size', 'i4'),
    ]

# Full header numpy dtypes
header_1_dtype = np.dtype(header_1_dtd)
header_2_dtype = np.dtype(header_2_dtd)

# affine to go from DICOM LPS to MNI RAS space
DPCS_TO_TAL = np.diag([-1, -1, 1, 1])


class HeaderError(Exception):
    pass


class DataError(Exception):
    pass


def read(fileobj, as_generator=False):
    ''' Read trackvis file, return streamlines, header

    Parameters
    ----------
    fileobj : string or file-like object
       If string, a filename; otherwise an open file-like object
       pointing to trackvis file (and ready to read from the beginning
       of the trackvis header data)
    as_generator : bool, optional
       Whether to return tracks as sequence (False, default) or as a generator
       (True).

    Returns
    -------
    streamlines : sequence or generator
       Returns sequence if `as_generator` is False, generator if True.  Value is
       sequence or generator of 3 element sequences with elements:

       #. points : ndarray shape (N,3)
          where N is the number of points
       #. scalars : None or ndarray shape (N, M)
          where M is the number of scalars per point
       #. properties : None or ndarray shape (P,)
          where P is the number of properties

    hdr : structured array
       structured array with trackvis header fields

    Notes
    -----
    The endianness of the input data can be deduced from the endianness
    of the returned `hdr` or `streamlines`
    '''
    fileobj = allopen(fileobj, mode='rb')
    hdr_str = fileobj.read(header_2_dtype.itemsize)
    # try defaulting to version 2 format
    hdr = np.ndarray(shape=(),
                     dtype=header_2_dtype,
                     buffer=hdr_str)
    if np.asscalar(hdr['id_string'])[:5] != asbytes('TRACK'):
        raise HeaderError('Expecting TRACK as first '
                          '5 characters of id_string')
    if hdr['hdr_size'] == 1000:
        endianness = native_code
    else:
        hdr = hdr.newbyteorder()
        if hdr['hdr_size'] != 1000:
            raise HeaderError('Invalid hdr_size of %s'
                              % hdr['hdr_size'])
        endianness = swapped_code
    # Check version and adapt structure accordingly
    version = hdr['version']
    if version not in (1, 2):
        raise HeaderError('Reader only supports versions 1 and 2')
    if version == 1: # make a new header with the same data
        hdr = np.ndarray(shape=(),
                         dtype=header_1_dtype,
                         buffer=hdr_str)
        if endianness == swapped_code:
            hdr = hdr.newbyteorder()
    n_s = hdr['n_scalars']
    n_p = hdr['n_properties']
    f4dt = np.dtype(endianness + 'f4')
    pt_cols = 3 + n_s
    pt_size = int(f4dt.itemsize * pt_cols)
    ps_size = int(f4dt.itemsize * n_p)
    i_fmt = endianness + 'i'
    stream_count = hdr['n_count']
    if stream_count < 0:
        raise HeaderError('Unexpected negative n_count')
    def track_gen():
        n_streams = 0
        # For case where there are no scalars or no properties
        scalars = None
        ps = None
        while True:
            n_str = fileobj.read(4)
            if len(n_str) < 4:
                if stream_count:
                    raise HeaderError(
                        'Expecting %s points, found only %s' % (
                                stream_count, n_streams))
                break
            n_pts = struct.unpack(i_fmt, n_str)[0]
            pts_str = fileobj.read(n_pts * pt_size)
            pts = np.ndarray(
                shape = (n_pts, pt_cols),
                dtype = f4dt,
                buffer = pts_str)
            if n_p:
                ps_str = fileobj.read(ps_size)
                ps = np.ndarray(
                    shape = (n_p,),
                    dtype = f4dt,
                    buffer = ps_str)
            xyz = pts[:,:3]
            if n_s:
                scalars = pts[:,3:]
            yield (xyz, scalars, ps)
            n_streams += 1
            # deliberately misses case where stream_count is 0
            if n_streams == stream_count:
                raise StopIteration
    streamlines = track_gen()
    if not as_generator:
        streamlines = list(streamlines)
    return streamlines, hdr


def write(fileobj, streamlines,  hdr_mapping=None, endianness=None):
    ''' Write header and `streamlines` to trackvis file `fileobj`

    The parameters from the streamlines override conflicting parameters
    in the `hdr_mapping` information.  In particular, the number of
    streamlines, the number of scalars, and the number of properties are
    written according to `streamlines` rather than `hdr_mapping`.

    Parameters
    ----------
    fileobj : filename or file-like
       If filename, open file as 'wb', otherwise `fileobj` should be an
       open file-like object, with a ``write`` method.
    streamlines : iterable
       iterable returning 3 element sequences with elements:

       #. points : ndarray shape (N,3)
          where N is the number of points
       #. scalars : None or ndarray shape (N, M)
          where M is the number of scalars per point
       #. properties : None or ndarray shape (P,)
          where P is the number of properties

       If `streamlines` has a ``len`` (for example, it is a list or a tuple),
       then we can write the number of streamlines into the header.  Otherwise
       we write 0 for the number of streamlines (a valid trackvis header) and
       write streamlines into the file until the iterable is exhausted.
       M - the number of scalars - has to be the same for each streamline in
       `streamlines`.  Similarly for P.
    hdr_mapping : None, ndarray or mapping, optional
       Information for filling header fields.  Can be something
       dict-like (implementing ``items``) or a structured numpy array
    endianness : {None, '<', '>'}, optional
       Endianness of file to be written.  '<' is little-endian, '>' is
       big-endian.  None (the default) is to use the endianness of the
       `streamlines` data.

    Returns
    -------
    None

    Examples
    --------
    >>> from StringIO import StringIO #23dt : BytesIO
    >>> file_obj = StringIO() #23dt : BytesIO
    >>> pts0 = np.random.uniform(size=(10,3))
    >>> pts1 = np.random.uniform(size=(10,3))
    >>> streamlines = ([(pts0, None, None), (pts1, None, None)])
    >>> write(file_obj, streamlines)
    >>> _ = file_obj.seek(0) # returns 0 in python 3
    >>> streams, hdr = read(file_obj)
    >>> len(streams)
    2

    If there are too many streamlines to fit in memory, you can pass an iterable
    thing instead of a list

    >>> file_obj = StringIO() #23dt : BytesIO
    >>> def gen():
    ...     yield (pts0, None, None)
    ...     yield (pts0, None, None)
    >>> write(file_obj, gen())
    >>> _ = file_obj.seek(0)
    >>> streams, hdr = read(file_obj)
    >>> len(streams)
    2
    '''
    stream_iter = iter(streamlines)
    try:
        streams0 = stream_iter.next()
    except StopIteration: # empty sequence or iterable
        streams0 = None
    if endianness is None:
        if streams0 is None:
            endianness = native_code
        else: # At least one streamline
            endianness = endian_codes[streams0[0].dtype.byteorder]
    # fill in a new header from mapping-like
    hdr = _hdr_from_mapping(None, hdr_mapping, endianness)
    # Try and get number of streams from streamlines.  If this is an iterable,
    # we don't have a len, so we write 0 for length.  This is a valid trackvis
    # header meaning, keep reading until you run out of data 
    try:
        n_streams = len(streamlines)
    except TypeError: # iterable; we don't know the number of streams
        n_streams = 0
    hdr['n_count'] = n_streams
    # If there are streamlines, get number of scalars and properties
    if not streams0 is None:
        pts, scalars, props = streams0
        # calculate number of scalars
        if not scalars is None:
            n_s = scalars.shape[1]
        else:
            n_s = 0
        hdr['n_scalars'] = n_s
        # calculate number of properties
        if not props is None:
            n_p = props.size
            hdr['n_properties'] = n_p
        else:
            n_p = 0
    # write header
    fileobj = allopen(fileobj, mode='wb')
    fileobj.write(hdr.tostring())
    if streams0 is None:
        return
    f4dt = np.dtype(endianness + 'f4')
    i_fmt = endianness + 'i'
    # Add back the read first streamline to the sequence
    for pts, scalars, props in itertools.chain([streams0], stream_iter):
        n_pts, n_coords = pts.shape
        if n_coords != 3:
            raise ValueError('pts should have 3 columns')
        fileobj.write(struct.pack(i_fmt, n_pts))
        # This call ensures that the data are 32-bit floats, and that
        # the endianness is OK.
        if pts.dtype != f4dt:
            pts = pts.astype(f4dt)
        if n_s == 0:
            if not (scalars is None or len(scalars) == 0):
                raise DataError('Expecting 0 scalars per point')
        else:
            if scalars.shape != (n_pts, n_s):
                raise DataError('Scalars should be shape (%s, %s)'
                                 % (n_pts, n_s))
            if scalars.dtype != f4dt:
                scalars = scalars.astype(f4dt)
                pts = np.c_[pts, scalars]
        fileobj.write(pts.tostring())
        if n_p == 0:
            if not (props is None or len(props) == 0):
                raise DataError('Expecting 0 properties per point')
        else:
            if props.size != n_p:
                raise DataError('Properties should be size %s' % n_p)
            if props.dtype != f4dt:
                props = props.astype(f4dt)
            fileobj.write(props.tostring())


def _hdr_from_mapping(hdr=None, mapping=None, endianness=native_code):
    ''' Fill `hdr` from mapping `mapping`, with given endianness '''
    if hdr is None:
        # passed a valid mapping as header?  Copy and return
        if isinstance(mapping, np.ndarray):
            test_dtype = mapping.dtype.newbyteorder('=')
            if test_dtype in (header_1_dtype, header_2_dtype):
                return mapping.copy()
        # otherwise make a new empty header.   If no version specified,
        # go for default (2)
        if mapping is None:
            version = 2
        else:
            version =  mapping.get('version', 2)
        hdr = empty_header(endianness, version)
    if mapping is None:
        return hdr
    if isinstance(mapping, np.ndarray):
        mapping = rec2dict(mapping)
    for key, value in mapping.items():
        hdr[key] = value
    # check header values
    if np.asscalar(hdr['id_string'])[:5] != asbytes('TRACK'):
        raise HeaderError('Expecting TRACK as first '
                          '5 characaters of id_string')
    if hdr['version'] not in (1, 2):
        raise HeaderError('Reader only supports version 1')
    if hdr['hdr_size'] != 1000:
        raise HeaderError('hdr_size should be 1000')
    return hdr


def empty_header(endianness=None, version=2):
    ''' Empty trackvis header

    Parameters
    ----------
    endianness : {'<','>'}, optional
       Endianness of empty header to return. Default is native endian.
    version : int, optional
       Header version.  1 or 2.  Default is 2

    Returns
    -------
    hdr : structured array
       structured array containing empty trackvis header

    Examples
    --------
    >>> hdr = empty_header()
    >>> print hdr['version']
    2
    >>> np.asscalar(hdr['id_string']) #23dt next : bytes
    'TRACK'
    >>> endian_codes[hdr['version'].dtype.byteorder] == native_code
    True
    >>> hdr = empty_header(swapped_code)
    >>> endian_codes[hdr['version'].dtype.byteorder] == swapped_code
    True
    >>> hdr = empty_header(version=1)
    >>> print hdr['version']
    1

    Notes
    -----
    The trackviz header can store enough information to give an affine
    mapping between voxel and world space.  Often this information is
    missing.  We make no attempt to fill it with sensible defaults on
    the basis that, if the information is missing, it is better to be
    explicit.
    '''
    if version == 1:
        dt = header_1_dtype
    elif version == 2:
        dt = header_2_dtype
    else:
        raise HeaderError('Header version should be 1 or 2')
    if endianness:
        dt = dt.newbyteorder(endianness)
    hdr = np.zeros((), dtype=dt)
    hdr['id_string'] = 'TRACK'
    hdr['version'] = version
    hdr['hdr_size'] = 1000
    return hdr


def aff_from_hdr(trk_hdr, atleast_v2=None):
    ''' Return voxel to mm affine from trackvis header

    Affine is mapping from voxel space to Nifti (RAS) output coordinate
    system convention; x: Left -> Right, y: Posterior -> Anterior, z:
    Inferior -> Superior.

    Parameters
    ----------
    trk_hdr : mapping
       Mapping with trackvis header keys ``version``. If ``version == 2``, we
       also expect ``vox_to_ras``.
    atleast_v2 : None or bool
        If None, currently defaults to False.  This will change to True in
        future versions.  If True, require that there is a valid 'vox_to_ras'
        affine, raise HeaderError otherwise.  If False, look for valid
        'vox_to_ras' affine, but fall back to best guess from version 1 fields
        otherwise.

    Returns
    -------
    aff : (4,4) array
       affine giving mapping from voxel coordinates (affine applied on
       the left to points on the right) to millimeter coordinates in the
       RAS coordinate system

    Notes
    -----
    Our initial idea was to try and work round the deficiencies of the version 1
    format by using the DICOM orientation fields to store the affine.  This
    proved difficult in practice because trackvis (the application) doesn't
    allow negative voxel sizes (needed for recording axis flips) and sets the
    origin field to 0. In future, we'll raise an error rather than try and
    estimate the affine from version 1 fields
    '''
    if atleast_v2 is None:
        warnings.warn('Defaulting to `atleast_v2` of False.  Future versions '
                      'will default to True',
                      FutureWarning,
                      stacklevel=2)
        atleast_v2 = False
    if trk_hdr['version'] == 2:
        aff = trk_hdr['vox_to_ras']
        if aff[3,3] != 0:
            return aff
        if atleast_v2:
            raise HeaderError('Requiring version 2 affine and this affine is '
                              'not valid')
    # Now we are in the dark world of the DICOM fields.  We might have made this
    # one ourselves, in which case the origin might be set, and it might have
    # negative voxel sizes
    aff = np.eye(4)
    # The IOP field has only two of the three columns we need
    iop = trk_hdr['image_orientation_patient'].reshape(2,3).T
    # R might be a rotation matrix (and so completed by the cross product of the
    # first two columns), or it might be an orthogonal matrix with negative
    # determinant. We try pure rotation first
    R = np.c_[iop, np.cross(*iop.T)]
    vox = trk_hdr['voxel_size']
    aff[:3,:3] = R * vox
    aff[:3,3] = trk_hdr['origin']
    aff = np.dot(DPCS_TO_TAL, aff)
    # Next we check against the 'voxel_order' field if present and not empty.
    try:
        voxel_order = trk_hdr['voxel_order']
    except KeyError, ValueError:
        voxel_order = ''
    if voxel_order == '':
        return aff
    # If the voxel_order conflicts with the affine by one flip, this may have
    # been a negative determinant affine saved with positive voxel sizes
    exp_order = ''.join(aff2axcodes(aff))
    if voxel_order != exp_order:
        # If first pass doesn't match, try flipping the (estimated) third column
        aff[:,2] *= -1
        exp_order = ''.join(aff2axcodes(aff))
        if voxel_order != exp_order:
            raise HeaderError('Estimate of header affine does not match '
                              'voxel_order of %s' % exp_order)
    return aff


def aff_to_hdr(affine, trk_hdr, pos_vox=None, set_order=None):
    ''' Set affine `affine` into trackvis header `trk_hdr`

    Affine is mapping from voxel space to Nifti RAS) output coordinate
    system convention; x: Left -> Right, y: Posterior -> Anterior, z:
    Inferior -> Superior.  Sets affine if possible, and voxel sizes, and voxel
    axis ordering.

    Parameters
    ----------
    affine : (4,4) array-like
       Affine voxel to mm transformation
    trk_hdr : mapping
       Mapping implementing __setitem__
    pos_vos : None or bool
        If None, currently defaults to False - this will change in future
        versions of nibabel.  If False, allow negative voxel sizes in header to
        record axis flips.  Negative voxels cause problems for trackvis (the
        application).  If True, enforce positive voxel sizes.
    set_order : None or bool
        If None, currently defaults to False - this will change in future
        versions of nibabel.  If False, do not set ``voxel_order`` field in
        `trk_hdr`.  If True, calculcate ``voxel_order`` from `affine` and set
        into `trk_hdr`.

    Returns
    -------
    None

    Notes
    -----
    version 2 of the trackvis header has a dedicated field for the nifti RAS
    affine. In theory trackvis 1 has enough information to store an affine, with
    the fields 'origin', 'voxel_size' and 'image_orientation_patient'.
    Unfortunately, to be able to store any affine, we'd need to be able to set
    negative voxel sizes, to encode axis flips. This is because
    'image_orientation_patient' is only two columns of the 3x3 rotation matrix,
    and we need to know the number of flips to reconstruct the third column
    reliably.  It turns out that negative flips upset trackvis (the
    application).  The application also ignores the origin field, and may not
    use the 'image_orientation_patient' field.
    '''
    if pos_vox is None:
        warnings.warn('Default for ``pos_vox`` will change to True in '
                      'future versions of nibabel',
                      FutureWarning,
                      stacklevel=2)
        pos_vox = False
    if set_order is None:
        warnings.warn('Default for ``set_order`` will change to True in '
                      'future versions of nibabel',
                      FutureWarning,
                      stacklevel=2)
        set_order = False
    try:
        version = trk_hdr['version']
    except KeyError, ValueError: # dict or structured array
        version = 2
    if version == 2:
        trk_hdr['vox_to_ras'] = affine
    if set_order:
        trk_hdr['voxel_order'] = ''.join(aff2axcodes(affine))
    # Now on dodgy ground with DICOM fields in header
    # RAS to DPCS output
    affine = np.dot(DPCS_TO_TAL, affine)
    trans = affine[:3, 3]
    # Get zooms
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    RS = RZS / zooms
    # If you said we could, adjust zooms to make RS correspond (below) to a true
    # rotation matrix.  We need to set the sign of one of the zooms to deal with
    # this.  Trackvis (the application) doesn't like negative zooms at all, so
    # you might want to disallow this with the pos_vox option.
    if not pos_vox and npl.det(RS) < 0:
        zooms[0] *= -1
        RS[:,0] *= -1
    # retrieve rotation matrix from RS with polar decomposition.
    # Discard shears because we cannot store them.
    P, S, Qs = npl.svd(RS)
    R = np.dot(P, Qs)
    # it's an orthogonal matrix
    assert np.allclose(np.dot(R, R.T), np.eye(3))
    # set into header
    trk_hdr['origin'] = trans
    trk_hdr['voxel_size'] = zooms
    trk_hdr['image_orientation_patient'] = R[:,0:2].T.ravel()


class TrackvisFileError(Exception):
    pass


class TrackvisFile(object):
    ''' Convenience class to encapsulate trackviz file information

    Parameters
    ----------
    streamlines : sequence
       sequence of streamlines.  This object does not accept generic iterables
       as input because these can be consumed and make the object unusable.
       Please use the function interface to work with generators / iterables
    mapping : None or mapping
       Mapping defining header attributes
    endianness : {None, '<', '>'}
       Set here explicit endianness if required.  Endianness otherwise inferred
       from `streamlines`
    filename : None or str
       filename
    '''
    def __init__(self,
                 streamlines,
                 mapping=None,
                 endianness=None,
                 filename=None):
        try:
            n_streams = len(streamlines)
        except TypeError:
            raise TrackvisFileError('Need sequence for streamlines input')
        self.streamlines = streamlines
        if endianness is None:
            if n_streams > 0:
                endianness = endian_codes[streamlines[0].dtype.byteorder]
            else:
                endianness = native_code
        self.header = _hdr_from_mapping(None, mapping, endianness)
        self.endianness = endianness
        self.filename = filename

    @classmethod
    def from_file(klass, file_like):
        streamlines, header = read(file_like)
        filename = (file_like if isinstance(file_like, basestring)
                    else None)
        return klass(streamlines, header, None, filename)

    def to_file(self, file_like):
        write(file_like, self.streamlines, self.header, self.endianness)
        self.filename = (file_like if isinstance(file_like, basestring)
                         else None)

    def get_affine(self, atleast_v2=None):
        """ Get affine from header in object

        Returns
        -------
        aff : (4,4) ndarray
            affine from header
        atleast_v2 : None or bool, optional
            See ``aff_from_hdr`` docstring for detail.  If True, require valid
            affine in ``vox_to_ras`` field of header.

        Notes
        -----
        This method currently works for trackvis version 1 headers, but we
        consider it unsafe for version 1 headers, and in future versions of
        nibabel we will raise an error for trackvis headers < version 2.
        """
        if atleast_v2 is None:
            warnings.warn('Defaulting to `atleast_v2` of False.  Future versions '
                          'will default to True',
                          FutureWarning,
                          stacklevel=2)
            atleast_v2 = False
        return aff_from_hdr(self.header, atleast_v2)

    def set_affine(self, affine, pos_vox=None, set_order=None):
        """ Set affine `affine` into trackvis header

        Affine is mapping from voxel space to Nifti RAS) output coordinate
        system convention; x: Left -> Right, y: Posterior -> Anterior, z:
        Inferior -> Superior.  Sets affine if possible, and voxel sizes, and voxel
        axis ordering.

        Parameters
        ----------
        affine : (4,4) array-like
            Affine voxel to mm transformation
        pos_vos : None or bool, optional
            If None, currently defaults to False - this will change in future
            versions of nibabel.  If False, allow negative voxel sizes in header to
            record axis flips.  Negative voxels cause problems for trackvis (the
            application).  If True, enforce positive voxel sizes.
        set_order : None or bool, optional
            If None, currently defaults to False - this will change in future
            versions of nibabel.  If False, do not set ``voxel_order`` field in
            `trk_hdr`.  If True, calculcate ``voxel_order`` from `affine` and set
            into `trk_hdr`.

        Returns
        -------
        None
        """
        if pos_vox is None:
            warnings.warn('Default for ``pos_vox`` will change to True in '
                          'future versions of nibabel',
                          FutureWarning,
                          stacklevel=2)
            pos_vox = False
        if set_order is None:
            warnings.warn('Default for ``set_order`` will change to True in '
                          'future versions of nibabel',
                          FutureWarning,
                          stacklevel=2)
            set_order = False
        return aff_to_hdr(affine, self.header, pos_vox, set_order)

from __future__ import division

# Definition of trackvis header structure:
# http://www.trackvis.org/docs/?subsect=fileformat

import os
import struct
import warnings
import string

import numpy as np
import nibabel as nib

from nibabel.openers import Opener
from nibabel.py3k import asstr
from nibabel.volumeutils import (native_code, swapped_code)
from nibabel.orientations import (aff2axcodes, axcodes2ornt)

from .array_sequence import create_arraysequences_from_generator
from .tractogram_file import TractogramFile
from .tractogram_file import DataError, HeaderError, HeaderWarning
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field


MAX_NB_NAMED_SCALARS_PER_POINT = 10
MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE = 10

# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_1_dtd = [(Field.MAGIC_NUMBER, 'S6'),
                (Field.DIMENSIONS, 'h', 3),
                (Field.VOXEL_SIZES, 'f4', 3),
                (Field.ORIGIN, 'f4', 3),
                (Field.NB_SCALARS_PER_POINT, 'h'),
                ('scalar_name', 'S20', MAX_NB_NAMED_SCALARS_PER_POINT),
                (Field.NB_PROPERTIES_PER_STREAMLINE, 'h'),
                ('property_name', 'S20',
                 MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE),
                ('reserved', 'S508'),
                (Field.VOXEL_ORDER, 'S4'),
                ('pad2', 'S4'),
                ('image_orientation_patient', 'f4', 6),
                ('pad1', 'S2'),
                ('invert_x', 'S1'),
                ('invert_y', 'S1'),
                ('invert_z', 'S1'),
                ('swap_xy', 'S1'),
                ('swap_yz', 'S1'),
                ('swap_zx', 'S1'),
                (Field.NB_STREAMLINES, 'i4'),
                ('version', 'i4'),
                ('hdr_size', 'i4'),
                ]

# Version 2 adds a 4x4 matrix giving the affine transformtation going
# from voxel coordinates in the referenced 3D voxel matrix, to xyz
# coordinates (axes L->R, P->A, I->S). If (0 based) value [3, 3] from
# this matrix is 0, this means the matrix is not recorded.
header_2_dtd = [(Field.MAGIC_NUMBER, 'S6'),
                (Field.DIMENSIONS, 'h', 3),
                (Field.VOXEL_SIZES, 'f4', 3),
                (Field.ORIGIN, 'f4', 3),
                (Field.NB_SCALARS_PER_POINT, 'h'),
                ('scalar_name', 'S20', MAX_NB_NAMED_SCALARS_PER_POINT),
                (Field.NB_PROPERTIES_PER_STREAMLINE, 'h'),
                ('property_name', 'S20',
                 MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE),
                (Field.VOXEL_TO_RASMM, 'f4', (4, 4)),  # New in version 2.
                ('reserved', 'S444'),
                (Field.VOXEL_ORDER, 'S4'),
                ('pad2', 'S4'),
                ('image_orientation_patient', 'f4', 6),
                ('pad1', 'S2'),
                ('invert_x', 'S1'),
                ('invert_y', 'S1'),
                ('invert_z', 'S1'),
                ('swap_xy', 'S1'),
                ('swap_yz', 'S1'),
                ('swap_zx', 'S1'),
                (Field.NB_STREAMLINES, 'i4'),
                ('version', 'i4'),
                ('hdr_size', 'i4'),
                ]

# Full header numpy dtypes
header_1_dtype = np.dtype(header_1_dtd)
header_2_dtype = np.dtype(header_2_dtd)


def get_affine_trackvis_to_rasmm(header):
    """ Get affine mapping trackvis voxelmm space to RAS+ mm space

    The streamlines in a trackvis file are in 'voxelmm' space, where the
    coordinates refer to the corner of the voxel.

    Compute the affine matrix that will bring them back to RAS+ mm space, where
    the coordinates refer to the center of the voxel.

    Parameters
    ----------
    header : dict
        Dict containing trackvis header.

    Returns
    -------
    aff_tv2ras : shape (4, 4) array
        Affine array mapping coordinates in 'voxelmm' space to RAS+ mm space.
    """
    # TRK's streamlines are in 'voxelmm' space, we will compute the
    # affine matrix that will bring them back to RAS+ and mm space.
    affine = np.eye(4)

    # The affine matrix found in the TRK header requires the points to
    # be in the voxel space.
    # voxelmm -> voxel
    scale = np.eye(4)
    scale[range(3), range(3)] /= header[Field.VOXEL_SIZES]
    affine = np.dot(scale, affine)

    # TrackVis considers coordinate (0,0,0) to be the corner of the
    # voxel whereas streamlines returned assumes (0,0,0) to be the
    # center of the voxel. Thus, streamlines are shifted by half a voxel.
    offset = np.eye(4)
    offset[:-1, -1] -= 0.5
    affine = np.dot(offset, affine)

    # If the voxel order implied by the affine does not match the voxel
    # order in the TRK header, change the orientation.
    # voxel (header) -> voxel (affine)
    header_ornt = asstr(header[Field.VOXEL_ORDER])
    affine_ornt = "".join(aff2axcodes(header[Field.VOXEL_TO_RASMM]))
    header_ornt = axcodes2ornt(header_ornt)
    affine_ornt = axcodes2ornt(affine_ornt)
    ornt = nib.orientations.ornt_transform(header_ornt, affine_ornt)
    M = nib.orientations.inv_ornt_aff(ornt, header[Field.DIMENSIONS])
    affine = np.dot(M, affine)

    # Applied the affine found in the TRK header.
    # voxel -> rasmm
    voxel_to_rasmm = header[Field.VOXEL_TO_RASMM]
    affine_voxmm_to_rasmm = np.dot(voxel_to_rasmm, affine)
    return affine_voxmm_to_rasmm.astype(np.float32)


def get_affine_rasmm_to_trackvis(header):
    return np.linalg.inv(get_affine_trackvis_to_rasmm(header))


def encode_value_in_name(value, name, max_name_len=20):
    """ Return `name` as fixed-length string, appending `value` as string.

    Form output from `name` if `value <= 1` else `name` + ``\x00`` +
    str(value).

    Return output as fixed length string length `max_name_len`, padded with
    ``\x00``.

    This function also verifies that the modified length of name is less than
    `max_name_len`.

    Parameters
    ----------
    value : int
        Integer value to encode.
    name : str
        Name to which we may append an ascii / latin-1 representation of
        `value`.
    max_name_len : int, optional
        Maximum length of byte string that output can have.

    Returns
    -------
    encoded_name : bytes
        Name maybe followed by ``\x00`` and ascii / latin-1 representation of
        `value`, padded with ``\x00`` bytes.
    """
    if len(name) > max_name_len:
        msg = ("Data information named '{0}' is too long"
               " (max {1} characters.)").format(name, max_name_len)
        raise ValueError(msg)
    encoded_name = name if value <= 1 else name + '\x00' + str(value)
    if len(encoded_name) > max_name_len:
        msg = ("Data information named '{0}' is too long (need to be less"
               " than {1} characters when storing more than one value"
               " for a given data information."
               ).format(name, max_name_len - (len(str(value)) + 1))
        raise ValueError(msg)
    # Fill to the end with zeros
    return encoded_name.ljust(max_name_len, '\x00').encode('latin1')


def decode_value_from_name(encoded_name):
    """ Decodes a value that has been encoded in the last bytes of a string.

    Check :func:`encode_value_in_name` to see how the value has been encoded.

    Parameters
    ----------
    encoded_name : bytes
        Name in which a value has been encoded or not.

    Returns
    -------
    name : bytes
        Name without the encoded value.
    value : int
        Value decoded from the name.
    """
    encoded_name = asstr(encoded_name)
    if len(encoded_name) == 0:
        return encoded_name, 0

    splits = encoded_name.rstrip('\x00').split('\x00')
    name = splits[0]
    value = 1

    if len(splits) == 2:
        value = int(splits[1])  # Decode value.
    elif len(splits) > 2:
        # The remaining bytes are not \x00, raising.
        msg = ("Wrong scalar_name or property_name: '{0}'."
               " Unused characters should be \\x00.").format(encoded_name)
        raise HeaderError(msg)

    return name, value


def create_empty_header():
    """ Return an empty compliant TRK header. """
    header = np.zeros(1, dtype=header_2_dtype)

    # Default values
    header[Field.MAGIC_NUMBER] = TrkFile.MAGIC_NUMBER
    header[Field.VOXEL_SIZES] = np.array((1, 1, 1), dtype="f4")
    header[Field.DIMENSIONS] = np.array((1, 1, 1), dtype="h")
    header[Field.VOXEL_TO_RASMM] = np.eye(4, dtype="f4")
    header[Field.VOXEL_ORDER] = b"RAS"
    header['version'] = 2
    header['hdr_size'] = TrkFile.HEADER_SIZE

    return header


class TrkFile(TractogramFile):
    """ Convenience class to encapsulate TRK file format.

    Notes
    -----

    TrackVis (so its file format: TRK) considers the streamline coordinate
    (0,0,0) to be in the corner of the voxel whereas NiBabel's streamlines
    internal representation (Voxel space) assumes (0,0,0) to be in the
    center of the voxel.

    Thus, streamlines are shifted by half a voxel on load and are shifted
    back on save.
    """

    # Constants
    MAGIC_NUMBER = b"TRACK"
    HEADER_SIZE = 1000
    SUPPORTS_DATA_PER_POINT = True
    SUPPORTS_DATA_PER_STREAMLINE = True

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : :class:`Tractogram` object
            Tractogram that will be contained in this :class:`TrkFile`.

        header : dict, optional
            Metadata associated to this tractogram file.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+*
        and *mm* space where coordinate (0,0,0) refers to the center
        of the voxel.
        """
        if header is None:
            header_rec = create_empty_header()
            header = dict(zip(header_rec.dtype.names, header_rec[0]))

        super(TrkFile, self).__init__(tractogram, header)

    @classmethod
    def is_correct_format(cls, fileobj):
        """ Check if the file is in TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data). Note that calling this function
            does not change the file position.

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is compatible with TRK format,
            otherwise returns False.
        """
        with Opener(fileobj) as f:
            magic_len = len(cls.MAGIC_NUMBER)
            magic_number = f.read(magic_len)
            f.seek(-magic_len, os.SEEK_CUR)
            return magic_number == cls.MAGIC_NUMBER

    @classmethod
    def load(cls, fileobj, lazy_load=False):
        """ Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.
        lazy_load : {False, True}, optional
            If True, load streamlines in a lazy manner i.e. they will not be
            kept in memory. Otherwise, load all streamlines in memory.

        Returns
        -------
        trk_file : :class:`TrkFile` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the returned tractogram are assumed to be in *RAS*
        and *mm* space where coordinate (0,0,0) refers to the center of the
        voxel.
        """
        hdr = cls._read_header(fileobj)

        # Find scalars and properties name
        data_per_point_slice = {}
        if hdr[Field.NB_SCALARS_PER_POINT] > 0:
            cpt = 0
            for scalar_name in hdr['scalar_name']:
                scalar_name, nb_scalars = decode_value_from_name(scalar_name)

                if nb_scalars == 0:
                    continue

                slice_obj = slice(cpt, cpt + nb_scalars)
                data_per_point_slice[scalar_name] = slice_obj
                cpt += nb_scalars

            if cpt < hdr[Field.NB_SCALARS_PER_POINT]:
                slice_obj = slice(cpt, hdr[Field.NB_SCALARS_PER_POINT])
                data_per_point_slice['scalars'] = slice_obj

        data_per_streamline_slice = {}
        if hdr[Field.NB_PROPERTIES_PER_STREAMLINE] > 0:
            cpt = 0
            for property_name in hdr['property_name']:
                results = decode_value_from_name(property_name)
                property_name, nb_properties = results

                if nb_properties == 0:
                    continue

                slice_obj = slice(cpt, cpt + nb_properties)
                data_per_streamline_slice[property_name] = slice_obj
                cpt += nb_properties

            if cpt < hdr[Field.NB_PROPERTIES_PER_STREAMLINE]:
                slice_obj = slice(cpt, hdr[Field.NB_PROPERTIES_PER_STREAMLINE])
                data_per_streamline_slice['properties'] = slice_obj

        if lazy_load:
            def _read():
                for pts, scals, props in cls._read(fileobj, hdr):
                    items = data_per_point_slice.items()
                    data_for_points = dict((k, scals[:, v]) for k, v in items)
                    items = data_per_streamline_slice.items()
                    data_for_streamline = dict((k, props[v]) for k, v in items)
                    yield TractogramItem(pts,
                                         data_for_streamline,
                                         data_for_points)

            tractogram = LazyTractogram.from_data_func(_read)

        else:
            trk_reader = cls._read(fileobj, hdr)
            arr_seqs = create_arraysequences_from_generator(trk_reader, n=3)
            streamlines, scalars, properties = arr_seqs
            properties = np.asarray(properties)  # Actually a 2d array.
            tractogram = Tractogram(streamlines)

            for name, slice_ in data_per_point_slice.items():
                tractogram.data_per_point[name] = scalars[:, slice_]

            for name, slice_ in data_per_streamline_slice.items():
                tractogram.data_per_streamline[name] = properties[:, slice_]

        tractogram.affine_to_rasmm = get_affine_trackvis_to_rasmm(hdr)
        tractogram = tractogram.to_world()

        return cls(tractogram, header=hdr)

    def save(self, fileobj):
        """ Save tractogram to a filename or file-like object using TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to write from the beginning
            of the TRK header data).
        """
        # Enforce little-endian byte order for header
        header = create_empty_header().newbyteorder('<')

        # Override hdr's fields by those contained in `header`.
        for k, v in self.header.items():
            if k in header_2_dtype.fields.keys():
                header[k] = v

        # By default, the voxel order is LPS.
        # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
        if header[Field.VOXEL_ORDER] == b"":
            header[Field.VOXEL_ORDER] = b"LPS"

        # Keep counts for correcting incoherent fields or warn.
        nb_streamlines = 0
        nb_points = 0
        nb_scalars = 0
        nb_properties = 0

        header = header[0]
        with Opener(fileobj, mode="wb") as f:
            # Keep track of the beginning of the header.
            beginning = f.tell()

            # Write temporary header that we will update at the end
            f.write(header.tostring())

            i4_dtype = np.dtype("<i4")  # Always save in little-endian.
            f4_dtype = np.dtype("<f4")  # Always save in little-endian.

            try:
                first_item = next(iter(self.tractogram))
            except StopIteration:
                # Empty tractogram
                header[Field.NB_STREAMLINES] = 0
                header[Field.NB_SCALARS_PER_POINT] = 0
                header[Field.NB_PROPERTIES_PER_STREAMLINE] = 0
                # Overwrite header with updated one.
                f.seek(beginning, os.SEEK_SET)
                f.write(header.tostring())
                return

            # Update field 'property_name' using 'data_per_streamline'.
            data_for_streamline = first_item.data_for_streamline
            if len(data_for_streamline) > MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE:
                msg = ("Can only store {0} named data_per_streamline (also"
                       " known as 'properties' in the TRK format)."
                       ).format(MAX_NB_NAMED_SCALARS_PER_POINT)
                raise ValueError(msg)

            data_for_streamline_keys = sorted(data_for_streamline.keys())
            property_name = np.zeros(MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE,
                                     dtype='S20')
            for i, name in enumerate(data_for_streamline_keys):
                # Append number of values as ascii to zero-terminated name
                # to encode number of values into trackvis name.
                nb_values = data_for_streamline[name].shape[-1]
                property_name[i] = encode_value_in_name(nb_values, name)
            header['property_name'][:] = property_name

            # Update field 'scalar_name' using 'tractogram.data_per_point'.
            data_for_points = first_item.data_for_points
            if len(data_for_points) > MAX_NB_NAMED_SCALARS_PER_POINT:
                msg = ("Can only store {0} named data_per_point (also known"
                       " as 'scalars' in the TRK format)."
                       ).format(MAX_NB_NAMED_SCALARS_PER_POINT)
                raise ValueError(msg)

            data_for_points_keys = sorted(data_for_points.keys())
            scalar_name = np.zeros(MAX_NB_NAMED_SCALARS_PER_POINT, dtype='S20')
            for i, name in enumerate(data_for_points_keys):
                # Append number of values as ascii to zero-terminated name
                # to encode number of values into trackvis name.
                nb_values = data_for_points[name].shape[-1]
                scalar_name[i] = encode_value_in_name(nb_values, name)
            header['scalar_name'][:] = scalar_name

            # Make sure streamlines are in rasmm then send them to voxmm.
            tractogram = self.tractogram.to_world(lazy=True)
            affine_to_trackvis = get_affine_rasmm_to_trackvis(header)
            tractogram = tractogram.apply_affine(affine_to_trackvis, lazy=True)

            for t in tractogram:
                if any((len(d) != len(t.streamline)
                        for d in t.data_for_points.values())):
                    raise DataError("Missing scalars for some points!")

                points = np.asarray(t.streamline)
                scalars = [np.asarray(t.data_for_points[k])
                           for k in data_for_points_keys]
                scalars = np.concatenate([np.ndarray((len(points), 0),)
                                          ] + scalars, axis=1)
                properties = [np.asarray(t.data_for_streamline[k])
                              for k in data_for_streamline_keys]
                properties = np.concatenate(
                    [np.array([])] + properties).astype(f4_dtype)

                data = struct.pack(i4_dtype.str[:-1], len(points))
                pts_scalars = np.concatenate(
                    [points, scalars], axis=1).astype(f4_dtype)
                data += pts_scalars.tostring()
                data += properties.tostring()
                f.write(data)

                nb_streamlines += 1
                nb_points += len(points)
                nb_scalars += scalars.size
                nb_properties += len(properties)

            # Use those values to update the header.
            nb_scalars_per_point = nb_scalars / nb_points
            nb_properties_per_streamline = nb_properties / nb_streamlines

            # Check for errors
            if nb_scalars_per_point != int(nb_scalars_per_point):
                msg = "Nb. of scalars differs from one point to another!"
                raise DataError(msg)

            if nb_properties_per_streamline != int(nb_properties_per_streamline):
                msg = ("Nb. of properties differs from one streamline to"
                       " another!")
                raise DataError(msg)

            header[Field.NB_STREAMLINES] = nb_streamlines
            header[Field.NB_SCALARS_PER_POINT] = nb_scalars_per_point
            header[Field.NB_PROPERTIES_PER_STREAMLINE] = nb_properties_per_streamline

            # Overwrite header with updated one.
            f.seek(beginning, os.SEEK_SET)
            f.write(header.tostring())

    @staticmethod
    def _read_header(fileobj):
        """ Reads a TRK header from a file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        header : dict
            Metadata associated with this tractogram file.
        """
        # Record start position if this is a file-like object
        start_position = fileobj.tell() if hasattr(fileobj, 'tell') else None

        with Opener(fileobj) as f:

            # Read the header in one block.
            header_str = f.read(header_2_dtype.itemsize)
            header_rec = np.fromstring(string=header_str, dtype=header_2_dtype)

            # Check endianness
            endianness = native_code
            if header_rec['hdr_size'] != TrkFile.HEADER_SIZE:
                endianness = swapped_code

                # Swap byte order
                header_rec = header_rec.newbyteorder()
                if header_rec['hdr_size'] != TrkFile.HEADER_SIZE:
                    msg = "Invalid hdr_size: {0} instead of {1}"
                    raise HeaderError(msg.format(header_rec['hdr_size'],
                                                 TrkFile.HEADER_SIZE))

            if header_rec['version'] == 1:
                header_rec = np.fromstring(string=header_str,
                                           dtype=header_1_dtype)
            elif header_rec['version'] == 2:
                pass  # Nothing more to do.
            else:
                raise HeaderError('NiBabel only supports versions 1 and 2 of '
                                  'the Trackvis file format')

            # Convert the first record of `header_rec` into a dictionnary
            header = dict(zip(header_rec.dtype.names, header_rec[0]))
            header[Field.ENDIANNESS] = endianness

            # If vox_to_ras[3][3] is 0, it means the matrix is not recorded.
            if header[Field.VOXEL_TO_RASMM][3][3] == 0:
                header[Field.VOXEL_TO_RASMM] = np.eye(4, dtype=np.float32)
                warnings.warn(("Field 'vox_to_ras' in the TRK's header was"
                               " not recorded. Will continue assuming it's"
                               " the identity."), HeaderWarning)

            # Check that the 'vox_to_ras' affine is valid, i.e. should be
            # able to determine the axis directions.
            axcodes = aff2axcodes(header[Field.VOXEL_TO_RASMM])
            if None in axcodes:
                msg = ("The 'vox_to_ras' affine is invalid! Could not"
                       " determine the axis directions from it.\n{0}"
                       ).format(header[Field.VOXEL_TO_RASMM])
                raise HeaderError(msg)

            # By default, the voxel order is LPS.
            # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
            if header[Field.VOXEL_ORDER] == b"":
                msg = ("Voxel order is not specified, will assume 'LPS' since"
                       "it is Trackvis software's default.")
                warnings.warn(msg, HeaderWarning)
                header[Field.VOXEL_ORDER] = b"LPS"

            # Keep the file position where the data begin.
            header['_offset_data'] = f.tell()

        # Set the file position where it was, if it was previously open
        if start_position is not None:
            fileobj.seek(start_position, os.SEEK_SET)

        return header

    @staticmethod
    def _read(fileobj, header):
        """ Return generator that reads TRK data from `fileobj` given `header`

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.
        header : dict
            Metadata associated with this tractogram file.

        Yields
        ------
        data : tuple of ndarrays
            Length 3 tuple of streamline data of form (points, scalars,
            properties), where:

            * points: ndarray of shape (n_pts, 3)
            * scalars: ndarray of shape (n_pts, nb_scalars_per_point)
            * properties: ndarray of shape (nb_properties_per_point,)
        """
        i4_dtype = np.dtype(header[Field.ENDIANNESS] + "i4")
        f4_dtype = np.dtype(header[Field.ENDIANNESS] + "f4")

        with Opener(fileobj) as f:
            start_position = f.tell()

            nb_pts_and_scalars = int(3 +
                                     header[Field.NB_SCALARS_PER_POINT])
            pts_and_scalars_size = int(nb_pts_and_scalars * f4_dtype.itemsize)
            nb_properties = header[Field.NB_PROPERTIES_PER_STREAMLINE]
            properties_size = int(nb_properties * f4_dtype.itemsize)

            # Set the file position at the beginning of the data.
            f.seek(header["_offset_data"], os.SEEK_SET)

            # If 'count' field is 0, i.e. not provided, we have to loop
            # until the EOF.
            nb_streamlines = header[Field.NB_STREAMLINES]
            if nb_streamlines == 0:
                nb_streamlines = np.inf

            count = 0
            nb_pts_dtype = i4_dtype.str[:-1]
            while count < nb_streamlines:
                nb_pts_str = f.read(i4_dtype.itemsize)

                # Check if we reached EOF
                if len(nb_pts_str) == 0:
                    break

                # Read number of points of the next streamline.
                nb_pts = struct.unpack(nb_pts_dtype, nb_pts_str)[0]

                # Read streamline's data
                points_and_scalars = np.ndarray(
                    shape=(nb_pts, nb_pts_and_scalars),
                    dtype=f4_dtype,
                    buffer=f.read(nb_pts * pts_and_scalars_size))

                points = points_and_scalars[:, :3]
                scalars = points_and_scalars[:, 3:]

                # Read properties
                properties = np.ndarray(
                    shape=(nb_properties,),
                    dtype=f4_dtype,
                    buffer=f.read(properties_size))

                yield points, scalars, properties
                count += 1

            # In case the 'count' field was not provided.
            header[Field.NB_STREAMLINES] = count

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)

    def __str__(self):
        """ Gets a formatted string of the header of a TRK file.

        Returns
        -------
        info : string
            Header information relevant to the TRK format.
        """
        vars = self.header.copy()
        for attr in dir(Field):
            if attr[0] in string.ascii_uppercase:
                hdr_field = getattr(Field, attr)
                if hdr_field in vars:
                    vars[attr] = vars[hdr_field]
        vars['scalar_names'] = '\n  '.join([asstr(s)
                                            for s in vars['scalar_name']
                                            if len(s) > 0])
        vars['property_names'] = "\n  ".join([asstr(s)
                                              for s in vars['property_name']
                                              if len(s) > 0])
        # Make all byte strings into strings
        # Fixes recursion error on Python 3.3
        vars = dict((k, asstr(v) if hasattr(v, 'decode') else v)
                    for k, v in vars.items())
        return """\
MAGIC NUMBER: {MAGIC_NUMBER}
v.{version}
dim: {DIMENSIONS}
voxel_sizes: {VOXEL_SIZES}
orgin: {ORIGIN}
nb_scalars: {NB_SCALARS_PER_POINT}
scalar_name:\n  {scalar_names}
nb_properties: {NB_PROPERTIES_PER_STREAMLINE}
property_name:\n  {property_names}
vox_to_world:\n{VOXEL_TO_RASMM}
voxel_order: {VOXEL_ORDER}
image_orientation_patient: {image_orientation_patient}
pad1: {pad1}
pad2: {pad2}
invert_x: {invert_x}
invert_y: {invert_y}
invert_z: {invert_z}
swap_xy: {swap_xy}
swap_yz: {swap_yz}
swap_zx: {swap_zx}
n_count: {NB_STREAMLINES}
hdr_size: {hdr_size}""".format(**vars)

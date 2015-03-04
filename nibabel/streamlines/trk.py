from __future__ import division

# Documentation available here:
# http://www.trackvis.org/docs/?subsect=fileformat

from ..externals.six.moves import xrange
import struct
import os
import warnings

import numpy as np

from nibabel.openers import Opener
from nibabel.volumeutils import (native_code, swapped_code)

from nibabel.streamlines.base_format import Streamlines, LazyStreamlines, StreamlinesFile
from nibabel.streamlines.header import Field
from nibabel.streamlines.base_format import DataError, HeaderError

# Definition of trackvis header structure.
# See http://www.trackvis.org/docs/?subsect=fileformat
# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_1_dtd = [(Field.MAGIC_NUMBER, 'S6'),
                (Field.DIMENSIONS, 'h', 3),
                (Field.VOXEL_SIZES, 'f4', 3),
                (Field.ORIGIN, 'f4', 3),
                (Field.NB_SCALARS_PER_POINT, 'h'),
                ('scalar_name', 'S20', 10),
                (Field.NB_PROPERTIES_PER_STREAMLINE, 'h'),
                ('property_name', 'S20', 10),
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
# coordinates (axes L->R, P->A, I->S).  IF (0 based) value [3, 3] from
# this matrix is 0, this means the matrix is not recorded.
header_2_dtd = [(Field.MAGIC_NUMBER, 'S6'),
                (Field.DIMENSIONS, 'h', 3),
                (Field.VOXEL_SIZES, 'f4', 3),
                (Field.ORIGIN, 'f4', 3),
                (Field.NB_SCALARS_PER_POINT, 'h'),
                ('scalar_name', 'S20', 10),
                (Field.NB_PROPERTIES_PER_STREAMLINE, 'h'),
                ('property_name', 'S20', 10),
                (Field.VOXEL_TO_WORLD, 'f4', (4, 4)),  # new field for version 2
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


class TrkFile(StreamlinesFile):
    ''' Convenience class to encapsulate TRK format. '''

    MAGIC_NUMBER = b"TRACK"
    HEADER_SIZE = 1000

    @classmethod
    def get_magic_number(cls):
        ''' Return TRK's magic number. '''
        return cls.MAGIC_NUMBER

    @classmethod
    def is_correct_format(cls, fileobj):
        ''' Check if the file is in TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data)

        Returns
        -------
        is_correct_format : boolean
            Returns True if `fileobj` is in TRK format.
        '''
        with Opener(fileobj) as f:
            magic_number = f.read(5)
            f.seek(-5, os.SEEK_CUR)
            return magic_number == cls.MAGIC_NUMBER

        return False

    @classmethod
    def sanity_check(cls, fileobj):
        ''' Check if data is consistent with information contained in the header.
        [Might be useful]

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data)

        Returns
        -------
        is_consistent : boolean
            Returns True if data is consistent with header, False otherwise.
        '''
        is_consistent = True

        with Opener(fileobj) as f:
            start_position = f.tell()

            # Read header
            hdr_str = f.read(header_2_dtype.itemsize)
            hdr_rec = np.fromstring(string=hdr_str, dtype=header_2_dtype)

            if hdr_rec['version'] == 1:
                hdr_rec = np.fromstring(string=hdr_str, dtype=header_1_dtype)
            elif hdr_rec['version'] == 2:
                pass  # Nothing more to do here
            else:
                warnings.warn("NiBabel only supports versions 1 and 2 (not v.{0}).".format(hdr_rec['version']))
                f.seek(start_position, os.SEEK_CUR)  # Set the file position where it was.
                return False

            # Convert the first record of `hdr_rec` into a dictionnary
            hdr = dict(zip(hdr_rec.dtype.names, hdr_rec[0]))

            # Check endianness
            hdr[Field.ENDIAN] = native_code
            if hdr['hdr_size'] != cls.HEADER_SIZE:
                hdr[Field.ENDIAN] = swapped_code
                hdr = dict(zip(hdr_rec.dtype.names, hdr_rec[0].newbyteorder()))  # Swap byte order
                if hdr['hdr_size'] != cls.HEADER_SIZE:
                    warnings.warn("Invalid hdr_size: {0} instead of {1}".format(hdr['hdr_size'], cls.HEADER_SIZE))
                    f.seek(start_position, os.SEEK_CUR)  # Set the file position where it was.
                    return False

            # By default, the voxel order is LPS.
            # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
            if hdr[Field.VOXEL_ORDER] == "":
                is_consistent = False
                warnings.warn("Voxel order is not specified, will assume 'LPS' since it is Trackvis software's default.")

            i4_dtype = np.dtype(hdr[Field.ENDIAN] + "i4")
            f4_dtype = np.dtype(hdr[Field.ENDIAN] + "f4")

            pts_and_scalars_size = (3 + hdr[Field.NB_SCALARS_PER_POINT]) * f4_dtype.itemsize
            properties_size = hdr[Field.NB_PROPERTIES_PER_STREAMLINE] * f4_dtype.itemsize

            #Verify the number of streamlines specified in the header is correct.
            nb_streamlines = 0
            while True:
                # Read number of points of the streamline
                buf = f.read(i4_dtype.itemsize)

                if len(buf) == 0:
                    break  # EOF

                nb_pts = struct.unpack(i4_dtype.str[:-1], buf)[0]

                bytes_to_skip = nb_pts * pts_and_scalars_size
                bytes_to_skip += properties_size

                # Seek to the next streamline in the file.
                f.seek(bytes_to_skip, os.SEEK_CUR)

                nb_streamlines += 1

            if hdr[Field.NB_STREAMLINES] != nb_streamlines:
                is_consistent = False
                warnings.warn(('The number of streamlines specified in header ({1}) does not match '
                              'the actual number of streamlines contained in this file ({1}). '
                               ).format(hdr[Field.NB_STREAMLINES], nb_streamlines))

            f.seek(start_position, os.SEEK_CUR)  # Set the file position where it was.

        return is_consistent

    @classmethod
    def load(cls, fileobj, hdr={}, lazy_load=False):
        ''' Loads streamlines from a file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header)

        hdr : dict (optional)

        lazy_load : boolean (optional)
            Load streamlines in a lazy manner i.e. they will not be kept
            in memory.

        Returns
        -------
        streamlines : Streamlines object
            Returns an object containing streamlines' data and header
            information. See 'nibabel.Streamlines'.
        '''
        with Opener(fileobj) as f:
            # Read header
            hdr_str = f.read(header_2_dtype.itemsize)
            hdr_rec = np.fromstring(string=hdr_str, dtype=header_2_dtype)

            if hdr_rec['version'] == 1:
                hdr_rec = np.fromstring(string=hdr_str, dtype=header_1_dtype)
            elif hdr_rec['version'] == 2:
                pass  # Nothing more to do
            else:
                raise HeaderError('NiBabel only supports versions 1 and 2.')

            # Convert the first record of `hdr_rec` into a dictionnary
            hdr.update(dict(zip(hdr_rec.dtype.names, hdr_rec[0])))

            # Check endianness
            hdr[Field.ENDIAN] = native_code
            if hdr['hdr_size'] != cls.HEADER_SIZE:
                hdr[Field.ENDIAN] = swapped_code
                hdr = dict(zip(hdr_rec.dtype.names, hdr_rec[0].newbyteorder()))  # Swap byte order
                if hdr['hdr_size'] != cls.HEADER_SIZE:
                    raise HeaderError('Invalid hdr_size: {0} instead of {1}'.format(hdr['hdr_size'], cls.HEADER_SIZE))

            # By default, the voxel order is LPS.
            # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
            if hdr[Field.VOXEL_ORDER] == "":
                hdr[Field.VOXEL_ORDER] = "LPS"

            # Keep the file position where the data begin.
            hdr['pos_data'] = f.tell()

            # If 'count' field is 0, i.e. not provided, we have to loop until the EOF.
            if hdr[Field.NB_STREAMLINES] == 0:
                del hdr[Field.NB_STREAMLINES]

        points      = lambda: (x[0] for x in TrkFile._read_data(hdr, fileobj))
        scalars     = lambda: (x[1] for x in TrkFile._read_data(hdr, fileobj))
        properties  = lambda: (x[2] for x in TrkFile._read_data(hdr, fileobj))
        data        = lambda: TrkFile._read_data(hdr, fileobj)

        if lazy_load:
            count = lambda: TrkFile._count(hdr, fileobj)
            if Field.NB_STREAMLINES in hdr:
                count = hdr[Field.NB_STREAMLINES]

            streamlines = LazyStreamlines(points, scalars, properties, data=data, count=count)
        else:
            streamlines = Streamlines(*zip(*data()))

        # Set available header's information
        streamlines.header.update(hdr)
        return streamlines

    @staticmethod
    def _read_data(hdr, fileobj):
        ''' Read streamlines' data from a file-like object using a TRK's header. '''
        i4_dtype = np.dtype(hdr[Field.ENDIAN] + "i4")
        f4_dtype = np.dtype(hdr[Field.ENDIAN] + "f4")

        with Opener(fileobj) as f:
            start_position = f.tell()

            nb_pts_and_scalars = 3 + int(hdr[Field.NB_SCALARS_PER_POINT])
            pts_and_scalars_size = nb_pts_and_scalars * f4_dtype.itemsize

            slice_pts_and_scalars = lambda data: (data, [])
            if hdr[Field.NB_SCALARS_PER_POINT] > 0:
                # This is faster than np.split
                slice_pts_and_scalars = lambda data: (data[:, :3], data[:, 3:])

            # Using np.fromfile would be faster, but does not support StringIO
            read_pts_and_scalars = lambda nb_pts: slice_pts_and_scalars(np.ndarray(shape=(nb_pts, nb_pts_and_scalars),
                                                                                   dtype=f4_dtype,
                                                                                   buffer=f.read(nb_pts * pts_and_scalars_size)))

            properties_size = int(hdr[Field.NB_PROPERTIES_PER_STREAMLINE]) * f4_dtype.itemsize
            read_properties = lambda: []
            if hdr[Field.NB_PROPERTIES_PER_STREAMLINE] > 0:
                read_properties = lambda: np.fromstring(f.read(properties_size),
                                                        dtype=f4_dtype,
                                                        count=hdr[Field.NB_PROPERTIES_PER_STREAMLINE])

            # Set the file position at the beginning of the data.
            f.seek(hdr['pos_data'], os.SEEK_SET)

            #for i in xrange(hdr[Field.NB_STREAMLINES]):
            nb_streamlines = hdr.get(Field.NB_STREAMLINES, np.inf)
            i = 0
            while i < nb_streamlines:
                nb_pts_str = f.read(i4_dtype.itemsize)

                # Check if we reached EOF
                if len(nb_pts_str) == 0:
                    break

                # Read number of points of the next streamline.
                nb_pts = struct.unpack(i4_dtype.str[:-1], nb_pts_str)[0]

                # Read streamline's data
                pts, scalars = read_pts_and_scalars(nb_pts)
                properties = read_properties()

                # TRK's streamlines are in 'voxelmm' space, we send them to voxel space.
                pts = pts / hdr[Field.VOXEL_SIZES]

                yield pts, scalars, properties
                i += 1

            # In case the 'count' field was not provided.
            hdr[Field.NB_STREAMLINES] = i

            # Set the file position where it was.
            f.seek(start_position, os.SEEK_CUR)

    @staticmethod
    def _count(hdr, fileobj):
        ''' Count streamlines from a file-like object using a TRK's header. '''
        nb_streamlines = 0

        with Opener(fileobj) as f:
            start_position = f.tell()

            i4_dtype = np.dtype(hdr[Field.ENDIAN] + "i4")
            f4_dtype = np.dtype(hdr[Field.ENDIAN] + "f4")

            pts_and_scalars_size = (3 + hdr[Field.NB_SCALARS_PER_POINT]) * f4_dtype.itemsize
            properties_size = hdr[Field.NB_PROPERTIES_PER_STREAMLINE] * f4_dtype.itemsize

            # Set the file position at the beginning of the data.
            f.seek(hdr['pos_data'], os.SEEK_SET)

            # Count the actual number of streamlines.
            while True:
                # Read number of points of the streamline
                buf = f.read(i4_dtype.itemsize)

                if len(buf) == 0:
                    break  # EOF

                nb_pts = struct.unpack(i4_dtype.str[:-1], buf)[0]
                bytes_to_skip = nb_pts * pts_and_scalars_size
                bytes_to_skip += properties_size

                # Seek to the next streamline in the file.
                f.seek(bytes_to_skip, os.SEEK_CUR)

                nb_streamlines += 1

            f.seek(start_position, os.SEEK_CUR)  # Set the file position where it was.

        return nb_streamlines

    @classmethod
    def create_empty_header(cls):
        ''' Return an empty TRK compliant header. '''
        hdr = np.zeros(1, dtype=header_2_dtype)

        #Default values
        hdr[Field.MAGIC_NUMBER] = cls.MAGIC_NUMBER
        hdr[Field.VOXEL_SIZES] = (1, 1, 1)
        hdr[Field.DIMENSIONS] = (1, 1, 1)
        hdr[Field.VOXEL_TO_WORLD] = np.eye(4)
        hdr['version'] = 2
        hdr['hdr_size'] = cls.HEADER_SIZE

        return hdr

    @classmethod
    def save(cls, streamlines, fileobj):
        ''' Saves streamlines to a file-like object.

        Parameters
        ----------
        streamlines : Streamlines object
            Object containing streamlines' data and header information.
            See 'nibabel.Streamlines'.

        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data)

        hdr : dict (optional)

        Notes
        -----
        Streamlines are assumed to be in voxel space.
        '''
        hdr = cls.create_empty_header()

        #Override hdr's fields by those contain in `streamlines`'s header
        for k, v in streamlines.header.items():
            if k in header_2_dtype.fields.keys():
                hdr[k] = v

        # Check which endianess to use to write data.
        endianess = streamlines.header.get(Field.ENDIAN, native_code)

        if endianess == swapped_code:
            hdr = hdr.newbyteorder()

        i4_dtype = np.dtype(endianess + "i4")
        f4_dtype = np.dtype(endianess + "f4")

        # Keep counts for correcting incoherent fields or warn.
        nb_streamlines  = 0
        nb_points       = 0
        nb_scalars      = 0
        nb_properties   = 0

        # Write header + data of streamlines
        with Opener(fileobj, mode="wb") as f:
            pos = f.tell()
            # Write header
            f.write(hdr[0].tostring())

            for points, scalars, properties in streamlines:
                if len(scalars) > 0 and len(scalars) != len(points):
                    raise DataError("Missing scalars for some points!")

                points = np.array(points, dtype=f4_dtype)
                scalars = np.array(scalars, dtype=f4_dtype).reshape((len(points), -1))
                properties = np.array(properties, dtype=f4_dtype)

                # TRK's streamlines need to be in 'voxelmm' space
                points = points * hdr[Field.VOXEL_SIZES]

                data = struct.pack(i4_dtype.str[:-1], len(points))
                data += np.concatenate((points, scalars), axis=1).tostring()
                data += properties.tostring()
                f.write(data)

                nb_streamlines  += 1
                nb_points       += len(points)
                nb_scalars      += scalars.size
                nb_properties   += len(properties)

            # Either correct or warn if header and data are incoherent.
            #TODO: add a warn option as a function parameter
            nb_scalars_per_point = nb_scalars / nb_points
            nb_properties_per_streamline = nb_properties / nb_streamlines

            # Check for errors
            if nb_scalars_per_point != int(nb_scalars_per_point):
                raise DataError("Nb. of scalars differs from one point to another!")

            if nb_properties_per_streamline != int(nb_properties_per_streamline):
                raise DataError("Nb. of properties differs from one streamline to another!")

            hdr[Field.NB_STREAMLINES] = nb_streamlines
            hdr[Field.NB_SCALARS_PER_POINT] = nb_scalars_per_point
            hdr[Field.NB_PROPERTIES_PER_STREAMLINE] = nb_properties_per_streamline

            f.seek(pos, os.SEEK_SET)
            f.write(hdr[0].tostring())  # Overwrite header with updated one.

    @staticmethod
    def pretty_print(streamlines):
        ''' Gets a formatted string contaning header's information
        relevant to the TRK format.

        Parameters
        ----------
        streamlines : Streamlines object
            Object containing streamlines' data and header information.
            See 'nibabel.Streamlines'.

        Returns
        -------
        info : string
            Header's information relevant to the TRK format.
        '''
        hdr = streamlines.header

        info = ""
        info += "MAGIC NUMBER: {0}".format(hdr[Field.MAGIC_NUMBER])
        info += "v.{0}".format(hdr['version'])
        info += "dim: {0}".format(hdr[Field.DIMENSIONS])
        info += "voxel_sizes: {0}".format(hdr[Field.VOXEL_SIZES])
        info += "orgin: {0}".format(hdr[Field.ORIGIN])
        info += "nb_scalars: {0}".format(hdr[Field.NB_SCALARS_PER_POINT])
        info += "scalar_name:\n {0}".format("\n".join(hdr['scalar_name']))
        info += "nb_properties: {0}".format(hdr[Field.NB_PROPERTIES_PER_STREAMLINE])
        info += "property_name:\n {0}".format("\n".join(hdr['property_name']))
        info += "vox_to_world: {0}".format(hdr[Field.VOXEL_TO_WORLD])
        #info += "world_order: {0}".format(hdr[Field.WORLD_ORDER])
        info += "voxel_order: {0}".format(hdr[Field.VOXEL_ORDER])
        info += "image_orientation_patient: {0}".format(hdr['image_orientation_patient'])
        info += "pad1: {0}".format(hdr['pad1'])
        info += "pad2: {0}".format(hdr['pad2'])
        info += "invert_x: {0}".format(hdr['invert_x'])
        info += "invert_y: {0}".format(hdr['invert_y'])
        info += "invert_z: {0}".format(hdr['invert_z'])
        info += "swap_xy: {0}".format(hdr['swap_xy'])
        info += "swap_yz: {0}".format(hdr['swap_yz'])
        info += "swap_zx: {0}".format(hdr['swap_zx'])
        info += "n_count: {0}".format(hdr[Field.NB_STREAMLINES])
        info += "hdr_size: {0}".format(hdr['hdr_size'])
        info += "endianess: {0}".format(hdr[Field.ENDIAN])

        return info

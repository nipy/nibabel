# Documentation available here:
# http://www.trackvis.org/docs/?subsect=fileformat
from pdb import set_trace as dbg

import os
import warnings
import numpy as np
from numpy.lib.recfunctions import append_fields

from nibabel.openers import Opener
from nibabel.volumeutils import (native_code, swapped_code, endian_codes)

from nibabel.streamlines.base_format import DynamicStreamlineFile
from nibabel.streamlines.header import Field

# Definition of trackvis header structure.
# See http://www.trackvis.org/docs/?subsect=fileformat
# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_1_dtd = [
    (Field.MAGIC_NUMBER, 'S6'),
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
header_2_dtd = [
    (Field.MAGIC_NUMBER, 'S6'),
    (Field.DIMENSIONS, 'h', 3),
    (Field.VOXEL_SIZES, 'f4', 3),
    (Field.ORIGIN, 'f4', 3),
    (Field.NB_SCALARS_PER_POINT, 'h'),
    ('scalar_name', 'S20', 10),
    (Field.NB_PROPERTIES_PER_STREAMLINE, 'h'),
    ('property_name', 'S20', 10),
    (Field.VOXEL_TO_WORLD, 'f4', (4,4)), # new field for version 2
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


class HeaderError(Exception):
    pass


class DataError(Exception):
    pass


class TrkFile(DynamicStreamlineFile):
    MAGIC_NUMBER = "TRACK"
    OFFSET = 1000

    def __init__(self, hdr, streamlines, scalars, properties):
        self.filename = None
        
        self.hdr = hdr
        self.streamlines = streamlines
        self.scalars = scalars
        self.properties = properties

    #####
    # Static Methods
    ###
    @classmethod
    def get_magic_number(cls):
        ''' Return TRK's magic number '''
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
           Returns True if `fileobj` is in TRK format, False otherwise.
        '''
        with Opener(fileobj) as fileobj:
            magic_number = fileobj.read(5)
            fileobj.seek(-5, os.SEEK_CUR)
            return magic_number == cls.MAGIC_NUMBER

        return False

    @classmethod
    def load(cls, fileobj):
        hdr = {}
        pos_header = 0
        pos_data = 0

        with Opener(fileobj) as fileobj:
            pos_header = fileobj.tell()

            #####
            # Read header
            ###
            hdr_str = fileobj.read(header_2_dtype.itemsize)
            hdr = np.fromstring(string=hdr_str, dtype=header_2_dtype)

            if hdr['version'] == 1:
                hdr = np.fromstring(string=hdr_str, dtype=header_1_dtype)
            elif hdr['version'] == 2:
                pass # Nothing more to do here
            else:
                raise HeaderError('NiBabel only supports versions 1 and 2.')

            # Make header a dictionnary instead of ndarray
            hdr = dict(zip(hdr.dtype.names, hdr[0]))

            # Check endianness
            #hdr = append_fields(hdr, Field.ENDIAN, [native_code], usemask=False)
            hdr[Field.ENDIAN] = native_code
            if hdr['hdr_size'] != 1000:
                hdr[Field.ENDIAN] = swapped_code
                hdr = hdr.newbyteorder()
                if hdr['hdr_size'] != 1000:
                    raise HeaderError('Invalid hdr_size of {0}'.format(hdr['hdr_size']))
            
            # Add more header fields implied by trk format.
            #hdr = append_fields(hdr, Field.WORLD_ORDER, ["RAS"], usemask=False)
            hdr[Field.WORLD_ORDER] = "RAS"

            pos_data = fileobj.tell()

            i4_dtype = np.dtype(hdr[Field.ENDIAN] + "i4")
            f4_dtype = np.dtype(hdr[Field.ENDIAN] + "f4")

            nb_streamlines = 0

            #Either verify the number of streamlines specified in the header is correct or
            # count the actual number of streamlines in case it was not specified in the header.
            while True:
                # Read number of points of the streamline
                buf = fileobj.read(i4_dtype.itemsize)

                if buf == '':
                    break # EOF

                nb_pts = np.fromstring(buf,
                                       dtype=i4_dtype,
                                       count=1)

                bytes_to_skip = nb_pts * 3  # x, y, z coordinates
                bytes_to_skip += nb_pts * hdr[Field.NB_SCALARS_PER_POINT]
                bytes_to_skip += hdr[Field.NB_PROPERTIES_PER_STREAMLINE]

                # Seek to the next streamline in the file.
                fileobj.seek(bytes_to_skip * f4_dtype.itemsize, os.SEEK_CUR)

                nb_streamlines += 1

            if hdr[Field.NB_STREAMLINES] != nb_streamlines:
                warnings.warn('The number of streamlines specified in header ({1}) does not match ' +
                             'the actual number of streamlines contained in this file ({1}). ' +
                             'The latter will be used.'.format(hdr[Field.NB_STREAMLINES], nb_streamlines))

            hdr[Field.NB_STREAMLINES] = nb_streamlines

        trk_file = cls(hdr, [], [], [])
        trk_file.pos_header = pos_header
        trk_file.pos_data = pos_data
        trk_file.streamlines

        return trk_file
        # cls(hdr, streamlines, scalars, properties)

    def get_header(self):
        return self.hdr

    def get_points(self, as_generator=False):
        self.fileobj.seek(self.pos_data, os.SEEK_SET)
        pos = self.pos_data

        i4_dtype = np.dtype(self.hdr[Field.ENDIAN] + "i4")
        f4_dtype = np.dtype(self.hdr[Field.ENDIAN] + "f4")

        for i in range(self.hdr[Field.NB_STREAMLINES]):
            # Read number of points of the streamline
            nb_pts = np.fromstring(self.fileobj.read(i4_dtype.itemsize),
                                   dtype=i4_dtype,
                                   count=1)
            
            # Read points of the streamline
            pts = np.fromstring(self.fileobj.read(nb_pts * 3 * i4_dtype.itemsize),
                                dtype=[f4_dtype, f4_dtype, f4_dtype],
                                count=nb_pts)

            pos = self.fileobj.tell()
            yield pts
            self.fileobj.seek(pos, os.SEEK_SET)

            bytes_to_skip = nb_pts * self.hdr[Field.NB_SCALARS_PER_POINT]
            bytes_to_skip += self.hdr[Field.NB_PROPERTIES_PER_STREAMLINE]

            # Seek to the next streamline in the file.
            self.fileobj.seek(bytes_to_skip * f4_dtype.itemsize, os.SEEK_CUR)

    #####
    # Methods
    ###




# import os
# import logging
# import numpy as np

# from tractconverter.formats.header import Header as H


# def readBinaryBytes(f, nbBytes, dtype):
#     buff = f.read(nbBytes * dtype.itemsize)
#     return np.frombuffer(buff, dtype=dtype)


# class TRK:
#     # self.hdr
#     # self.filename
#     # self.hdr[H.ENDIAN]
#     # self.FIBER_DELIMITER
#     # self.END_DELIMITER

#     @staticmethod
#     def create(filename, hdr, anatFile=None):
#         f = open(filename, 'wb')
#         f.write(TRK.MAGIC_NUMBER + "\n")
#         f.close()

#         trk = TRK(filename, load=False)
#         trk.hdr = hdr
#         trk.writeHeader()

#         return trk

#     #####
#     # Methods
#     ###
#     def __init__(self, filename, anatFile=None, load=True):
#         if not TRK._check(filename):
#             raise NameError("Not a TRK file.")

#         self.filename = filename
#         self.hdr = {}
#         if load:
#             self._load()

#     def _load(self):
#         f = open(self.filename, 'rb')

#         #####
#         # Read header
#         ###
#         self.hdr[H.MAGIC_NUMBER] = f.read(6)
#         self.hdr[H.DIMENSIONS] = np.frombuffer(f.read(6), dtype='<i2')
#         self.hdr[H.VOXEL_SIZES] = np.frombuffer(f.read(12), dtype='<f4')
#         self.hdr[H.ORIGIN] = np.frombuffer(f.read(12), dtype='<f4')
#         self.hdr[H.NB_SCALARS_PER_POINT] = np.frombuffer(f.read(2), dtype='<i2')[0]
#         self.hdr['scalar_name'] = [f.read(20) for i in range(10)]
#         self.hdr[H.NB_PROPERTIES_PER_STREAMLINE] = np.frombuffer(f.read(2), dtype='<i2')[0]
#         self.hdr['property_name'] = [f.read(20) for i in range(10)]

#         self.hdr[H.VOXEL_TO_WORLD] = np.frombuffer(f.read(64), dtype='<f4').reshape(4, 4)
#         self.hdr[H.WORLD_ORDER] = "RAS"

#         # Skip reserved bytes
#         f.seek(444, os.SEEK_CUR)

#         self.hdr[H.VOXEL_ORDER] = f.read(4)
#         self.hdr["pad2"] = f.read(4)
#         self.hdr["image_orientation_patient"] = np.frombuffer(f.read(24), dtype='<f4')
#         self.hdr["pad1"] = f.read(2)

#         self.hdr["invert_x"] = f.read(1) == '\x01'
#         self.hdr["invert_y"] = f.read(1) == '\x01'
#         self.hdr["invert_z"] = f.read(1) == '\x01'
#         self.hdr["swap_xy"] = f.read(1) == '\x01'
#         self.hdr["swap_yz"] = f.read(1) == '\x01'
#         self.hdr["swap_zx"] = f.read(1) == '\x01'

#         self.hdr[H.NB_FIBERS] = np.frombuffer(f.read(4), dtype='<i4')
#         self.hdr["version"] = np.frombuffer(f.read(4), dtype='<i4')
#         self.hdr["hdr_size"] = np.frombuffer(f.read(4), dtype='<i4')

#         # Check if little or big endian
#         self.hdr[H.ENDIAN] = '<'
#         if self.hdr["hdr_size"] != self.OFFSET:
#             self.hdr[H.ENDIAN] = '>'
#             self.hdr[H.NB_FIBERS] = self.hdr[H.NB_FIBERS].astype('>i4')
#             self.hdr["version"] = self.hdr["version"].astype('>i4')
#             self.hdr["hdr_size"] = self.hdr["hdr_size"].astype('>i4')

#         nb_fibers = 0
#         self.hdr[H.NB_POINTS] = 0

#         #Either verify the number of streamlines specified in the header is correct or
#         # count the actual number of streamlines in case it is not specified in the header.
#         remainingBytes = os.path.getsize(self.filename) - self.OFFSET
#         while remainingBytes > 0:
#             # Read points
#             nbPoints = readBinaryBytes(f, 1, np.dtype(self.hdr[H.ENDIAN] + "i4"))[0]
#             self.hdr[H.NB_POINTS] += nbPoints
#             # This seek is used to go to the next points number indication in the file.
#             f.seek((nbPoints * (3 + self.hdr[H.NB_SCALARS_PER_POINT])
#                    + self.hdr[H.NB_PROPERTIES_PER_STREAMLINE]) * 4, 1)  # Relative seek
#             remainingBytes -= (nbPoints * (3 + self.hdr[H.NB_SCALARS_PER_POINT])
#                                + self.hdr[H.NB_PROPERTIES_PER_STREAMLINE]) * 4 + 4
#             nb_fibers += 1

#         if self.hdr[H.NB_FIBERS] != nb_fibers:
#             logging.warn('The number of streamlines specified in header ({1}) does not match ' +
#                          'the actual number of streamlines contained in this file ({1}). ' +
#                          'The latter will be used.'.format(self.hdr[H.NB_FIBERS], nb_fibers))

#         self.hdr[H.NB_FIBERS] = nb_fibers

#         f.close()

#     def writeHeader(self):
#         # Get the voxel size and format it as an array.
#         voxel_sizes = np.asarray(self.hdr.get(H.VOXEL_SIZES, (1.0, 1.0, 1.0)), dtype='<f4')
#         dimensions = np.asarray(self.hdr.get(H.DIMENSIONS, (0, 0, 0)), dtype='<i2')

#         f = open(self.filename, 'wb')
#         f.write(self.MAGIC_NUMBER + "\0")  # id_string
#         f.write(dimensions)  # dim
#         f.write(voxel_sizes)  # voxel_size
#         f.write(np.zeros(12, dtype='i1'))  # origin
#         f.write(np.zeros(2, dtype='i1'))  # n_scalars
#         f.write(np.zeros(200, dtype='i1'))  # scalar_name
#         f.write(np.zeros(2, dtype='i1'))  # n_properties
#         f.write(np.zeros(200, dtype='i1'))  # property_name
#         f.write(np.eye(4, dtype='<f4'))  # vos_to_ras
#         f.write(np.zeros(444, dtype='i1'))  # reserved
#         f.write(np.zeros(4, dtype='i1'))  # voxel_order
#         f.write(np.zeros(4, dtype='i1'))  # pad2
#         f.write(np.zeros(24, dtype='i1'))  # image_orientation_patient
#         f.write(np.zeros(2, dtype='i1'))  # pad1
#         f.write(np.zeros(1, dtype='i1'))  # invert_x
#         f.write(np.zeros(1, dtype='i1'))  # invert_y
#         f.write(np.zeros(1, dtype='i1'))  # invert_z
#         f.write(np.zeros(1, dtype='i1'))  # swap_xy
#         f.write(np.zeros(1, dtype='i1'))  # swap_yz
#         f.write(np.zeros(1, dtype='i1'))  # swap_zx
#         f.write(np.array(self.hdr[H.NB_FIBERS], dtype='<i4'))
#         f.write(np.array([2], dtype='<i4'))  # version
#         f.write(np.array(self.OFFSET, dtype='<i4'))  # hdr_size, should be 1000
#         f.close()

#     def close(self):
#         pass

#     def __iadd__(self, fibers):
#         f = open(self.filename, 'ab')
#         for fib in fibers:
#             f.write(np.array([len(fib)], '<i4').tostring())
#             f.write(fib.astype("<f4").tostring())
#         f.close()

#         return self

#     #####
#     # Iterate through fibers
#     ###
#     def __iter__(self):
#         f = open(self.filename, 'rb')
#         f.seek(self.OFFSET)

#         remainingBytes = os.path.getsize(self.filename) - self.OFFSET

#         cpt = 0
#         while cpt < self.hdr[H.NB_FIBERS] or remainingBytes > 0:
#             # Read points
#             nbPoints = readBinaryBytes(f, 1, np.dtype(self.hdr[H.ENDIAN] + "i4"))[0]
#             ptsAndScalars = readBinaryBytes(f,
#                                             nbPoints * (3 + self.hdr[H.NB_SCALARS_PER_POINT]),
#                                             np.dtype(self.hdr[H.ENDIAN] + "f4"))

#             newShape = [-1, 3 + self.hdr[H.NB_SCALARS_PER_POINT]]
#             ptsAndScalars = ptsAndScalars.reshape(newShape)

#             pointsWithoutScalars = ptsAndScalars[:, 0:3]
#             yield pointsWithoutScalars

#             # For now, we do not process the tract properties, so just skip over them.
#             remainingBytes -= nbPoints * (3 + self.hdr[H.NB_SCALARS_PER_POINT]) * 4 + 4
#             remainingBytes -= self.hdr[H.NB_PROPERTIES_PER_STREAMLINE] * 4
#             cpt += 1

#         f.close()

#     def __str__(self):
#         text = ""
#         text += "MAGIC NUMBER: {0}".format(self.hdr[H.MAGIC_NUMBER])
#         text += "v.{0}".format(self.hdr['version'])
#         text += "dim: {0}".format(self.hdr[H.DIMENSIONS])
#         text += "voxel_sizes: {0}".format(self.hdr[H.VOXEL_SIZES])
#         text += "orgin: {0}".format(self.hdr[H.ORIGIN])
#         text += "nb_scalars: {0}".format(self.hdr[H.NB_SCALARS_PER_POINT])
#         text += "scalar_name:\n {0}".format("\n".join(self.hdr['scalar_name']))
#         text += "nb_properties: {0}".format(self.hdr[H.NB_PROPERTIES_PER_STREAMLINE])
#         text += "property_name:\n {0}".format("\n".join(self.hdr['property_name']))
#         text += "vox_to_world: {0}".format(self.hdr[H.VOXEL_TO_WORLD])
#         text += "world_order: {0}".format(self.hdr[H.WORLD_ORDER])
#         text += "voxel_order: {0}".format(self.hdr[H.VOXEL_ORDER])
#         text += "image_orientation_patient: {0}".format(self.hdr['image_orientation_patient'])
#         text += "pad1: {0}".format(self.hdr['pad1'])
#         text += "pad2: {0}".format(self.hdr['pad2'])
#         text += "invert_x: {0}".format(self.hdr['invert_x'])
#         text += "invert_y: {0}".format(self.hdr['invert_y'])
#         text += "invert_z: {0}".format(self.hdr['invert_z'])
#         text += "swap_xy: {0}".format(self.hdr['swap_xy'])
#         text += "swap_yz: {0}".format(self.hdr['swap_yz'])
#         text += "swap_zx: {0}".format(self.hdr['swap_zx'])
#         text += "n_count: {0}".format(self.hdr[H.NB_FIBERS])
#         text += "hdr_size: {0}".format(self.hdr['hdr_size'])
#         text += "endianess: {0}".format(self.hdr[H.ENDIAN])

#         return text

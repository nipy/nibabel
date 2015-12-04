from __future__ import division

# Documentation available here:
# http://www.trackvis.org/docs/?subsect=fileformat

import os
import struct
import warnings
import itertools

import numpy as np
import nibabel as nib

from nibabel.affines import apply_affine
from nibabel.openers import Opener
from nibabel.py3k import asbytes, asstr
from nibabel.volumeutils import (native_code, swapped_code)

from .compact_list import CompactList
from .tractogram_file import TractogramFile
from .tractogram_file import DataError, HeaderError, HeaderWarning
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field


MAX_NB_NAMED_SCALARS_PER_POINT = 10
MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE = 10

# Definition of trackvis header structure.
# See http://www.trackvis.org/docs/?subsect=fileformat
# See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
header_1_dtd = [(Field.MAGIC_NUMBER, 'S6'),
                (Field.DIMENSIONS, 'h', 3),
                (Field.VOXEL_SIZES, 'f4', 3),
                (Field.ORIGIN, 'f4', 3),
                (Field.NB_SCALARS_PER_POINT, 'h'),
                ('scalar_name', 'S20', MAX_NB_NAMED_SCALARS_PER_POINT),
                (Field.NB_PROPERTIES_PER_STREAMLINE, 'h'),
                ('property_name', 'S20', MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE),
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
                ('property_name', 'S20', MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE),
                (Field.VOXEL_TO_RASMM, 'f4', (4, 4)),  # new field for version 2
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


class TrkReader(object):
    ''' Convenience class to encapsulate TRK file format.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object
        pointing to TRK file (and ready to read from the beginning
        of the TRK header)

    Note
    ----
    TrackVis (so its file format: TRK) considers the streamline coordinate
    (0,0,0) to be in the corner of the voxel whereas NiBabel's streamlines
    internal representation (Voxel space) assume (0,0,0) to be in the
    center of the voxel.

    Thus, streamlines are shifted of half a voxel on load and are shifted
    back on save.
    '''
    def __init__(self, fileobj):
        self.fileobj = fileobj

        with Opener(self.fileobj) as f:
            # Read header
            header_str = f.read(header_2_dtype.itemsize)
            header_rec = np.fromstring(string=header_str, dtype=header_2_dtype)

            if header_rec['version'] == 1:
                header_rec = np.fromstring(string=header_str, dtype=header_1_dtype)
            elif header_rec['version'] == 2:
                pass  # Nothing more to do
            else:
                raise HeaderError('NiBabel only supports versions 1 and 2.')

            # Convert the first record of `header_rec` into a dictionnary
            self.header = dict(zip(header_rec.dtype.names, header_rec[0]))

            # Check endianness
            self.endianness = native_code
            if self.header['hdr_size'] != TrkFile.HEADER_SIZE:
                self.endianness = swapped_code

                # Swap byte order
                self.header = dict(zip(header_rec.dtype.names, header_rec[0].newbyteorder()))
                if self.header['hdr_size'] != TrkFile.HEADER_SIZE:
                    raise HeaderError('Invalid hdr_size: {0} instead of {1}'.format(self.header['hdr_size'], TrkFile.HEADER_SIZE))

            # By default, the voxel order is LPS.
            # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
            if self.header[Field.VOXEL_ORDER] == b"":
                warnings.warn(("Voxel order is not specified, will assume"
                               " 'LPS' since it is Trackvis software's"
                               " default."), HeaderWarning)
                self.header[Field.VOXEL_ORDER] = b"LPS"

            # Keep the file position where the data begin.
            self.offset_data = f.tell()

    def __iter__(self):
        i4_dtype = np.dtype(self.endianness + "i4")
        f4_dtype = np.dtype(self.endianness + "f4")

        with Opener(self.fileobj) as f:
            start_position = f.tell()

            nb_pts_and_scalars = int(3 + self.header[Field.NB_SCALARS_PER_POINT])
            pts_and_scalars_size = int(nb_pts_and_scalars * f4_dtype.itemsize)
            properties_size = int(self.header[Field.NB_PROPERTIES_PER_STREAMLINE] * f4_dtype.itemsize)

            # Set the file position at the beginning of the data.
            f.seek(self.offset_data, os.SEEK_SET)

            # If 'count' field is 0, i.e. not provided, we have to loop until the EOF.
            nb_streamlines = self.header[Field.NB_STREAMLINES]
            if nb_streamlines == 0:
                nb_streamlines = np.inf

            i = 0
            nb_pts_dtype = i4_dtype.str[:-1]
            while i < nb_streamlines:
                nb_pts_str = f.read(i4_dtype.itemsize)

                # Check if we reached EOF
                if len(nb_pts_str) == 0:
                    break

                # Read number of points of the next streamline.
                nb_pts = struct.unpack(nb_pts_dtype, nb_pts_str)[0]

                # Read streamline's data
                points_and_scalars = np.ndarray(shape=(nb_pts, nb_pts_and_scalars),
                                                dtype=f4_dtype,
                                                buffer=f.read(nb_pts * pts_and_scalars_size))

                points = points_and_scalars[:, :3]
                scalars = points_and_scalars[:, 3:]

                # Read properties
                properties = np.ndarray(shape=(self.header[Field.NB_PROPERTIES_PER_STREAMLINE],),
                                        dtype=f4_dtype,
                                        buffer=f.read(properties_size))

                yield points, scalars, properties
                i += 1

            # In case the 'count' field was not provided.
            self.header[Field.NB_STREAMLINES] = i

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)


class TrkWriter(object):
    @classmethod
    def create_empty_header(cls):
        ''' Return an empty compliant TRK header. '''
        header = np.zeros(1, dtype=header_2_dtype)

        #Default values
        header[Field.MAGIC_NUMBER] = TrkFile.MAGIC_NUMBER
        header[Field.VOXEL_SIZES] = (1, 1, 1)
        header[Field.DIMENSIONS] = (1, 1, 1)
        header[Field.VOXEL_TO_RASMM] = np.eye(4)
        header[Field.VOXEL_ORDER] = b"RAS"
        header['version'] = 2
        header['hdr_size'] = TrkFile.HEADER_SIZE

        return header

    def __init__(self, fileobj, header):
        self.header = self.create_empty_header()

        # Override hdr's fields by those contained in `header`.
        for k, v in header.items():
            if k in header_2_dtype.fields.keys():
                self.header[k] = v

        # By default, the voxel order is LPS.
        # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
        if self.header[Field.VOXEL_ORDER] == b"":
            self.header[Field.VOXEL_ORDER] = b"LPS"

        # Keep counts for correcting incoherent fields or warn.
        self.nb_streamlines = 0
        self.nb_points = 0
        self.nb_scalars = 0
        self.nb_properties = 0

        # Write header
        self.header = self.header[0]
        self.file = Opener(fileobj, mode="wb")
        # Keep track of the beginning of the header.
        self.beginning = self.file.tell()
        self.file.write(self.header.tostring())

    def write(self, tractogram):
        i4_dtype = np.dtype("i4")
        f4_dtype = np.dtype("f4")

        try:
            first_item = next(iter(tractogram))
        except StopIteration:
            # Empty tractogram
            self.header[Field.NB_STREAMLINES] = 0
            self.header[Field.NB_SCALARS_PER_POINT] = 0
            self.header[Field.NB_PROPERTIES_PER_STREAMLINE] = 0
            # Overwrite header with updated one.
            self.file.seek(self.beginning, os.SEEK_SET)
            self.file.write(self.header.tostring())
            return

        # Update the 'property_name' field using 'data_per_streamline' of the tractogram.
        data_for_streamline = first_item.data_for_streamline
        if len(data_for_streamline) > MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE:
            raise ValueError("Can only store {0} named data_per_streamline (properties).".format(MAX_NB_NAMED_SCALARS_PER_POINT))

        data_for_streamline_keys = sorted(data_for_streamline.keys())
        self.header['property_name'] = np.zeros(MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE, dtype='S20')
        for i, k in enumerate(data_for_streamline_keys):
            nb_values = data_for_streamline[k].shape[0]

            if len(k) > 20:
                raise ValueError("Property name '{0}' is too long (max 20 char.)".format(k))
            elif len(k) > 18 and nb_values > 1:
                raise ValueError("Property name '{0}' is too long (need to be less than 18 characters when storing more than one value".format(k))

            property_name = k
            if nb_values > 1:
                # Use the last to bytes of the name to store the nb of values associated to this data_for_streamline.
                property_name = asbytes(k[:18].ljust(18, '\x00')) + b'\x00' + np.array(nb_values, dtype=np.int8).tostring()

            self.header['property_name'][i] = property_name

        # Update the 'scalar_name' field using 'data_per_point' of the tractogram.
        data_for_points = first_item.data_for_points
        if len(data_for_points) > MAX_NB_NAMED_SCALARS_PER_POINT:
            raise ValueError("Can only store {0} named data_per_point (scalars).".format(MAX_NB_NAMED_SCALARS_PER_POINT))

        data_for_points_keys = sorted(data_for_points.keys())
        self.header['scalar_name'] = np.zeros(MAX_NB_NAMED_SCALARS_PER_POINT, dtype='S20')
        for i, k in enumerate(data_for_points_keys):
            nb_values = data_for_points[k].shape[1]

            if len(k) > 20:
                raise ValueError("Scalar name '{0}' is too long (max 18 char.)".format(k))
            elif len(k) > 18 and nb_values > 1:
                raise ValueError("Scalar name '{0}' is too long (need to be less than 18 characters when storing more than one value".format(k))

            scalar_name = k
            if nb_values > 1:
                # Use the last to bytes of the name to store the nb of values associated to this data_for_streamline.
                scalar_name = asbytes(k[:18].ljust(18, '\x00')) + b'\x00' + np.array(nb_values, dtype=np.int8).tostring()

            self.header['scalar_name'][i] = scalar_name

        # `Tractogram` streamlines are in RAS+ and mm space, we will compute
        # the affine matrix that will bring them back to 'voxelmm' as required
        # by the TRK format.
        affine = np.eye(4)

        # Applied the inverse of the affine found in the TRK header.
        # rasmm -> voxel
        affine = np.dot(np.linalg.inv(self.header[Field.VOXEL_TO_RASMM]), affine)

        # If the voxel order implied by the affine does not match the voxel
        # order in the TRK header, change the orientation.
        # voxel (affine) -> voxel (header)
        header_ornt = asstr(self.header[Field.VOXEL_ORDER])
        affine_ornt = "".join(nib.orientations.aff2axcodes(self.header[Field.VOXEL_TO_RASMM]))
        header_ornt = nib.orientations.axcodes2ornt(header_ornt)
        affine_ornt = nib.orientations.axcodes2ornt(affine_ornt)
        ornt = nib.orientations.ornt_transform(affine_ornt, header_ornt)
        M = nib.orientations.inv_ornt_aff(ornt, self.header[Field.DIMENSIONS])
        affine = np.dot(M, affine)

        # TrackVis considers coordinate (0,0,0) to be the corner of the
        # voxel whereas `Tractogram` streamlines assume (0,0,0) is the
        # center of the voxel. Thus, streamlines are shifted of half a voxel.
        offset = np.eye(4)
        offset[:-1, -1] += 0.5
        affine = np.dot(offset, affine)

        # Finally send the streamlines in mm space.
        # voxel -> voxelmm
        scale = np.eye(4)
        scale[range(3), range(3)] *= self.header[Field.VOXEL_SIZES]
        affine = np.dot(scale, affine)

        # The TRK format uses float32 as the data type for points.
        affine = affine.astype(np.float32)

        for t in tractogram:
            if any((len(d) != len(t.streamline) for d in t.data_for_points.values())):
                raise DataError("Missing scalars for some points!")

            points = apply_affine(affine, np.asarray(t.streamline, dtype=f4_dtype))
            scalars = [np.asarray(t.data_for_points[k], dtype=f4_dtype) for k in data_for_points_keys]
            scalars = np.concatenate([np.ndarray((len(points), 0), dtype=f4_dtype)] + scalars, axis=1)
            properties = [np.asarray(t.data_for_streamline[k], dtype=f4_dtype) for k in data_for_streamline_keys]
            properties = np.concatenate([np.array([], dtype=f4_dtype)] + properties)

            data = struct.pack(i4_dtype.str[:-1], len(points))
            data += np.concatenate([points, scalars], axis=1).tostring()
            data += properties.tostring()
            self.file.write(data)

            self.nb_streamlines += 1
            self.nb_points += len(points)
            self.nb_scalars += scalars.size
            self.nb_properties += len(properties)

        # Use those values to update the header.
        nb_scalars_per_point = self.nb_scalars / self.nb_points
        nb_properties_per_streamline = self.nb_properties / self.nb_streamlines

        # Check for errors
        if nb_scalars_per_point != int(nb_scalars_per_point):
            raise DataError("Nb. of scalars differs from one point to another!")

        if nb_properties_per_streamline != int(nb_properties_per_streamline):
            raise DataError("Nb. of properties differs from one streamline to another!")

        self.header[Field.NB_STREAMLINES] = self.nb_streamlines
        self.header[Field.NB_SCALARS_PER_POINT] = nb_scalars_per_point
        self.header[Field.NB_PROPERTIES_PER_STREAMLINE] = nb_properties_per_streamline

        # Overwrite header with updated one.
        self.file.seek(self.beginning, os.SEEK_SET)
        self.file.write(self.header.tostring())


class TrkFile(TractogramFile):
    ''' Convenience class to encapsulate TRK file format.

    Note
    ----
    TrackVis (so its file format: TRK) considers the streamline coordinate
    (0,0,0) to be in the corner of the voxel whereas NiBabel's streamlines
    internal representation (Voxel space) assume (0,0,0) to be in the
    center of the voxel.

    Thus, streamlines are shifted of half a voxel on load and are shifted
    back on save.
    '''

    # Contants
    MAGIC_NUMBER = b"TRACK"
    HEADER_SIZE = 1000
    READ_BUFFER_SIZE = 10000000  # About 128 Mb if only no scalars nor properties.

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : ``Tractogram`` object
            Tractogram that will be contained in this ``TrkFile``.

        header : ``TractogramHeader`` file (optional)
            Metadata associated to this tractogram file.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm* space
        where coordinate (0,0,0) refers to the center of the voxel.
        """
        if header is None:
            header_rec = TrkWriter.create_empty_header()
            header = dict(zip(header_rec.dtype.names, header_rec))

        super(TrkFile, self).__init__(tractogram, header)

    @classmethod
    def get_magic_number(cls):
        ''' Return TRK's magic number. '''
        return cls.MAGIC_NUMBER

    @classmethod
    def support_data_per_point(cls):
        ''' Tells if this tractogram format supports saving data per point. '''
        return True

    @classmethod
    def support_data_per_streamline(cls):
        ''' Tells if this tractogram format supports saving data per streamline. '''
        return True

    @classmethod
    def is_correct_format(cls, fileobj):
        ''' Check if the file is in TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data).

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
    def _create_compactlist_from_generator(cls, gen):
        """ Creates a CompactList object from a generator yielding tuples of
            points, scalars and properties. """

        streamlines = CompactList()
        scalars = CompactList()
        properties = np.array([])

        gen = iter(gen)
        try:
            first_element = next(gen)
            gen = itertools.chain([first_element], gen)
        except StopIteration:
            return streamlines, scalars, properties

        # Allocated some buffer memory.
        pts = np.asarray(first_element[0])
        scals = np.asarray(first_element[1])
        props = np.asarray(first_element[2])

        scals_shape = scals.shape
        props_shape = props.shape

        streamlines._data = np.empty((cls.READ_BUFFER_SIZE, pts.shape[1]), dtype=pts.dtype)
        scalars._data = np.empty((cls.READ_BUFFER_SIZE, scals.shape[1]), dtype=scals.dtype)
        properties = np.empty((cls.READ_BUFFER_SIZE, props.shape[0]), dtype=props.dtype)

        offset = 0
        for i, (pts, scals, props) in enumerate(gen):
            pts = np.asarray(pts)
            scals = np.asarray(scals)
            props = np.asarray(props)

            if scals.shape[1] != scals_shape[1]:
                raise ValueError("Number of scalars differs from one"
                                 " point or streamline to another")

            if props.shape != props_shape:
                raise ValueError("Number of properties differs from one"
                                 " streamline to another")

            end = offset + len(pts)
            if end >= len(streamlines._data):
                # Resize is needed (at least `len(pts)` items will be added).
                streamlines._data.resize((len(streamlines._data) + len(pts)+cls.READ_BUFFER_SIZE, pts.shape[1]))
                scalars._data.resize((len(scalars._data) + len(scals)+cls.READ_BUFFER_SIZE, scals.shape[1]))

            streamlines._offsets.append(offset)
            streamlines._lengths.append(len(pts))
            streamlines._data[offset:offset+len(pts)] = pts
            scalars._data[offset:offset+len(scals)] = scals

            offset += len(pts)

            if i >= len(properties):
                properties.resize((len(properties) + cls.READ_BUFFER_SIZE, props.shape[0]))

            properties[i] = props

        # Clear unused memory.
        streamlines._data.resize((offset, pts.shape[1]))

        if scals_shape[1] == 0:
            # Because resizing an empty ndarray creates memory!
            scalars._data = np.empty((offset, scals.shape[1]))
        else:
            scalars._data.resize((offset, scals.shape[1]))

        # Share offsets and lengths between streamlines and scalars.
        scalars._offsets = streamlines._offsets
        scalars._lengths = streamlines._lengths

        if props_shape[0] == 0:
            # Because resizing an empty ndarray creates memory!
            properties = np.empty((i+1, props.shape[0]))
        else:
            properties.resize((i+1, props.shape[0]))

        return streamlines, scalars, properties

    @classmethod
    def load(cls, fileobj, lazy_load=False):
        ''' Loads streamlines from a file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header).

        lazy_load : boolean (optional)
            Load streamlines in a lazy manner i.e. they will not be kept
            in memory.

        Returns
        -------
        trk_file : ``TrkFile`` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the returned tractogram are assumed to be in RASmm
        space where coordinate (0,0,0) refers to the center of the voxel.
        '''
        trk_reader = TrkReader(fileobj)

        # TRK's streamlines are in 'voxelmm' space, we will compute the
        # affine matrix that will bring them back to RAS+ and mm space.
        affine = np.eye(4)

        # The affine matrix found in the TRK header requires the points to be
        # in the voxel space.
        # voxelmm -> voxel
        scale = np.eye(4)
        scale[range(3), range(3)] /= trk_reader.header[Field.VOXEL_SIZES]
        affine = np.dot(scale, affine)

        # TrackVis considers coordinate (0,0,0) to be the corner of the voxel
        # whereas streamlines returned assume (0,0,0) to be the center of the
        # voxel. Thus, streamlines are shifted of half a voxel.
        offset = np.eye(4)
        offset[:-1, -1] -= 0.5
        affine = np.dot(offset, affine)

        # If the voxel order implied by the affine does not match the voxel
        # order in the TRK header, change the orientation.
        # voxel (header) -> voxel (affine)
        header_ornt = asstr(trk_reader.header[Field.VOXEL_ORDER])
        affine_ornt = "".join(nib.orientations.aff2axcodes(trk_reader.header[Field.VOXEL_TO_RASMM]))
        header_ornt = nib.orientations.axcodes2ornt(header_ornt)
        affine_ornt = nib.orientations.axcodes2ornt(affine_ornt)
        ornt = nib.orientations.ornt_transform(header_ornt, affine_ornt)
        M = nib.orientations.inv_ornt_aff(ornt, trk_reader.header[Field.DIMENSIONS])
        affine = np.dot(M, affine)

        # Applied the affine found in the TRK header.
        # voxel -> rasmm
        affine = np.dot(trk_reader.header[Field.VOXEL_TO_RASMM], affine)

        # Find scalars and properties name
        data_per_point_slice = {}
        if trk_reader.header[Field.NB_SCALARS_PER_POINT] > 0:
            cpt = 0
            for scalar_name in trk_reader.header['scalar_name']:
                scalar_name = asstr(scalar_name)
                if len(scalar_name) == 0:
                    continue

                # Check if we encoded the number of values we stocked for this scalar name.
                nb_scalars = 1
                if scalar_name[-2] == '\x00' and scalar_name[-1] != '\x00':
                    nb_scalars = int(np.fromstring(scalar_name[-1], np.int8))

                scalar_name = scalar_name.split('\x00')[0]
                data_per_point_slice[scalar_name] = slice(cpt, cpt+nb_scalars)
                cpt += nb_scalars

            if cpt < trk_reader.header[Field.NB_SCALARS_PER_POINT]:
                data_per_point_slice['scalars'] = slice(cpt, trk_reader.header[Field.NB_SCALARS_PER_POINT])

        data_per_streamline_slice = {}
        if trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE] > 0:
            cpt = 0
            for property_name in trk_reader.header['property_name']:
                property_name = asstr(property_name)
                if len(property_name) == 0:
                    continue

                # Check if we encoded the number of values we stocked for this property name.
                nb_properties = 1
                if property_name[-2] == '\x00' and property_name[-1] != '\x00':
                    nb_properties = int(np.fromstring(property_name[-1], np.int8))

                property_name = property_name.split('\x00')[0]
                data_per_streamline_slice[property_name] = slice(cpt, cpt+nb_properties)
                cpt += nb_properties

            if cpt < trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE]:
                data_per_streamline_slice['properties'] = slice(cpt, trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE])


        if lazy_load:
            def _read():
                for pts, scals, props in trk_reader:
                    data_for_points = dict((k, scals[:, v]) for k, v in data_per_point_slice.items())
                    data_for_streamline = dict((k, props[v]) for k, v in data_per_streamline_slice.items())
                    yield TractogramItem(pts, data_for_streamline, data_for_points)

            tractogram = LazyTractogram.create_from(_read)

        else:
            streamlines, scalars, properties = cls._create_compactlist_from_generator(trk_reader)
            tractogram = Tractogram(streamlines)

            for scalar_name, slice_ in data_per_point_slice.items():
                clist = CompactList()
                clist._data = scalars._data[:, slice_]
                clist._offsets = scalars._offsets
                clist._lengths = scalars._lengths
                tractogram.data_per_point[scalar_name] = clist

            for property_name, slice_ in data_per_streamline_slice.items():
                tractogram.data_per_streamline[property_name] = properties[:, slice_]

        # Bring tractogram to RAS+ and mm space
        tractogram.apply_affine(affine.astype(np.float32))

        ## Perform some integrity checks
        #if tractogram.header.voxel_sizes != trk_reader.header[Field.VOXEL_SIZES]:
        #    raise HeaderError("'voxel_sizes' does not match the affine.")
        #if tractogram.header.nb_scalars_per_point != trk_reader.header[Field.NB_SCALARS_PER_POINT]:
        #    raise HeaderError("'nb_scalars_per_point' does not match.")
        #if tractogram.header.nb_properties_per_streamline != trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE]:
        #    raise HeaderError("'nb_properties_per_streamline' does not match.")

        return cls(tractogram, header=trk_reader.header)

    def save(self, fileobj):
        ''' Saves tractogram to a file-like object using TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data).
        '''
        trk_writer = TrkWriter(fileobj, self.header)
        trk_writer.write(self.tractogram)

    def __str__(self):
        ''' Gets a formatted string of the header of a TRK file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the header).

        Returns
        -------
        info : string
            Header information relevant to the TRK format.
        '''
        #trk_reader = TrkReader(fileobj)
        hdr = self.header

        info = ""
        info += "\nMAGIC NUMBER: {0}".format(hdr[Field.MAGIC_NUMBER])
        info += "\nv.{0}".format(hdr['version'])
        info += "\ndim: {0}".format(hdr[Field.DIMENSIONS])
        info += "\nvoxel_sizes: {0}".format(hdr[Field.VOXEL_SIZES])
        info += "\norgin: {0}".format(hdr[Field.ORIGIN])
        info += "\nnb_scalars: {0}".format(hdr[Field.NB_SCALARS_PER_POINT])
        info += "\nscalar_name:\n {0}".format("\n".join(map(asstr, hdr['scalar_name'])))
        info += "\nnb_properties: {0}".format(hdr[Field.NB_PROPERTIES_PER_STREAMLINE])
        info += "\nproperty_name:\n {0}".format("\n".join(map(asstr, hdr['property_name'])))
        info += "\nvox_to_world: {0}".format(hdr[Field.VOXEL_TO_RASMM])
        info += "\nvoxel_order: {0}".format(hdr[Field.VOXEL_ORDER])
        info += "\nimage_orientation_patient: {0}".format(hdr['image_orientation_patient'])
        info += "\npad1: {0}".format(hdr['pad1'])
        info += "\npad2: {0}".format(hdr['pad2'])
        info += "\ninvert_x: {0}".format(hdr['invert_x'])
        info += "\ninvert_y: {0}".format(hdr['invert_y'])
        info += "\ninvert_z: {0}".format(hdr['invert_z'])
        info += "\nswap_xy: {0}".format(hdr['swap_xy'])
        info += "\nswap_yz: {0}".format(hdr['swap_yz'])
        info += "\nswap_zx: {0}".format(hdr['swap_zx'])
        info += "\nn_count: {0}".format(hdr[Field.NB_STREAMLINES])
        info += "\nhdr_size: {0}".format(hdr['hdr_size'])
        #info += "endianess: {0}".format(hdr[Field.ENDIAN])

        return info

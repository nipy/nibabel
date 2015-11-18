from __future__ import division

# Documentation available here:
# http://www.trackvis.org/docs/?subsect=fileformat

import os
import struct
import warnings
import itertools

import numpy as np
import nibabel as nib

from nibabel.openers import Opener
from nibabel.volumeutils import (native_code, swapped_code)

from .compact_list import CompactList
from .tractogram_file import TractogramFile
from .base_format import DataError, HeaderError, HeaderWarning
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field

from .utils import get_affine_from_reference

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
        header['version'] = 2
        header['hdr_size'] = TrkFile.HEADER_SIZE

        return header[0]

    def __init__(self, fileobj, header):
        self.header = self.create_empty_header()

        # Override hdr's fields by those contain in `header`.
        for k, v in header.extra.items():
            if k in header_2_dtype.fields.keys():
                self.header[k] = v

        self.header[Field.NB_STREAMLINES] = 0
        if header.nb_streamlines is not None:
            self.header[Field.NB_STREAMLINES] = header.nb_streamlines

        self.header[Field.NB_SCALARS_PER_POINT] = header.nb_scalars_per_point
        self.header[Field.NB_PROPERTIES_PER_STREAMLINE] = header.nb_properties_per_streamline
        self.header[Field.VOXEL_SIZES] = header.voxel_sizes
        self.header[Field.VOXEL_TO_RASMM] = header.to_world_space
        self.header[Field.VOXEL_ORDER] = header.voxel_order

        # Keep counts for correcting incoherent fields or warn.
        self.nb_streamlines = 0
        self.nb_points = 0
        self.nb_scalars = 0
        self.nb_properties = 0

        # Write header
        self.file = Opener(fileobj, mode="wb")
        # Keep track of the beginning of the header.
        self.beginning = self.file.tell()
        self.file.write(self.header.tostring())

    def write(self, tractogram):
        i4_dtype = np.dtype("i4")
        f4_dtype = np.dtype("f4")

        # TRK's streamlines need to be in 'voxelmm' space and by definition
        # tractogram streamlines are in RAS+ and mm space.
        affine = np.linalg.inv(self.header[Field.VOXEL_TO_RASMM])
        affine[range(3), range(3)] *= self.header[Field.VOXEL_SIZES]

        # TrackVis considers coordinate (0,0,0) to be the corner of the
        # voxel whereas streamlines passed in parameters assume (0,0,0)
        # to be the center of the voxel. Thus, streamlines are shifted of
        # half a voxel.
        affine[:-1, -1] += np.array(self.header[Field.VOXEL_SIZES])/2.

        tractogram.apply_affine(affine)

        for t in tractogram:
            if any((len(d) != len(t.streamline) for d in t.data_for_points.values())):
                raise DataError("Missing scalars for some points!")

            points = np.asarray(t.streamline, dtype=f4_dtype)
            keys = sorted(t.data_for_points.keys())[:MAX_NB_NAMED_SCALARS_PER_POINT]
            scalars = np.asarray([t.data_for_points[k] for k in keys], dtype=f4_dtype).reshape((len(points), -1))
            keys = sorted(t.data_for_streamline.keys())[:MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE]
            properties = np.asarray([t.data_for_streamline[k] for k in keys], dtype=f4_dtype).flatten()

            data = struct.pack(i4_dtype.str[:-1], len(points))
            data += np.concatenate((points, scalars), axis=1).tostring()
            data += properties.tostring()
            self.file.write(data)

            self.nb_streamlines += 1
            self.nb_points += len(points)
            self.nb_scalars += scalars.size
            self.nb_properties += len(properties)

        # Either correct or warn if header and data are incoherent.
        #TODO: add a warn option as a function parameter
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


def create_compactlist_from_generator(gen):
    BUFFER_SIZE = 10000000  # About 128 Mb if item shape is 3.

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

    streamlines._data = np.empty((BUFFER_SIZE, pts.shape[1]), dtype=pts.dtype)
    scalars._data = np.empty((BUFFER_SIZE, scals.shape[1]), dtype=scals.dtype)
    properties = np.empty((BUFFER_SIZE, props.shape[0]), dtype=props.dtype)

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
            streamlines._data.resize((len(streamlines._data) + len(pts)+BUFFER_SIZE, pts.shape[1]))
            scalars._data.resize((len(scalars._data) + len(scals)+BUFFER_SIZE, scals.shape[1]))

        streamlines._offsets.append(offset)
        streamlines._lengths.append(len(pts))
        streamlines._data[offset:offset+len(pts)] = pts
        scalars._data[offset:offset+len(scals)] = scals

        offset += len(pts)

        if i >= len(properties):
            properties.resize((len(properties) + BUFFER_SIZE, props.shape[0]))

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

    def __init__(self, tractogram, header=None, ref=np.eye(4)):
        """
        Parameters
        ----------
        tractogram : ``Tractogram`` object
            Tractogram that will be contained in this ``TrkFile``.

        header : ``TractogramHeader`` file (optional)
            Metadata associated to this tractogram file.

        ref : filename | `Nifti1Image` object | 2D array (4,4) (optional)
            Reference space where streamlines live in.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm* space
        where coordinate (0,0,0) refers to the center of the voxel.
        """
        super(TrkFile, self).__init__(tractogram, header)
        self._affine = get_affine_from_reference(ref)

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

        # TRK's streamlines are in 'voxelmm' space, we send them to rasmm.
        # First send them to voxel space.
        affine = np.eye(4)
        affine[range(3), range(3)] /= trk_reader.header[Field.VOXEL_SIZES]

        # If voxel order implied from the affine does not match the voxel
        # order save in the TRK header, change the orientation.
        header_ornt = trk_reader.header[Field.VOXEL_ORDER]
        affine_ornt = "".join(nib.orientations.aff2axcodes(trk_reader.header[Field.VOXEL_TO_RASMM]))
        header_ornt = nib.orientations.axcodes2ornt(header_ornt)
        affine_ornt = nib.orientations.axcodes2ornt(affine_ornt)
        ornt = nib.orientations.ornt_transform(header_ornt, affine_ornt)
        M = nib.orientations.inv_ornt_aff(ornt, trk_reader.header[Field.DIMENSIONS])
        affine = np.dot(M, affine)

        # Applied the affine going from voxel space to rasmm.
        affine = np.dot(trk_reader.header[Field.VOXEL_TO_RASMM], affine)

        # TrackVis considers coordinate (0,0,0) to be the corner of the
        # voxel whereas streamlines returned assume (0,0,0) to be the
        # center of the voxel. Thus, streamlines are shifted of half
        #a voxel.
        affine[:-1, -1] -= np.array(trk_reader.header[Field.VOXEL_SIZES])/2.

        if lazy_load:
            # TODO when LazyTractogram has been refactored.
            def _apply_transform(trk_reader):
                for pts, scals, props in trk_reader:
                    # TRK's streamlines are in 'voxelmm' space, we send them to voxel space.
                    pts = pts / trk_reader.header[Field.VOXEL_SIZES]
                    # TrackVis considers coordinate (0,0,0) to be the corner of the
                    # voxel whereas streamlines returned assume (0,0,0) to be the
                    # center of the voxel. Thus, streamlines are shifted of half
                    #a voxel.
                    pts -= np.array(trk_reader.header[Field.VOXEL_SIZES])/2.
                    trk_reader
                    yield pts, scals, props

            def _read():
                for pts, scals, props in trk_reader:
                    # TODO
                    data_for_streamline = {}
                    data_for_points = {}
                    yield TractogramItem(pts, data_for_streamline, data_for_points)

            tractogram = LazyTractogram.create_from(_read)

        else:
            streamlines, scalars, properties = create_compactlist_from_generator(trk_reader)
            tractogram = Tractogram(streamlines)

            if trk_reader.header[Field.NB_SCALARS_PER_POINT] > 0:
                cpt = 0
                for scalar_name in trk_reader.header['scalar_name']:
                    if len(scalar_name) == 0:
                        continue

                    nb_scalars = np.fromstring(scalar_name[-1], np.int8)

                    clist = CompactList()
                    clist._data = scalars._data[:, cpt:cpt+nb_scalars]
                    clist._offsets = scalars._offsets
                    clist._lengths = scalars._lengths

                    scalar_name = scalar_name.split('\x00')[0]
                    tractogram.data_per_point[scalar_name] = clist
                    cpt += nb_scalars

                if cpt < trk_reader.header[Field.NB_SCALARS_PER_POINT]:
                    #tractogram.data_per_point['scalars'] = scalars
                    clist = CompactList()
                    clist._data = scalars._data[:, cpt:]
                    clist._offsets = scalars._offsets
                    clist._lengths = scalars._lengths
                    tractogram.data_per_point['scalars'] = clist

            if trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE] > 0:
                cpt = 0
                for property_name in trk_reader.header['property_name']:
                    if len(property_name) == 0:
                        continue

                    nb_properties = np.fromstring(property_name[-1], np.int8)
                    property_name = property_name.split('\x00')[0]
                    tractogram.data_per_streamline[property_name] = properties[:, cpt:cpt+nb_properties]
                    cpt += nb_properties

                if cpt < trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE]:
                    #tractogram.data_per_streamline['properties'] = properties
                    tractogram.data_per_streamline['properties'] = properties[:, cpt:]

        # Bring tractogram to RAS+ and mm space
        tractogram.apply_affine(affine)

        ## Perform some integrity checks
        #if tractogram.header.voxel_sizes != trk_reader.header[Field.VOXEL_SIZES]:
        #    raise HeaderError("'voxel_sizes' does not match the affine.")
        #if tractogram.header.nb_scalars_per_point != trk_reader.header[Field.NB_SCALARS_PER_POINT]:
        #    raise HeaderError("'nb_scalars_per_point' does not match.")
        #if tractogram.header.nb_properties_per_streamline != trk_reader.header[Field.NB_PROPERTIES_PER_STREAMLINE]:
        #    raise HeaderError("'nb_properties_per_streamline' does not match.")

        return cls(tractogram, header=trk_reader.header, ref=affine)

    def save(self, fileobj):
        ''' Saves tractogram to a file-like object using TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data).
        '''
        # Compute how many properties per streamline the tractogram has.
        self.header.nb_properties_per_streamline = 0
        self.header.extra['property_name'] = np.zeros(MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE, dtype='S20')
        data_for_streamline = self.tractogram[0].data_for_streamline
        for i, k in enumerate(sorted(data_for_streamline.keys())):
            if i >= MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE:
                warnings.warn(("Can only store {0} named properties: '{1}' will be omitted.".format(MAX_NB_NAMED_SCALARS_PER_POINT, k)), HeaderWarning)

            if len(k) > 19:
                warnings.warn(("Property name '{0}' has be truncated to {1}.".format(k, k[:19])), HeaderWarning)

            v = data_for_streamline[k]
            self.header.nb_properties_per_streamline += v.shape[0]

            property_name = k[:19].ljust(19, '\x00') + np.array(v.shape[0], dtype=np.int8).tostring()
            self.header.extra['property_name'][i] = property_name

        # Compute how many scalars per point the tractogram has.
        self.header.nb_scalars_per_point = 0
        self.header.extra['scalar_name'] = np.zeros(MAX_NB_NAMED_SCALARS_PER_POINT, dtype='S20')
        data_for_points = self.tractogram[0].data_for_points
        for i, k in enumerate(sorted(data_for_points.keys())):
            if i >= MAX_NB_NAMED_SCALARS_PER_POINT:
                warnings.warn(("Can only store {0} named scalars: '{1}' will be omitted.".format(MAX_NB_NAMED_SCALARS_PER_POINT, k)), HeaderWarning)

            if len(k) > 19:
                warnings.warn(("Scalar name '{0}' has be truncated to {1}.".format(k, k[:19])), HeaderWarning)

            v = data_for_points[k]
            self.header.nb_scalars_per_point += v.shape[1]

            scalar_name = k[:19].ljust(19, '\x00') + np.array(v.shape[1], dtype=np.int8).tostring()
            self.header.extra['scalar_name'][i] = scalar_name

        trk_writer = TrkWriter(fileobj, self.header)
        trk_writer.write(self.tractogram)

    @staticmethod
    def pretty_print(fileobj):
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
        trk_reader = TrkReader(fileobj)
        hdr = trk_reader.header

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
        info += "vox_to_world: {0}".format(hdr[Field.VOXEL_TO_RASMM])
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

from __future__ import division

# Documentation available here:
# https://github.com/MRtrix3/mrtrix3/wiki/MRtrix-Image-formats-(.mif-&-.mih)#tracks-file-format-tck

import os
import struct
import warnings
import itertools
from collections import OrderedDict

import numpy as np
import nibabel as nib

from nibabel.affines import apply_affine
from nibabel.openers import Opener
from nibabel.volumeutils import (native_code, swapped_code)

from .array_sequence import ArraySequence
from .tractogram_file import TractogramFile
from .tractogram_file import DataError, HeaderError, HeaderWarning
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field


BUFFER_SIZE = 1000000


class TckReader(object):
    ''' Convenience class to encapsulate TCK file format.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object
        pointing to TCK file (and ready to read from the beginning
        of the TCK header)

    Notes
    -----
    MRtrix (so its file format: TCK) considers the streamlines to be saved in
    world space. It uses the same convention as Nifti: RAS+ and mm space with
    the coordinate (0,0,0) being at the center of the voxel.
    '''
    def __init__(self, fileobj):
        self.fileobj = fileobj

        with Opener(self.fileobj) as f:
            # Skip magic number
            buffer = f.fobj.readline()

            #####
            # Read header
            ###
            buffer = f.fobj.readline()
            while not buffer.rstrip().endswith("END"):
                buffer += f.fobj.readline()

            # Build dictionary from header (not used)
            self.header = dict(item.split(': ') for item in buffer.rstrip().split('\n')[:-1])

            # Set datatype
            self.dtype = np.dtype('>f4')
            if self.header['datatype'].endswith('LE'):
                self.dtype = np.dtype('<f4')

            # Keep the file position where the data begin.
            self.offset_data = int(self.header['file'].split()[1])

    def __iter__(self):
        with Opener(self.fileobj) as f:
            start_position = f.tell()

            # Set the file position at the beginning of the data.
            f.seek(self.offset_data, os.SEEK_SET)

            eof = False
            buff = ""
            pts = []

            i = 0

            while not eof or not np.all(np.isinf(pts)):
                if not eof:
                    # Read BUFFER_SIZE triplets of coordinates (float32)
                    nb_bytes_to_read = BUFFER_SIZE * 3 * self.dtype.itemsize
                    bytes_read = f.read(nb_bytes_to_read)
                    buff += bytes_read
                    eof = len(bytes_read) == 0

                pts = np.frombuffer(buff, dtype=self.dtype)  # Convert binary to float

                if self.dtype != '<f4':
                    pts = pts.astype('<f4')

                pts = pts.reshape([-1, 3])
                idx_nan = np.arange(len(pts))[np.isnan(pts[:, 0])]

                if len(idx_nan) == 0:
                    continue

                nb_pts_total = 0
                idx_start = 0
                for idx_end in idx_nan:
                    nb_pts = len(pts[idx_start:idx_end, :])
                    nb_pts_total += nb_pts

                    if nb_pts > 0:
                        yield pts[idx_start:idx_end, :]
                        i += 1

                    idx_start = idx_end + 1

                # Remove pts plus the first triplet of NaN.
                nb_bytes_to_remove = (nb_pts_total + len(idx_nan)) * 3 * self.dtype.itemsize
                buff = buff[nb_bytes_to_remove:]

            # In case the 'count' field was not provided.
            self.header[Field.NB_STREAMLINES] = i

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)


class TckWriter(object):

    FIBER_DELIMITER = np.array([[np.nan, np.nan, np.nan]], '<f4')
    EOF_DELIMITER = np.array([[np.inf, np.inf, np.inf]], '<f4')

    @classmethod
    def create_empty_header(cls):
        ''' Return an empty compliant TCK header. '''
        header = OrderedDict()

        #Default values
        header[Field.MAGIC_NUMBER] = TckFile.MAGIC_NUMBER
        header[Field.NB_STREAMLINES] = 0
        header['datatype'] = "Float32LE"

        return header

    def __init__(self, fileobj, header):
        self.header = self.create_empty_header()

        # Override hdr's fields by those contained in `header`.
        for k, v in header.items():
            self.header[k] = v

        # Write header
        self.file = Opener(fileobj, mode="wb")

        # Keep track of the beginning of the header.
        self.beginning = self.file.tell()

        # Fields to exclude
        exclude = [Field.MAGIC_NUMBER, Field.NB_STREAMLINES, "datatype", "file"]

        # We always put the field count after the magic number (the line after).
        self.count_offset = len(self.header[Field.MAGIC_NUMBER])+1

        lines = []
        lines.append(self.header[Field.MAGIC_NUMBER])
        lines.append("count: {0:010}".format(self.header[Field.NB_STREAMLINES]))
        lines.append("datatype: Float32LE")  # We always use Float32LE to save TCK files.
        lines.extend(["{0}: {1}".format(k, v) for k, v in self.header.items() if k not in exclude])
        lines.append("file: . ")  # Manually add this last field.
        out = "\n".join(lines)
        self.file.write(out)

        # Compute offset to the beginning of the binary data
        offset = len(out) + 5  # +5 is for "\nEND\n" added just before the data.

        # Take in account the number of characters needed to write 'offset' in ASCII.
        self.offset = offset + len(str(offset))

        # Corner case: the new 'offset' needs one more character to write it in ASCII
        # e.g. offset = 98 (i.e. 2 char.), so offset += 2 = 100 (i.e. 3 char.)
        #      thus the final offset = 101.
        if len(str(self.offset)) != len(str(offset)):
            self.offset += 1  # +1, we need one more character for that new digit.

        self.file.write(str(self.offset) + "\n")
        self.file.write("END\n")
        self.file.write(self.EOF_DELIMITER.tostring())

    def write(self, tractogram):
        # Start writing before the EOF_DELIMITER.
        self.file.seek(-len(self.EOF_DELIMITER.tostring()), os.SEEK_END)

        for s in tractogram.streamlines:
            self.header[Field.NB_STREAMLINES] += 1

            # TODO: use a buffer instead of writing one streamline at once.
            self.file.write(np.r_[s.astype('<f4'), self.FIBER_DELIMITER].tostring())

        # Add the EOF_DELIMITER.
        self.file.write(self.EOF_DELIMITER.tostring())

        # Overwrite the streamlines count in the header.
        self.file.seek(self.count_offset, os.SEEK_SET)
        self.file.write("count: {0:010}\n".format(self.header[Field.NB_STREAMLINES]))

        # Go back at the end of the file.
        self.file.seek(0, os.SEEK_END)


def _create_array_sequence_from_generator(gen):
    BUFFER_SIZE = 10000000  # About 128 Mb if item shape is 3.

    streamlines = ArraySequence()

    gen = iter(gen)
    try:
        first_element = next(gen)
        gen = itertools.chain([first_element], gen)
    except StopIteration:
        return streamlines

    # Allocated some buffer memory.
    pts = np.asarray(first_element)
    streamlines._data = np.empty((BUFFER_SIZE, pts.shape[1]), dtype=pts.dtype)

    offset = 0
    offsets = []
    lengths = []
    for i, pts in enumerate(gen):
        pts = np.asarray(pts)

        end = offset + len(pts)
        if end >= len(streamlines._data):
            # Resize is needed (at least `len(pts)` items will be added).
            streamlines._data.resize((len(streamlines._data) + len(pts)+BUFFER_SIZE, pts.shape[1]))

        offsets.append(offset)
        lengths.append(len(pts))
        streamlines._data[offset:offset+len(pts)] = pts

        offset += len(pts)

    streamlines._offsets = np.asarray(offsets)
    streamlines._lengths = np.asarray(lengths)

    # Clear unused memory.
    streamlines._data.resize((offset, pts.shape[1]))

    return streamlines


class TckFile(TractogramFile):
    ''' Convenience class to encapsulate TCK file format.

    Notes
    -----
    MRtrix (so its file format: TCK) considers the streamlines to be saved in
    world space. It uses the same convention as Nifti: RAS+ and mm space with
    the coordinate (0,0,0) being at the center of the voxel.
    '''

    # Contants
    MAGIC_NUMBER = "mrtrix tracks"

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : ``Tractogram`` object
            Tractogram that will be contained in this ``TckFile``.

        header : dict (optional)
            Metadata associated to this tractogram file.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm* space
        where coordinate (0,0,0) refers to the center of the voxel.
        """
        if header is None:
            header = TckWriter.create_empty_header()

        super(TckFile, self).__init__(tractogram, header)

    @classmethod
    def get_magic_number(cls):
        ''' Return TRK's magic number. '''
        return cls.MAGIC_NUMBER

    @classmethod
    def support_data_per_point(cls):
        ''' Tells if this tractogram format supports saving data per point. '''
        return False

    @classmethod
    def support_data_per_streamline(cls):
        ''' Tells if this tractogram format supports saving data per streamline. '''
        return False

    @classmethod
    def is_correct_format(cls, fileobj):
        ''' Check if the file is in TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to read from the beginning
            of the TCK header data).

        Returns
        -------
        is_correct_format : boolean
            Returns True if `fileobj` is in TCK format.
        '''
        with Opener(fileobj) as f:
            magic_number = f.fobj.readline()
            f.seek(-len(magic_number), os.SEEK_CUR)
            return magic_number.strip() == cls.MAGIC_NUMBER

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
        tck_file : ``TckFile`` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the returned tractogram are assumed to be in RASmm
        space where coordinate (0,0,0) refers to the center of the voxel.
        '''
        tck_reader = TckReader(fileobj)

        if lazy_load:
            def _read():
                for pts in tck_reader:
                    yield TractogramItem(pts, {}, {})

            tractogram = LazyTractogram.create_from(_read)

        else:
            streamlines = _create_array_sequence_from_generator(tck_reader)
            tractogram = Tractogram(streamlines)

        return cls(tractogram, header=tck_reader.header)

    def save(self, fileobj):
        ''' Saves tractogram to a file-like object using TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to read from the beginning
            of the TCK header data).
        '''
        tck_writer = TckWriter(fileobj, self.header)
        tck_writer.write(self.tractogram)

    def __str__(self):
        ''' Gets a formatted string of the header of a TCK file.

        Returns
        -------
        info : string
            Header information relevant to the TCK format.
        '''
        #trk_reader = TrkReader(fileobj)
        hdr = self.header

        info = ""
        info += "\nMAGIC NUMBER: {0}".format(hdr[Field.MAGIC_NUMBER])
        info += "\n"
        info += "\n".join(["{}: {}".format(k, v) for k, v in self.header.items()])

        return info

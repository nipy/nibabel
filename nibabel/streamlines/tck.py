from __future__ import division

# Documentation available here:
# http://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html?highlight=format#tracks-file-format-tck

import os
import warnings
from collections import OrderedDict

import numpy as np

from nibabel.openers import Opener
from nibabel.py3k import asbytes, asstr

from .array_sequence import ArraySequence
from .tractogram_file import TractogramFile
from .tractogram_file import HeaderError, DataWarning, DataError
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field

MEGABYTE = 1024 * 1024
BUFFER_SIZE = 1000000


def create_empty_header():
    ''' Return an empty compliant TCK header. '''
    header = OrderedDict()

    # Default values
    header[Field.MAGIC_NUMBER] = TckFile.MAGIC_NUMBER
    header[Field.NB_STREAMLINES] = 0
    header['datatype'] = "Float32LE"
    return header


class TckFile(TractogramFile):
    """ Convenience class to encapsulate TCK file format.

    Notes
    -----
    MRtrix (so its file format: TCK) considers the streamline coordinate
    (0,0,0) to be in the center of the voxel which is also the case for
    NiBabel's streamlines internal representation.

    Moreover, streamlines coordinates coming from a TCK file are considered
    to be in world space (RAS+ and mm space). MRtrix refers to that space
    as the "real" or "scanner" space _[1].

    References
    ----------
    [1] http://www.nitrc.org/pipermail/mrtrix-discussion/2014-January/000859.html
    """

    # Contants
    MAGIC_NUMBER = b"mrtrix tracks"
    SUPPORTS_DATA_PER_POINT = False  # Not yet
    SUPPORTS_DATA_PER_STREAMLINE = False  # Not yet

    FIBER_DELIMITER = np.array([[np.nan, np.nan, np.nan]], '<f4')
    EOF_DELIMITER = np.array([[np.inf, np.inf, np.inf]], '<f4')

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : :class:`Tractogram` object
            Tractogram that will be contained in this :class:`TckFile`.

        header : dict (optional)
            Metadata associated to this tractogram file.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+*
        and *mm* space where coordinate (0,0,0) refers to the center
        of the voxel.
        """
        if header is None:
            header = create_empty_header()

        super(TckFile, self).__init__(tractogram, header)

    @classmethod
    def is_correct_format(cls, fileobj):
        """ Check if the file is in TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to read from the beginning
            of the TCK header data). Note that calling this function
            does not change the file position.

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is compatible with TCK format,
            otherwise returns False.
        """
        with Opener(fileobj) as f:
            magic_number = f.fobj.readline()
            f.seek(-len(magic_number), os.SEEK_CUR)
            return magic_number.strip() == cls.MAGIC_NUMBER

    @classmethod
    def load(cls, fileobj, lazy_load=False):
        """ Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to read from the beginning
            of the TCK header). Note that calling this function
            does not change the file position.
        lazy_load : {False, True}, optional
            If True, load streamlines in a lazy manner i.e. they will not be
            kept in memory. Otherwise, load all streamlines in memory.

        Returns
        -------
        tck_file : :class:`TckFile` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+*
        and *mm* space where coordinate (0,0,0) refers to the center
        of the voxel.
        """
        hdr = cls._read_header(fileobj)

        if lazy_load:
            def _read():
                for pts in cls._read(fileobj, hdr):
                    yield TractogramItem(pts, {}, {})

            tractogram = LazyTractogram.from_data_func(_read)

        else:
            tck_reader = cls._read(fileobj, hdr)
            streamlines = ArraySequence(tck_reader)
            tractogram = Tractogram(streamlines)

        tractogram.affine_to_rasmm = np.eye(4)
        return cls(tractogram, header=hdr)

    def save(self, fileobj):
        """ Save tractogram to a filename or file-like object using TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to write from the beginning
            of the TCK header data).
        """
        # Enforce float32 in little-endian byte order for data.
        dtype = np.dtype('<f4')
        header = create_empty_header()

        # Override hdr's fields by those contained in `header`.
        header.update(self.header)

        # Keep counts for correcting incoherent fields or warn.
        nb_streamlines = 0

        with Opener(fileobj, mode="wb") as f:
            # Keep track of the beginning of the header.
            beginning = f.tell()

            # Write temporary header that we will update at the end
            self._write_header(f, header)

            try:
                first_item = next(iter(self.tractogram))
            except StopIteration:
                # Empty tractogram
                header[Field.NB_STREAMLINES] = 0
                # Overwrite header with updated one.
                f.seek(beginning, os.SEEK_SET)
                self._write_header(f, header)

                # Add the EOF_DELIMITER.
                f.write(asbytes(self.EOF_DELIMITER.tostring()))
                return

            data_for_streamline = first_item.data_for_streamline
            if len(data_for_streamline) > 0:
                keys = ", ".join(data_for_streamline.keys())
                msg = ("TCK format does not support saving additional data"
                       " alongside streamlines. Dropping: {}".format(keys))
                warnings.warn(msg, DataWarning)

            data_for_points = first_item.data_for_points
            if len(data_for_points) > 0:
                keys = ", ".join(data_for_points.keys())
                msg = ("TCK format does not support saving additional data"
                       " alongside points. Dropping: {}".format(keys))
                warnings.warn(msg, DataWarning)

            # Make sure streamlines are in rasmm.
            tractogram = self.tractogram.to_world(lazy=True)

            for t in tractogram:
                s = t.streamline.astype(dtype)
                data = np.r_[s, self.FIBER_DELIMITER]
                f.write(asbytes(data.tostring()))
                nb_streamlines += 1

            header[Field.NB_STREAMLINES] = nb_streamlines

            # Add the EOF_DELIMITER.
            f.write(asbytes(self.EOF_DELIMITER.tostring()))

            # Overwrite header with updated one.
            f.seek(beginning, os.SEEK_SET)
            self._write_header(f, header)

    @staticmethod
    def _write_header(fileobj, header):
        """ Write TCK header to file-like object.

        Parameters
        ----------
        fileobj : file-like object
            An open file-like object pointing to TCK file (and ready to read
            from the beginning of the TCK header).
        """
        # Fields to exclude
        exclude = [Field.MAGIC_NUMBER, Field.NB_STREAMLINES,
                   "datatype", "file"]

        lines = []
        lines.append(header[Field.MAGIC_NUMBER])
        lines.append("count: {0:010}".format(header[Field.NB_STREAMLINES]))
        lines.append("datatype: Float32LE")  # Always Float32LE.
        lines.extend(["{0}: {1}".format(k, v)
                      for k, v in header.items() if k not in exclude])
        lines.append("file: . ")  # Manually add this last field.
        out = "\n".join((asstr(line).replace('\n', '\t') for line in lines))
        fileobj.write(asbytes(out))

        # Compute offset to the beginning of the binary data.
        # We add 5 for "\nEND\n" that will be added just before the data.
        tentative_offset = len(out) + 5

        # Count the number of characters needed to write the offset in ASCII.
        offset = tentative_offset + len(str(tentative_offset))

        # The new offset might need one more character to write it in ASCII.
        # e.g. offset = 98 (i.e. 2 char.), so offset += 2 = 100 (i.e. 3 char.)
        # thus the final offset = 101.
        if len(str(tentative_offset)) != len(str(offset)):
            offset += 1  # +1, we need one more character for that new digit.

        fileobj.write(asbytes(str(offset) + "\n"))
        fileobj.write(asbytes(b"END\n"))

    @staticmethod
    def _read_header(fileobj):
        """ Reads a TCK header from a file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to read from the beginning
            of the TCK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        header : dict
            Metadata associated with this tractogram file.
        """
        # Record start position if this is a file-like object
        start_position = fileobj.tell() if hasattr(fileobj, 'tell') else None

        with Opener(fileobj) as f:
            # Read magic number
            magic_number = asstr(f.fobj.readline())

            # Read all key-value pairs contained in the header.
            buf = asstr(f.fobj.readline())
            while not buf.rstrip().endswith("END"):
                buf += asstr(f.fobj.readline())

        # Build header dictionary from the buffer.
        hdr = dict(item.split(': ') for item in buf.rstrip().split('\n')[:-1])
        hdr[Field.MAGIC_NUMBER] = magic_number

        # Check integrity of TCK header.
        if 'datatype' not in hdr:
            raise HeaderError("Missing 'datatype' attribute in TCK header.")

        if not hdr['datatype'].startswith('Float32'):
            msg = ("TCK only supports float32 dtype but 'dataype: {}' was"
                   " specified in the header.").format(hdr['datatype'])
            raise HeaderError(msg)

        if 'file' not in hdr:
            raise HeaderError("Missing 'file' attribute in TCK header.")

        if hdr['file'].split()[0] != '.':
            msg = ("TCK only supports single-file - in other words the"
                   " filename part must be specified as '.' but '{}' was"
                   " specified.").format(hdr['file'].split()[0])
            raise HeaderError("Missing 'file' attribute in TCK header.")

        # Set endianness and _dtype attributes in the header.
        hdr[Field.ENDIANNESS] = '<'
        if hdr['datatype'].endswith('BE'):
            hdr[Field.ENDIANNESS] = '>'

        hdr['_dtype'] = np.dtype(hdr[Field.ENDIANNESS] + 'f4')

        # Keep the file position where the data begin.
        hdr['_offset_data'] = int(hdr['file'].split()[1])

        # Set the file position where it was, if it was previously open.
        if start_position is not None:
            fileobj.seek(start_position, os.SEEK_SET)

        return hdr

    @staticmethod
    def _read(fileobj, header, buffer_size=4):
        """ Return generator that reads TCK data from `fileobj` given `header`

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TCK file (and ready to read from the beginning
            of the TCK header). Note that calling this function
            does not change the file position.
        header : dict
            Metadata associated with this tractogram file.
        buffer_size : float, optional
            Size (in Mb) for buffering.

        Yields
        ------
        points : ndarray of shape (n_pts, 3)
            Streamline points
        """
        buffer_size = buffer_size * MEGABYTE
        buffer_size += 3 - (buffer_size % 3)  # Make it a multiple of 3.
        dtype = header["_dtype"]

        with Opener(fileobj) as f:
            start_position = f.tell()

            # Set the file position at the beginning of the data.
            f.seek(header["_offset_data"], os.SEEK_SET)

            eof = False
            buff = b""
            pts = []

            i = 0

            while not eof or not np.all(np.isinf(pts)):

                if not eof:
                    bytes_read = f.read(buffer_size)
                    buff += bytes_read
                    eof = len(bytes_read) == 0

                # Read floats.
                pts = np.frombuffer(buff, dtype=dtype)

                # Convert data to little-endian if needed.
                if dtype != '<f4':
                    pts = pts.astype('<f4')

                pts = pts.reshape([-1, 3])
                idx_nan = np.arange(len(pts))[np.isnan(pts[:, 0])]

                # Make sure we've read enough to find a streamline delimiter.
                if len(idx_nan) == 0:
                    # If we've read the whole file, then fail.
                    if eof and not np.all(np.isinf(pts)):
                        msg = ("Cannot find a streamline delimiter. This file"
                               " might be corrupted.")
                        raise DataError(msg)

                    # Otherwise read a bit more.
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
                nb_tiplets_to_remove = nb_pts_total + len(idx_nan)
                nb_bytes_to_remove = nb_tiplets_to_remove * 3 * dtype.itemsize
                buff = buff[nb_bytes_to_remove:]

            # In case the 'count' field was not provided.
            header[Field.NB_STREAMLINES] = i

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)

    def __str__(self):
        ''' Gets a formatted string of the header of a TCK file.

        Returns
        -------
        info : string
            Header information relevant to the TCK format.
        '''
        hdr = self.header

        info = ""
        info += "\nMAGIC NUMBER: {0}".format(hdr[Field.MAGIC_NUMBER])
        info += "\n"
        info += "\n".join(["{}: {}".format(k, v)
                           for k, v in hdr.items() if not k.startswith('_')])
        return info

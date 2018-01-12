""" Read / write access to TCK streamlines format.

TCK format is defined at
http://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html?highlight=format#tracks-file-format-tck
"""
from __future__ import division

import os
import warnings

import numpy as np

from nibabel.openers import Opener
from nibabel.py3k import asbytes, asstr

from .array_sequence import ArraySequence
from .tractogram_file import TractogramFile
from .tractogram_file import HeaderWarning, DataWarning
from .tractogram_file import HeaderError, DataError
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field
from .utils import peek_next

MEGABYTE = 1024 * 1024


class TckFile(TractogramFile):
    """ Convenience class to encapsulate TCK file format.

    Notes
    -----
    MRtrix (so its file format: TCK) considers streamlines coordinates
    to be in world space (RAS+ and mm space). MRtrix refers to that space
    as the "real" or "scanner" space [1]_.

    Moreover, when streamlines are mapped back to voxel space [2]_, a
    streamline point located at an integer coordinate (i,j,k) is considered
    to be at the center of the corresponding voxel. This is in contrast with
    TRK's internal convention where it would have referred to a corner.

    NiBabel's streamlines internal representation follows the same
    convention as MRtrix.

    References
    ----------
    [1] http://www.nitrc.org/pipermail/mrtrix-discussion/2014-January/000859.html
    [2] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-in-voxel-space
    """
    # Constants
    MAGIC_NUMBER = "mrtrix tracks"
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
        header : None or dict, optional
            Metadata associated to this tractogram file. If None, make
            default empty header.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm*
        space. It is also assumed that when streamlines are mapped back to
        voxel space, a streamline point located at an integer coordinate
        (i,j,k) is considered to be at the center of the corresponding voxel.
        This is in contrast with TRK's internal convention where it would
        have referred to a corner.
        """
        super(TckFile, self).__init__(tractogram, header)

    @classmethod
    def is_correct_format(cls, fileobj):
        """ Check if the file is in TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is compatible with TCK format,
            otherwise returns False.
        """
        with Opener(fileobj) as f:
            magic_number = asstr(f.fobj.readline())
            f.seek(-len(magic_number), os.SEEK_CUR)

        return magic_number.strip() == cls.MAGIC_NUMBER

    @classmethod
    def create_empty_header(cls):
        """ Return an empty compliant TCK header as dict """
        header = {}

        # Default values
        header[Field.MAGIC_NUMBER] = cls.MAGIC_NUMBER
        header[Field.NB_STREAMLINES] = 0
        header['datatype'] = "Float32LE"
        return header

    @classmethod
    def load(cls, fileobj, lazy_load=False):
        """ Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
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
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm*
        space. It is also assumed that when streamlines are mapped back to
        voxel space, a streamline point located at an integer coordinate
        (i,j,k) is considered to be at the center of the corresponding voxel.
        This is in contrast with TRK's internal convention where it would
        have referred to a corner.
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

        # By definition.
        tractogram.affine_to_rasmm = np.eye(4)
        hdr[Field.VOXEL_TO_RASMM] = np.eye(4)

        return cls(tractogram, header=hdr)

    def _finalize_header(self, f, header, offset=0):
        # Overwrite header with updated one.
        f.seek(offset, os.SEEK_SET)
        self._write_header(f, header)

    def save(self, fileobj):
        """ Save tractogram to a filename or file-like object using TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to write from the
            beginning of the TCK header data).
        """
        # Enforce float32 in little-endian byte order for data.
        dtype = np.dtype('<f4')
        header = self.create_empty_header()

        # Override hdr's fields by those contained in `header`.
        header.update(self.header)

        # Keep counts for correcting incoherent fields or warn.
        nb_streamlines = 0

        with Opener(fileobj, mode="wb") as f:
            # Keep track of the beginning of the header.
            beginning = f.tell()

            # Write temporary header that we will update at the end
            self._write_header(f, header)

            # Make sure streamlines are in rasmm.
            tractogram = self.tractogram.to_world(lazy=True)
            # Assume looping over the streamlines can be done only once.
            tractogram = iter(tractogram)

            try:
                # Use the first element to check
                #  1) the tractogram is not empty;
                #  2) quantity of information saved along each streamline.
                first_item, tractogram = peek_next(tractogram)
            except StopIteration:
                # Empty tractogram
                header[Field.NB_STREAMLINES] = 0
                self._finalize_header(f, header, offset=beginning)

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

            for t in tractogram:
                data = np.r_[t.streamline, self.FIBER_DELIMITER]
                f.write(data.astype(dtype).tostring())
                nb_streamlines += 1

            header[Field.NB_STREAMLINES] = nb_streamlines

            # Add the EOF_DELIMITER.
            f.write(asbytes(self.EOF_DELIMITER.tostring()))
            self._finalize_header(f, header, offset=beginning)

    @staticmethod
    def _write_header(fileobj, header):
        """ Write TCK header to file-like object.

        Parameters
        ----------
        fileobj : file-like object
            An open file-like object in binary mode pointing to TCK file (and
            ready to read from the beginning of the TCK header).
        """
        # Fields to exclude
        exclude = [Field.MAGIC_NUMBER,  # Handled separately.
                   Field.NB_STREAMLINES,  # Handled separately.
                   Field.ENDIANNESS,  # Handled separately.
                   Field.VOXEL_TO_RASMM,  # Streamlines are always in RAS+ mm.
                   "count", "datatype", "file"]  # Fields being replaced.

        lines = []
        lines.append(asstr(header[Field.MAGIC_NUMBER]))
        lines.append("count: {0:010}".format(header[Field.NB_STREAMLINES]))
        lines.append("datatype: Float32LE")  # Always Float32LE.
        lines.extend(["{0}: {1}".format(k, v)
                      for k, v in header.items()
                      if k not in exclude and not k.startswith("_")])
        lines.append("file: . ")  # Manually add this last field.
        out = "\n".join(lines)

        # Check the header is well formatted.
        if out.count("\n") > len(lines) - 1:  # \n only allowed between lines.
            msg = "Key-value pairs cannot contain '\\n':\n{}".format(out)
            raise HeaderError(msg)

        if out.count(":") > len(lines) - 1:
            # : only one per line (except the last one which contains END).
            msg = "Key-value pairs cannot contain ':':\n{}".format(out)
            raise HeaderError(msg)

        # Write header to file.
        fileobj.write(asbytes(out))

        hdr_len_no_offset = len(out) + 5
        # Need to add number of bytes to store offset as decimal string. We
        # start with estimate without string, then update if the
        # offset-as-decimal-string got longer after adding length of the
        # offset string.
        new_offset = -1
        old_offset = hdr_len_no_offset
        while new_offset != old_offset:
            old_offset = new_offset
            new_offset = hdr_len_no_offset + len(str(old_offset))

        fileobj.write(asbytes(str(new_offset) + "\n"))
        fileobj.write(asbytes("END\n"))

    @staticmethod
    def _read_header(fileobj):
        """ Reads a TCK header from a file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
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
            magic_number = f.fobj.readline().strip()

            # Read all key-value pairs contained in the header.
            buf = asstr(f.fobj.readline())
            while not buf.rstrip().endswith("END"):
                buf += asstr(f.fobj.readline())

            offset_data = f.tell()

        # Build header dictionary from the buffer.
        hdr = dict(item.split(': ') for item in buf.rstrip().split('\n')[:-1])
        hdr[Field.MAGIC_NUMBER] = magic_number

        # Check integrity of TCK header.
        if 'datatype' not in hdr:
            msg = ("Missing 'datatype' attribute in TCK header."
                   " Assuming it is Float32LE.")
            warnings.warn(msg, HeaderWarning)
            hdr['datatype'] = "Float32LE"

        if not hdr['datatype'].startswith('Float32'):
            msg = ("TCK only supports float32 dtype but 'datatype: {}' was"
                   " specified in the header.").format(hdr['datatype'])
            raise HeaderError(msg)

        if 'file' not in hdr:
            msg = ("Missing 'file' attribute in TCK header."
                   " Will try to guess it.")
            warnings.warn(msg, HeaderWarning)
            hdr['file'] = '. {}'.format(offset_data)

        if hdr['file'].split()[0] != '.':
            msg = ("TCK only supports single-file - in other words the"
                   " filename part must be specified as '.' but '{}' was"
                   " specified.").format(hdr['file'].split()[0])
            raise HeaderError("Missing 'file' attribute in TCK header.")

        # Set endianness and _dtype attributes in the header.
        hdr[Field.ENDIANNESS] = '>' if hdr['datatype'].endswith('BE') else '<'

        hdr['_dtype'] = np.dtype(hdr[Field.ENDIANNESS] + 'f4')

        # Keep the file position where the data begin.
        hdr['_offset_data'] = int(hdr['file'].split()[1])

        # Set the file position where it was, if it was previously open.
        if start_position is not None:
            fileobj.seek(start_position, os.SEEK_SET)

        return hdr

    @classmethod
    def _read(cls, fileobj, header, buffer_size=4):
        """ Return generator that reads TCK data from `fileobj` given `header`

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
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
        dtype = header["_dtype"]
        coordinate_size = 3 * dtype.itemsize
        # Make buffer_size an integer and a multiple of coordinate_size.
        buffer_size = int(buffer_size * MEGABYTE)
        buffer_size += coordinate_size - (buffer_size % coordinate_size)

        # Markers for streamline end and file end
        fiber_marker = cls.FIBER_DELIMITER.astype(dtype).tostring()
        eof_marker = cls.EOF_DELIMITER.astype(dtype).tostring()

        with Opener(fileobj) as f:
            start_position = f.tell()

            # Set the file position at the beginning of the data.
            f.seek(header["_offset_data"], os.SEEK_SET)

            eof = False
            buffs = []
            n_streams = 0

            while not eof:

                bytes_read = f.read(buffer_size)
                buffs.append(bytes_read)
                eof = len(bytes_read) != buffer_size

                # Make sure we've read enough to find a streamline delimiter.
                if fiber_marker not in bytes_read:
                    # If we've read the whole file, then fail.
                    if eof:
                        # Could have minimal buffering, and have read only the
                        # EOF delimiter
                        buffs = [b''.join(buffs)]
                        if not buffs[0] == eof_marker:
                            raise DataError(
                                "Cannot find a streamline delimiter. This file"
                                " might be corrupted.")
                    else:
                        # Otherwise read a bit more.
                        continue

                all_parts = b''.join(buffs).split(fiber_marker)
                point_parts, buffs = all_parts[:-1], all_parts[-1:]
                point_parts = [p for p in point_parts if p != b'']

                for point_part in point_parts:
                    # Read floats.
                    pts = np.frombuffer(point_part, dtype=dtype)
                    # Enforce ability to write to underlying bytes object
                    pts.flags.writeable = True
                    # Convert data to little-endian if needed.
                    yield pts.astype('<f4', copy=False).reshape([-1, 3])

                n_streams += len(point_parts)

            if not buffs[-1] == eof_marker:
                raise DataError("Expecting end-of-file marker 'inf inf inf'")

            # In case the 'count' field was not provided.
            header[Field.NB_STREAMLINES] = n_streams

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)

    def __str__(self):
        """ Gets a formatted string of the header of a TCK file.

        Returns
        -------
        info : string
            Header information relevant to the TCK format.
        """
        hdr = self.header

        info = ""
        info += "\nMAGIC NUMBER: {0}".format(hdr[Field.MAGIC_NUMBER])
        info += "\n"
        info += "\n".join(["{}: {}".format(k, v)
                           for k, v in hdr.items() if not k.startswith('_')])
        return info


from nibabel.streamlines.header import Field

from ..externals.six.moves import zip_longest


class HeaderError(Exception):
    pass


class DataError(Exception):
    pass


class Streamlines(object):
    ''' Class containing information about streamlines.

    Streamlines objects have three main properties: ``points``, ``scalars``
    and ``properties``. Streamlines objects can be iterate over producing
    tuple of ``points``, ``scalars`` and ``properties`` for each streamline.

    Parameters
    ----------
    points : sequence of ndarray of shape (N, 3)
        Sequence of T streamlines. One streamline is an ndarray of shape (N, 3)
        where N is the number of points in a streamline.

    scalars : sequence of ndarray of shape (N, M)
        Sequence of T ndarrays of shape (N, M) where T is the number of
        streamlines defined by ``points``, N is the number of points
        for a particular streamline and M is the number of scalars
        associated to each point (excluding the three coordinates).

    properties : sequence of ndarray of shape (P,)
        Sequence of T ndarrays of shape (P,) where T is the number of
        streamlines defined by ``points``, P is the number of properties
        associated to each streamlines.

    hdr : dict
        Header containing meta information about the streamlines. For a list
        of common header's fields to use as keys see `nibabel.streamlines.Field`.
    '''
    def __init__(self, points=[], scalars=[], properties=[], hdr={}):
        self.hdr = hdr

        self.points      = points
        self.scalars     = scalars
        self.properties  = properties
        self.data        = lambda: zip_longest(self.points, self.scalars, self.properties, fillvalue=[])

        try:
            self.length = len(points)
        except:
            if Field.NB_STREAMLINES in hdr:
                self.length = hdr[Field.NB_STREAMLINES]
            else:
                raise HeaderError(("Neither parameter 'points' nor 'hdr' contain information about"
                                  " number of streamlines. Use key '{0}' to set the number of "
                                  "streamlines in 'hdr'.").format(Field.NB_STREAMLINES))

    def get_header(self):
        return self.hdr

    @property
    def points(self):
        return self._points()

    @points.setter
    def points(self, value):
        self._points = value if callable(value) else (lambda: value)

    @property
    def scalars(self):
        return self._scalars()

    @scalars.setter
    def scalars(self, value):
        self._scalars = value if callable(value) else lambda: value

    @property
    def properties(self):
        return self._properties()

    @properties.setter
    def properties(self, value):
        self._properties = value if callable(value) else lambda: value

    def __iter__(self):
        return self.data()

    def __len__(self):
        return self.length


class StreamlinesFile:
    ''' Convenience class to encapsulate streamlines file format. '''

    @classmethod
    def get_magic_number(cls):
        ''' Return streamlines file's magic number. '''
        raise NotImplementedError()

    @classmethod
    def is_correct_format(cls, fileobj):
        ''' Check if the file has the right streamlines file format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header)

        Returns
        -------
        is_correct_format : boolean
            Returns True if `fileobj` is in the right streamlines file format.
        '''
        raise NotImplementedError()

    @classmethod
    def get_empty_header(cls):
        ''' Return an empty streamlines file's header. '''
        raise NotImplementedError()

    @classmethod
    def load(cls, fileobj, lazy_load=True):
        ''' Loads streamlines from a file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header)

        lazy_load : boolean
            Load streamlines in a lazy manner i.e. they will not be kept
            in memory. For postprocessing speed, turn off this option.

        Returns
        -------
        streamlines : Streamlines object
            Returns an object containing streamlines' data and header
            information. See 'nibabel.Streamlines'.
        '''
        raise NotImplementedError()

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
            opened and ready to write.
        '''
        raise NotImplementedError()

    @staticmethod
    def pretty_print(streamlines):
        ''' Gets a formatted string contaning header's information
        relevant to the streamlines file format.

        Parameters
        ----------
        streamlines : Streamlines object
            Object containing streamlines' data and header information.
            See 'nibabel.Streamlines'.

        Returns
        -------
        info : string
            Header's information relevant to the streamlines file format.
        '''
        raise NotImplementedError()


class DynamicStreamlineFile(StreamlinesFile):
    ''' Convenience class to encapsulate streamlines file format
    that supports appending streamlines to an existing file.
    '''

    def append(self, streamlines):
        raise NotImplementedError()

    def __iadd__(self, streamlines):
        return self.append(streamlines)

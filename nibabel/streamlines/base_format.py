import numpy as np
from warnings import warn

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
    points : list of ndarray of shape (N, 3)
        Sequence of T streamlines. One streamline is an ndarray of shape (N, 3)
        where N is the number of points in a streamline.

    scalars : list of ndarray of shape (N, M)
        Sequence of T ndarrays of shape (N, M) where T is the number of
        streamlines defined by ``points``, N is the number of points
        for a particular streamline and M is the number of scalars
        associated to each point (excluding the three coordinates).

    properties : list of ndarray of shape (P,)
        Sequence of T ndarrays of shape (P,) where T is the number of
        streamlines defined by ``points``, P is the number of properties
        associated to each streamline.

    hdr : dict
        Header containing meta information about the streamlines. For a list
        of common header's fields to use as keys see `nibabel.streamlines.Field`.
    '''
    def __init__(self, points=[], scalars=[], properties=[]):  #, hdr={}):
        # Create basic header from given informations.
        self._header = {}
        self._header[Field.VOXEL_TO_WORLD] = np.eye(4)

        self.points      = points
        self.scalars     = scalars
        self.properties  = properties


    @property
    def header(self):
        return self._header

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value
        self._header[Field.NB_STREAMLINES] = len(self.points)

    @property
    def scalars(self):
        return self._scalars

    @scalars.setter
    def scalars(self, value):
        self._scalars = value
        self._header[Field.NB_SCALARS_PER_POINT] = 0
        if len(self.scalars) > 0:
            self._header[Field.NB_SCALARS_PER_POINT] = len(self.scalars[0])

    @property
    def properties(self):
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value
        self._header[Field.NB_PROPERTIES_PER_STREAMLINE] = 0
        if len(self.properties) > 0:
            self._header[Field.NB_PROPERTIES_PER_STREAMLINE] = len(self.properties[0])

    def __iter__(self):
        return zip_longest(self.points, self.scalars, self.properties, fillvalue=[])

    def __getitem__(self, idx):
        pts = self.points[idx]
        scalars = []
        if len(self.scalars) > 0:
            scalars = self.scalars[idx]

        properties = []
        if len(self.properties) > 0:
            properties = self.properties[idx]

        return pts, scalars, properties

    def __len__(self):
        return len(self.points)


class LazyStreamlines(Streamlines):
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
        associated to each streamline.

    hdr : dict
        Header containing meta information about the streamlines. For a list
        of common header's fields to use as keys see `nibabel.streamlines.Field`.
    '''
    def __init__(self, points=[], scalars=[], properties=[], data=None, count=None, getitem=None):  #, hdr={}):
        super(LazyStreamlines, self).__init__(points, scalars, properties)

        self._data = lambda: zip_longest(self.points, self.scalars, self.properties, fillvalue=[])
        if data is not None:
            self._data = data if callable(data) else lambda: data

        self._count = count
        self._getitem = getitem

    @property
    def points(self):
        return self._points()

    @points.setter
    def points(self, value):
        self._points = value if callable(value) else lambda: value

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

    def __getitem__(self, idx):
        if self._getitem is None:
            raise AttributeError('`LazyStreamlines` does not support indexing.')

        return self._getitem(idx)

    def __iter__(self):
        return self._data()

    def __len__(self):
        # If length is unknown, we'll try to get it as rapidely and accurately as possible.
        if self._count is None:
            # Length might be contained in the header.
            if Field.NB_STREAMLINES in self.header:
                return self.header[Field.NB_STREAMLINES]

        if callable(self._count):
            # Length might be obtained by re-parsing the file (if streamlines come from one).
            self._count = self._count()

        if self._count is None:
            try:
                # Will work if `points` is a finite sequence (e.g. list, ndarray)
                self._count = len(self.points)
            except:
                pass

        if self._count is None:
            # As a last resort, count them by iterating through the list of points (while keeping a copy).
            warn("Number of streamlines will be determined manually by looping"
                 " through the streamlines. Note this will consume any"
                 " generator used to create this `Streamlines`object. If you"
                 " know the actual number of streamlines, you might want to"
                 " set `Field.NB_STREAMLINES` of `self.header` beforehand.")

            return sum(1 for _ in self)

        return self._count


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

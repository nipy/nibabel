from warnings import warn

from nibabel.externals.six.moves import zip_longest
from nibabel.affines import apply_affine

from .header import StreamlinesHeader
from .utils import pop


class UsageWarning(Warning):
    pass


class HeaderWarning(Warning):
    pass


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
    '''
    def __init__(self, points=None, scalars=None, properties=None):
        self._header = StreamlinesHeader()
        self.points = points
        self.scalars = scalars
        self.properties = properties

    @property
    def header(self):
        return self._header

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value if value else []
        self.header.nb_streamlines = len(self.points)

    @property
    def scalars(self):
        return self._scalars

    @scalars.setter
    def scalars(self, value):
        self._scalars = value if value else []
        self.header.nb_scalars_per_point = 0

        if len(self.scalars) > 0 and len(self.scalars[0]) > 0:
            self.header.nb_scalars_per_point = len(self.scalars[0][0])

    @property
    def properties(self):
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value if value else []
        self.header.nb_properties_per_streamline = 0

        if len(self.properties) > 0:
            self.header.nb_properties_per_streamline = len(self.properties[0])

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

        if type(idx) is slice:
            return list(zip_longest(pts, scalars, properties, fillvalue=[]))

        return pts, scalars, properties

    def __len__(self):
        return len(self.points)

    def to_world_space(self, as_generator=False):
        affine = self.header.voxel_to_world
        new_points = (apply_affine(affine, pts) for pts in self.points)

        if not as_generator:
            return list(new_points)

        return new_points


class LazyStreamlines(Streamlines):
    ''' Class containing information about streamlines.

    Streamlines objects have four main properties: ``header``, ``points``,
    ``scalars`` and ``properties``. Streamlines objects are iterable and
    produce tuple of ``points``, ``scalars`` and ``properties`` for each
    streamline.

    Parameters
    ----------
    points_func : coroutine ouputting (N,3) array-like (optional)
        Function yielding streamlines' points. One streamline's points is
        an array-like of shape (N,3) where N is the number of points in a
        streamline.

    scalars_func : coroutine ouputting (N,M) array-like (optional)
        Function yielding streamlines' scalars. One streamline's scalars is
        an array-like of shape (N,M) where N is the number of points for a
        particular streamline and M is the number of scalars associated to
        each point (excluding the three coordinates).

    properties_func : coroutine ouputting (P,) array-like (optional)
        Function yielding streamlines' properties. One streamline's properties
        is an array-like of shape (P,) where P is the number of properties
        associated to each streamline.

    getitem_func : function `idx -> 3-tuples` (optional)
        Function returning streamlines (one or a list of 3-tuples) given
        an index or a slice (i.e. the __getitem__ function to use).

    Notes
    -----
    If provided, ``scalars`` and ``properties`` must yield the same number of
    values as ``points``.
    '''
    def __init__(self, points_func=lambda:[], scalars_func=lambda: [], properties_func=lambda: [], getitem_func=None):
        super(LazyStreamlines, self).__init__(points_func, scalars_func, properties_func)
        self._data = lambda: zip_longest(self.points, self.scalars, self.properties, fillvalue=[])
        self._getitem = getitem_func

    @classmethod
    def create_from_data(cls, data_func):
        ''' Saves streamlines to a file-like object.

        Parameters
        ----------
        data_func : coroutine ouputting tuple (optional)
            Function yielding 3-tuples, (streamline's points, streamline's
            scalars, streamline's properties). A streamline's points is an
            array-like of shape (N,3), a streamline's scalars is an array-like
            of shape (N,M) and streamline's properties is an array-like of
            shape (P,) where N is the number of points for a particular
            streamline, M is the number of scalars associated to each point
            (excluding the three coordinates) and P is the number of properties
            associated to each streamline.
        '''
        if not callable(data_func):
            raise TypeError("`data` must be a coroutine.")

        lazy_streamlines = cls()
        lazy_streamlines._data = data_func
        lazy_streamlines.points = lambda: (x[0] for x in data_func())
        lazy_streamlines.scalars = lambda: (x[1] for x in data_func())
        lazy_streamlines.properties = lambda: (x[2] for x in data_func())
        return lazy_streamlines

    @property
    def points(self):
        return self._points()

    @points.setter
    def points(self, value):
        if not callable(value):
            raise TypeError("`points` must be a coroutine.")

        self._points = value

    @property
    def scalars(self):
        return self._scalars()

    @scalars.setter
    def scalars(self, value):
        if not callable(value):
            raise TypeError("`scalars` must be a coroutine.")

        self._scalars = value
        self.header.nb_scalars_per_point = 0
        scalars = pop(self.scalars)
        if scalars is not None and len(scalars) > 0:
            self.header.nb_scalars_per_point = len(scalars[0])

    @property
    def properties(self):
        return self._properties()

    @properties.setter
    def properties(self, value):
        if not callable(value):
            raise TypeError("`properties` must be a coroutine.")

        self._properties = value
        self.header.nb_properties_per_streamline = 0
        properties = pop(self.properties)
        if properties is not None:
            self.header.nb_properties_per_streamline = len(properties)

    def __getitem__(self, idx):
        if self._getitem is None:
            raise AttributeError('`LazyStreamlines` does not support indexing.')

        return self._getitem(idx)

    def __iter__(self):
        i = 0
        for i, s in enumerate(self._data(), start=1):
            yield s

        # To be safe, update information about number of streamlines.
        self.header.nb_streamlines = i

    def __len__(self):
        # If length is unknown, we obtain it by iterating through streamlines.
        if self.header.nb_streamlines is None:
            warn("Number of streamlines will be determined manually by looping"
                 " through the streamlines. If you know the actual number of"
                 " streamlines, you might want to set it beforehand via"
                 " `self.header.nb_streamlines`."
                 " Note this will consume any generators used to create this"
                 " `LazyStreamlines` object.", UsageWarning)
            return sum(1 for _ in self)

        return self.header.nb_streamlines

    def to_world_space(self):
        return super(LazyStreamlines, self).to_world_space(as_generator=True)


class StreamlinesFile:
    ''' Convenience class to encapsulate streamlines file format. '''

    @classmethod
    def get_magic_number(cls):
        ''' Returns streamlines file's magic number. '''
        raise NotImplementedError()

    @classmethod
    def can_save_scalars(cls):
        ''' Tells if the streamlines format supports saving scalars. '''
        raise NotImplementedError()

    @classmethod
    def can_save_properties(cls):
        ''' Tells if the streamlines format supports saving properties. '''
        raise NotImplementedError()

    @classmethod
    def is_correct_format(cls, fileobj):
        ''' Checks if the file has the right streamlines file format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).

        Returns
        -------
        is_correct_format : boolean
            Returns True if `fileobj` is in the right streamlines file format.
        '''
        raise NotImplementedError()

    @staticmethod
    def load(fileobj, ref, lazy_load=True):
        ''' Loads streamlines from a file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).

        ref : filename | `Nifti1Image` object | 2D array (4,4)
            Reference space where streamlines live in `fileobj`.

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

    @staticmethod
    def save(streamlines, fileobj, ref=None):
        ''' Saves streamlines to a file-like object.

        Parameters
        ----------
        streamlines : Streamlines object
            Object containing streamlines' data and header information.
            See 'nibabel.Streamlines'.

        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            opened and ready to write.

        ref : filename | `Nifti1Image` object | 2D array (4,4) (optional)
            Reference space where streamlines will live in `fileobj`.
        '''
        raise NotImplementedError()

    @staticmethod
    def pretty_print(streamlines):
        ''' Gets a formatted string of the header of a streamlines file format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).

        Returns
        -------
        info : string
            Header information relevant to the streamlines file format.
        '''
        raise NotImplementedError()


# class DynamicStreamlineFile(StreamlinesFile):
#     ''' Convenience class to encapsulate streamlines file format
#     that supports appending streamlines to an existing file.
#     '''

#     def append(self, streamlines):
#         raise NotImplementedError()

#     def __iadd__(self, streamlines):
#         return self.append(streamlines)

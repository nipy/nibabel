import itertools
import numpy as np
from warnings import warn

from abc import ABCMeta, abstractmethod, abstractproperty

from nibabel.externals.six.moves import zip_longest
from nibabel.affines import apply_affine

from .header import TractogramHeader
from .utils import pop


class UsageWarning(Warning):
    pass


class HeaderWarning(Warning):
    pass


class HeaderError(Exception):
    pass


class DataError(Exception):
    pass


class CompactList(object):
    """ Class for compacting list of ndarrays with matching shape except for
    the first dimension.
    """
    def __init__(self, iterable=None):
        """
        Parameters
        ----------
        iterable : iterable (optional)
            If specified, create a ``CompactList`` object initialized from
            iterable's items. Otherwise, create an empty ``CompactList``.
        """
        # Create new empty `CompactList` object.
        self._data = None
        self._offsets = []
        self._lengths = []

        if iterable is not None:
            # Initialize the `CompactList` object from iterable's item.
            BUFFER_SIZE = 10000000  # About 128 Mb if item shape is 3.

            offset = 0
            for i, e in enumerate(iterable):
                e = np.asarray(e)
                if i == 0:
                    self._data = np.empty((BUFFER_SIZE,) + e.shape[1:], dtype=e.dtype)

                end = offset + len(e)
                if end >= len(self._data):
                    # Resize is needed (at least `len(e)` items will be added).
                    self._data.resize((len(self._data) + len(e)+BUFFER_SIZE,) + self.shape)

                self._offsets.append(offset)
                self._lengths.append(len(e))
                self._data[offset:offset+len(e)] = e
                offset += len(e)

            # Clear unused memory.
            if self._data is not None:
                self._data.resize((offset,) + self.shape)

    @property
    def shape(self):
        """ Returns the matching shape of the elements in this compact list. """
        if self._data is None:
            return None

        return self._data.shape[1:]

    def append(self, element):
        """ Appends `element` to this compact list.

        Parameters
        ----------
        element : ndarray
            Element to append. The shape must match already inserted elements
            shape except for the first dimension.

        Notes
        -----
        If you need to add multiple elements you should consider
        `CompactList.extend`.
        """
        if self._data is None:
            self._data = np.asarray(element).copy()
            self._offsets.append(0)
            self._lengths.append(len(element))
            return

        if element.shape[1:] != self.shape:
            raise ValueError("All dimensions, except the first one, must match exactly")

        self._offsets.append(len(self._data))
        self._lengths.append(len(element))
        self._data = np.append(self._data, element, axis=0)

    def extend(self, elements):
        """ Appends all `elements` to this compact list.

        Parameters
        ----------
        element : list of ndarrays, ``CompactList`` object
            Elements to append. The shape must match already inserted elements
            shape except for the first dimension.
        """
        if isinstance(elements, CompactList):
            self._data = np.concatenate([self._data, elements._data], axis=0)
            offset = self._offsets[-1] + self._lengths[-1] if len(self) > 0 else 0
            self._lengths.extend(elements._lengths)
            self._offsets.extend(np.cumsum([offset] + elements._lengths).tolist()[:-1])
        else:
            self._data = np.concatenate([self._data] + list(elements), axis=0)
            offset = self._offsets[-1] + self._lengths[-1] if len(self) > 0 else 0
            lengths = map(len, elements)
            self._lengths.extend(lengths)
            self._offsets.extend(np.cumsum([offset] + lengths).tolist()[:-1])

    def copy(self):
        """ Creates a copy of this ``CompactList`` object. """
        # We cannot just deepcopy this object since we don't know if it has been created
        # using slicing. If it is the case, `self.data` probably contains more data than necessary
        # so we copy only elements according to `self._offsets`.
        compact_list = CompactList()
        total_lengths = np.sum(self._lengths)
        compact_list._data = np.empty((total_lengths,) + self._data.shape[1:], dtype=self._data.dtype)

        cur_offset = 0
        for offset, lengths in zip(self._offsets, self._lengths):
            compact_list._offsets.append(cur_offset)
            compact_list._lengths.append(lengths)
            compact_list._data[cur_offset:cur_offset+lengths] = self._data[offset:offset+lengths]
            cur_offset += lengths

        return compact_list

    def __getitem__(self, idx):
        """ Gets element(s) through indexing.

        Parameters
        ----------
        idx : int, slice or list
            Index of the element(s) to get.

        Returns
        -------
        ndarray object(s)
            When `idx` is a int, returns a single ndarray.
            When `idx` is either a slice or a list, returns a list of ndarrays.
        """
        if isinstance(idx, int) or isinstance(idx, np.integer):
            return self._data[self._offsets[idx]:self._offsets[idx]+self._lengths[idx]]

        elif type(idx) is slice:
            # TODO: Should we have a CompactListView class that would be
            #       returned when slicing?
            compact_list = CompactList()
            compact_list._data = self._data
            compact_list._offsets = self._offsets[idx]
            compact_list._lengths = self._lengths[idx]
            return compact_list

        elif type(idx) is list:
            # TODO: Should we have a CompactListView class that would be
            #       returned when doing advance indexing?
            compact_list = CompactList()
            compact_list._data = self._data
            compact_list._offsets = [self._offsets[i] for i in idx]
            compact_list._lengths = [self._lengths[i] for i in idx]
            return compact_list

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

    def __iter__(self):
        if len(self._lengths) != len(self._offsets):
            raise ValueError("CompactList object corrupted: len(self._lengths) != len(self._offsets)")

        for offset, lengths in zip(self._offsets, self._lengths):
            yield self._data[offset: offset+lengths]

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        return repr(list(self))


class TractogramItem(object):
    ''' Class containing information about one streamline.

    ``TractogramItem`` objects have three main properties: `streamline`, `scalars`
    and ``properties``.

    Parameters
    ----------
    streamline : ndarray of shape (N, 3)
        Points of this streamline represented as an ndarray of shape (N, 3)
        where N is the number of points.

    scalars : ndarray of shape (N, M)
        Scalars associated with each point of this streamline and represented
        as an ndarray of shape (N, M) where N is the number of points and
        M is the number of scalars (excluding the three coordinates).

    properties : ndarray of shape (P,)
        Properties associated with this streamline and represented as an
        ndarray of shape (P,) where P is the number of properties.
    '''
    def __init__(self, streamline, scalars=None, properties=None):
        #if scalars is not None and len(streamline) != len(scalars):
        #    raise ValueError("First dimension of streamline and scalars must match.")

        self.streamline = np.asarray(streamline)
        self.scalars = np.asarray([] if scalars is None else scalars)
        self.properties = np.asarray([] if properties is None else properties)

    def __iter__(self):
        return iter(self.streamline)

    def __len__(self):
        return len(self.streamline)


class Tractogram(object):
    ''' Class containing information about streamlines.

    Tractogram objects have three main properties: ``streamlines``, ``scalars``
    and ``properties``. Tractogram objects can be iterate over producing
    tuple of ``streamlines``, ``scalars`` and ``properties`` for each streamline.

    Parameters
    ----------
    streamlines : list of ndarray of shape (Nt, 3)
        Sequence of T streamlines. One streamline is an ndarray of shape (Nt, 3)
        where Nt is the number of points of streamline t.

    scalars : list of ndarray of shape (Nt, M)
        Sequence of T ndarrays of shape (Nt, M) where T is the number of
        streamlines defined by ``streamlines``, Nt is the number of points
        for a particular streamline t and M is the number of scalars
        associated to each point (excluding the three coordinates).

    properties : list of ndarray of shape (P,)
        Sequence of T ndarrays of shape (P,) where T is the number of
        streamlines defined by ``streamlines``, P is the number of properties
        associated to each streamline.
    '''
    def __init__(self, streamlines=None, scalars=None, properties=None):
        self._header = TractogramHeader()
        self.streamlines = streamlines
        self.scalars = scalars
        self.properties = properties

    @classmethod
    def create_from_generator(cls, gen):
        BUFFER_SIZE = 10000000  # About 128 Mb if item shape is 3.

        streamlines = CompactList()
        scalars = CompactList()
        properties = np.array([])

        gen = iter(gen)
        try:
            first_element = next(gen)
            gen = itertools.chain([first_element], gen)
        except StopIteration:
            return cls(streamlines, scalars, properties)

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

        return cls(streamlines, scalars, properties)

    @property
    def header(self):
        return self._header

    @property
    def streamlines(self):
        return self._streamlines

    @streamlines.setter
    def streamlines(self, value):
        self._streamlines = value
        if not isinstance(value, CompactList):
            self._streamlines = CompactList(value)

        self.header.nb_streamlines = len(self.streamlines)

    @property
    def scalars(self):
        return self._scalars

    @scalars.setter
    def scalars(self, value):
        self._scalars = value
        if not isinstance(value, CompactList):
            self._scalars = CompactList(value)

        self.header.nb_scalars_per_point = 0
        if len(self.scalars) > 0 and len(self.scalars[0]) > 0:
            self.header.nb_scalars_per_point = len(self.scalars[0][0])

    @property
    def properties(self):
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = np.asarray(value)
        if value is None:
            self._properties = np.empty((len(self), 0), dtype=np.float32)

        self.header.nb_properties_per_streamline = 0
        if len(self.properties) > 0:
            self.header.nb_properties_per_streamline = len(self.properties[0])

    def __iter__(self):
        for data in zip_longest(self.streamlines, self.scalars, self.properties, fillvalue=None):
            yield TractogramItem(*data)

    def __getitem__(self, idx):
        pts = self.streamlines[idx]
        scalars = []
        if len(self.scalars) > 0:
            scalars = self.scalars[idx]

        properties = []
        if len(self.properties) > 0:
            properties = self.properties[idx]

        if type(idx) is slice:
            return Tractogram(pts, scalars, properties)

        return TractogramItem(pts, scalars, properties)

    def __len__(self):
        return len(self.streamlines)

    def copy(self):
        """ Returns a copy of this `Tractogram` object. """
        streamlines = Tractogram(self.streamlines.copy(), self.scalars.copy(), self.properties.copy())
        streamlines._header = self.header.copy()
        return streamlines

    def apply_affine(self, affine):
        """ Applies an affine transformation on the points of each streamline.

        This is performed in-place.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation that will be applied on each streamline.
        """
        if len(self.streamlines) == 0:
            return

        BUFFER_SIZE = 10000000  # About 128 Mb since pts shape is 3.
        for i in range(0, len(self.streamlines._data), BUFFER_SIZE):
            pts = self.streamlines._data[i:i+BUFFER_SIZE]
            self.streamlines._data[i:i+BUFFER_SIZE] = apply_affine(affine, pts)


class LazyTractogram(Tractogram):
    ''' Class containing information about streamlines.

    Tractogram objects have four main properties: ``header``, ``streamlines``,
    ``scalars`` and ``properties``. Tractogram objects are iterable and
    produce tuple of ``streamlines``, ``scalars`` and ``properties`` for each
    streamline.

    Parameters
    ----------
    streamlines_func : coroutine ouputting (Nt,3) array-like (optional)
        Function yielding streamlines. One streamline is
        an ndarray of shape (Nt,3) where Nt is the number of points of
        streamline t.

    scalars_func : coroutine ouputting (Nt,M) array-like (optional)
        Function yielding scalars for a particular streamline t. The scalars
        are represented as an ndarray of shape (Nt,M) where Nt is the number
        of points of that streamline t and M is the number of scalars
        associated to each point (excluding the three coordinates).

    properties_func : coroutine ouputting (P,) array-like (optional)
        Function yielding properties for a particular streamline t. The
        properties are represented as an ndarray of shape (P,) where P is
        the number of properties associated to each streamline.

    getitem_func : function `idx -> 3-tuples` (optional)
        Function returning a subset of the tractogram given an index or a
        slice (i.e. the __getitem__ function to use).

    Notes
    -----
    If provided, ``scalars`` and ``properties`` must yield the same number of
    values as ``streamlines``.
    '''
    def __init__(self, streamlines_func=lambda:[], scalars_func=lambda: [], properties_func=lambda: [], getitem_func=None):
        super(LazyTractogram, self).__init__(streamlines_func, scalars_func, properties_func)
        self._data = lambda: zip_longest(self.streamlines, self.scalars, self.properties, fillvalue=[])
        self._getitem = getitem_func

    @classmethod
    def create_from_data(cls, data_func):
        ''' Saves streamlines to a file-like object.

        Parameters
        ----------
        data_func : coroutine ouputting tuple (optional)
            Function yielding 3-tuples, (streamlines, scalars, properties).
            Streamlines are represented as an ndarray of shape (Nt,3), scalars
            as an ndarray of shape (Nt,M) and properties as an ndarray of shape
            (P,) where Nt is the number of points for a particular
            streamline t, M is the number of scalars associated to each point
            (excluding the three coordinates) and P is the number of properties
            associated to each streamline.
        '''
        if not callable(data_func):
            raise TypeError("`data` must be a coroutine.")

        lazy_streamlines = cls()
        lazy_streamlines._data = data_func
        lazy_streamlines.streamlines = lambda: (x[0] for x in data_func())
        lazy_streamlines.scalars = lambda: (x[1] for x in data_func())
        lazy_streamlines.properties = lambda: (x[2] for x in data_func())
        return lazy_streamlines

    @property
    def streamlines(self):
        return self._streamlines()

    @streamlines.setter
    def streamlines(self, value):
        if not callable(value):
            raise TypeError("`streamlines` must be a coroutine.")

        self._streamlines = value

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
            raise AttributeError('`LazyTractogram` does not support indexing.')

        return self._getitem(idx)

    def __iter__(self):
        i = 0
        for i, s in enumerate(self._data(), start=1):
            yield TractogramItem(*s)

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
                 " `LazyTractogram` object.", UsageWarning)
            return sum(1 for _ in self)

        return self.header.nb_streamlines

    def copy(self):
        """ Returns a copy of this `LazyTractogram` object. """
        streamlines = LazyTractogram(self._streamlines, self._scalars, self._properties)
        streamlines._header = self.header.copy()
        return streamlines

    def transform(self, affine):
        """ Applies an affine transformation on the streamlines.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation that will be applied on each streamline.

        Returns
        -------
        streamlines : `LazyTractogram` object
            Tractogram living in a space defined by `affine`.
        """
        return super(LazyTractogram, self).transform(affine, lazy=True)


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class TractogramFile(object):
    ''' Convenience class to encapsulate tractogram file format. '''
    __metaclass__ = ABCMeta

    def __init__(self, tractogram, header=None):
        self._tractogram = tractogram
        self._header = TractogramHeader() if header is None else header

    @property
    def tractogram(self):
        return self._tractogram

    @property
    def streamlines(self):
        return self.tractogram.streamlines

    @property
    def scalars(self):
        return self.tractogram.scalars

    @property
    def properties(self):
        return self.tractogram.properties

    @property
    def header(self):
        return self._header

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

    @abstractclassmethod
    def load(cls, fileobj, lazy_load=True):
        ''' Loads streamlines from a file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).

        lazy_load : boolean (optional)
            Load streamlines in a lazy manner i.e. they will not be kept
            in memory. For postprocessing speed, turn off this option.

        Returns
        -------
        tractogram_file : ``TractogramFile`` object
            Returns an object containing tractogram data and header
            information.
        '''
        raise NotImplementedError()

    @abstractmethod
    def save(self, fileobj):
        ''' Saves streamlines to a file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            opened and ready to write.
        '''
        raise NotImplementedError()

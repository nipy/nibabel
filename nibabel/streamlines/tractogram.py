import numpy as np
import collections
from warnings import warn

from nibabel.affines import apply_affine

from .compact_list import CompactList


class UsageWarning(Warning):
    pass


class TractogramItem(object):
    """ Class containing information about one streamline.

    ``TractogramItem`` objects have three main properties: `streamline`,
    `data_for_streamline`, and `data_for_points`.

    Parameters
    ----------
    streamline : ndarray of shape (N, 3)
        Points of this streamline represented as an ndarray of shape (N, 3)
        where N is the number of points.

    data_for_streamline : dict

    data_for_points : dict
    """
    def __init__(self, streamline, data_for_streamline, data_for_points):
        self.streamline = np.asarray(streamline)
        self.data_for_streamline = data_for_streamline
        self.data_for_points = data_for_points

    def __iter__(self):
        return iter(self.streamline)

    def __len__(self):
        return len(self.streamline)


class Tractogram(object):
    """ Class containing information about streamlines.

    Tractogram objects have three main properties: ``streamlines``

    Attributes
    ----------
    affine_to_rasmm : 2D array (4,4)
        Affine that brings the streamlines back to *RAS+* and *mm* space
        where coordinate (0,0,0) refers to the center of the voxel.

    """
    class DataDict(collections.MutableMapping):
        def __init__(self, tractogram, *args, **kwargs):
            self.tractogram = tractogram
            self.store = dict()

            # Use update to set the keys.
            if len(args) == 1:
                if isinstance(args[0], Tractogram.DataDict):
                    self.update(dict(args[0].store.items()))
                elif args[0] is None:
                    return
                else:
                    self.update(dict(*args, **kwargs))
            else:
                self.update(dict(*args, **kwargs))

        def __getitem__(self, key):
            return self.store[key]

        def __delitem__(self, key):
            del self.store[key]

        def __iter__(self):
            return iter(self.store)

        def __len__(self):
            return len(self.store)

    class DataPerStreamlineDict(DataDict):
        """ Internal dictionary that makes sure data are 2D ndarray. """

        def __setitem__(self, key, value):
            value = np.asarray(value)

            if value.ndim == 1 and value.dtype != object:
                # Reshape without copy
                value.shape = ((len(value), 1))

            if value.ndim != 2:
                raise ValueError("data_per_streamline must be a 2D ndarray.")

            # We make sure there is the right amount of values
            # (i.e. same as the number of streamlines in the tractogram).
            if len(value) != len(self.tractogram):
                msg = ("The number of values ({0}) should match the number of"
                       " streamlines ({1}).")
                raise ValueError(msg.format(len(value), len(self.tractogram)))

            self.store[key] = value

    class DataPerPointDict(DataDict):
        """ Internal dictionary that makes sure data are `CompactList`. """

        def __setitem__(self, key, value):
            value = CompactList(value)

            # We make sure we have the right amount of values (i.e. same as
            # the total number of points of all streamlines in the tractogram).
            if len(value._data) != len(self.tractogram.streamlines._data):
                msg = ("The number of values ({0}) should match the total"
                       " number of points of all streamlines ({1}).")
                nb_streamlines_points = self.tractogram.streamlines._data
                raise ValueError(msg.format(len(value._data),
                                            len(nb_streamlines_points)))

            self.store[key] = value

    def __init__(self, streamlines=None,
                 data_per_streamline=None,
                 data_per_point=None):
        """
        Parameters
        ----------
        streamlines : list of ndarray of shape (Nt, 3) (optional)
            Sequence of T streamlines. One streamline is an ndarray of
            shape (Nt, 3) where Nt is the number of points of streamline t.

        data_per_streamline : dict of list of ndarray of shape (P,) (optional)
            Sequence of T ndarrays of shape (P,) where T is the number of
            streamlines defined by ``streamlines``, P is the number of
            properties associated to each streamline.

        data_per_point : dict of list of ndarray of shape (Nt, M) (optional)
            Sequence of T ndarrays of shape (Nt, M) where T is the number
            of streamlines defined by ``streamlines``, Nt is the number of
            points for a particular streamline t and M is the number of
            scalars associated to each point (excluding the three
            coordinates).

        """
        self.streamlines = streamlines
        self.data_per_streamline = data_per_streamline
        self.data_per_point = data_per_point
        self._affine_to_rasmm = np.eye(4)

    @property
    def streamlines(self):
        return self._streamlines

    @streamlines.setter
    def streamlines(self, value):
        self._streamlines = CompactList(value)

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = Tractogram.DataPerStreamlineDict(self,
                                                                     value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = Tractogram.DataPerPointDict(self, value)

    @property
    def affine_to_rasmm(self):
        # Return a copy. User should use self.apply_affine` to modify it.
        return self._affine_to_rasmm.copy()

    def __iter__(self):
        for i in range(len(self.streamlines)):
            yield self[i]

    def __getitem__(self, idx):
        pts = self.streamlines[idx]

        data_per_streamline = {}
        for key in self.data_per_streamline:
            data_per_streamline[key] = self.data_per_streamline[key][idx]

        data_per_point = {}
        for key in self.data_per_point:
            data_per_point[key] = self.data_per_point[key][idx]

        if isinstance(idx, int) or isinstance(idx, np.integer):
            return TractogramItem(pts, data_per_streamline, data_per_point)

        return Tractogram(pts, data_per_streamline, data_per_point)

    def __len__(self):
        return len(self.streamlines)

    def copy(self):
        """ Returns a copy of this `Tractogram` object. """
        data_per_streamline = {}
        for key in self.data_per_streamline:
            data_per_streamline[key] = self.data_per_streamline[key].copy()

        data_per_point = {}
        for key in self.data_per_point:
                data_per_point[key] = self.data_per_point[key].copy()

        tractogram = Tractogram(self.streamlines.copy(),
                                data_per_streamline,
                                data_per_point)

        tractogram._affine_to_rasmm = self.affine_to_rasmm
        return tractogram

    def apply_affine(self, affine, lazy=False):
        """ Applies an affine transformation on the points of each streamline.

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation that will be applied to every streamline.

        Returns
        -------
        tractogram : ``Tractogram`` or ``LazyTractogram`` object
            Tractogram where the streamlines have been transformed according
            to the given affine transformation. If the `lazy` option is true,
            it returns a ``LazyTractogram`` object, otherwise it returns a
            reference to this ``Tractogram`` object with updated streamlines.

        """
        if lazy:
            lazy_tractogram = LazyTractogram.from_tractogram(self)
            lazy_tractogram.apply_affine(affine)
            return lazy_tractogram

        if len(self.streamlines) == 0:
            return self

        BUFFER_SIZE = 10000000  # About 128 Mb since pts shape is 3.
        for i in range(0, len(self.streamlines._data), BUFFER_SIZE):
            pts = self.streamlines._data[i:i+BUFFER_SIZE]
            self.streamlines._data[i:i+BUFFER_SIZE] = apply_affine(affine, pts)

        # Update the affine that brings back the streamlines to RASmm.
        self._affine_to_rasmm = np.dot(self._affine_to_rasmm,
                                       np.linalg.inv(affine))

        return self


class LazyTractogram(Tractogram):
    ''' Class containing information about streamlines.

    Tractogram objects have four main properties: ``header``, ``streamlines``,
    ``scalars`` and ``properties``. Tractogram objects are iterable and
    produce tuple of ``streamlines``, ``scalars`` and ``properties`` for each
    streamline.

    Notes
    -----
    If provided, ``scalars`` and ``properties`` must yield the same number of
    values as ``streamlines``.
    '''

    class LazyDict(collections.MutableMapping):
        """ Internal dictionary with lazy evaluations. """

        def __init__(self, *args, **kwargs):
            self.store = dict()

            # Use update to set keys.
            if len(args) == 1 and isinstance(args[0], LazyTractogram.LazyDict):
                self.update(dict(args[0].store.items()))
            else:
                self.update(dict(*args, **kwargs))

        def __getitem__(self, key):
            return self.store[key]()

        def __setitem__(self, key, value):
            if value is not None and not callable(value):
                raise TypeError("`value` must be a coroutine or None.")

            self.store[key] = value

        def __delitem__(self, key):
            del self.store[key]

        def __iter__(self):
            return iter(self.store)

        def __len__(self):
            return len(self.store)

    def __init__(self, streamlines=None,
                 data_per_streamline=None,
                 data_per_point=None):
        """
        Parameters
        ----------
        streamlines : coroutine yielding ndarrays of shape (Nt,3) (optional)
            Function yielding streamlines. One streamline is an ndarray of
            shape (Nt,3) where Nt is the number of points of streamline t.

        data_per_streamline : dict of coroutines yielding ndarrays of shape (P,) (optional)
            Function yielding properties for a particular streamline t. The
            properties are represented as an ndarray of shape (P,) where P is
            the number of properties associated to each streamline.

        data_per_point : dict of coroutines yielding ndarrays of shape (Nt,M) (optional)
            Function yielding scalars for a particular streamline t. The
            scalars are represented as an ndarray of shape (Nt,M) where Nt
            is the number of points of that streamline t and M is the number
            of scalars associated to each point (excluding the three
            coordinates).

        """
        super(LazyTractogram, self).__init__(streamlines,
                                             data_per_streamline,
                                             data_per_point)
        self._nb_streamlines = None
        self._data = None
        self._affine_to_apply = np.eye(4)

    @classmethod
    def from_tractogram(cls, tractogram):
        ''' Creates a ``LazyTractogram`` object from a ``Tractogram`` object.

        Parameters
        ----------
        tractogram : ``Tractgogram`` object
            Tractogram from which to create a ``LazyTractogram`` object.

        Returns
        -------
        lazy_tractogram : ``LazyTractogram`` object
            New lazy tractogram.

        '''
        data_per_streamline = {}
        for key, value in tractogram.data_per_streamline.items():
            data_per_streamline[key] = lambda: value

        data_per_point = {}
        for key, value in tractogram.data_per_point.items():
                data_per_point[key] = lambda: value

        lazy_tractogram = cls(lambda: tractogram.streamlines.copy(),
                              data_per_streamline,
                              data_per_point)

        lazy_tractogram._nb_streamlines = len(tractogram)
        lazy_tractogram._affine_to_rasmm = tractogram.affine_to_rasmm
        return lazy_tractogram

    @classmethod
    def create_from(cls, data_func):
        ''' Creates a ``LazyTractogram`` from a coroutine yielding
        ``TractogramItem`` objects.

        Parameters
        ----------
        data_func : coroutine yielding ``TractogramItem`` objects
            A function that whenever it is called starts yielding
            ``TractogramItem`` objects that should be part of this
            LazyTractogram.

        Returns
        -------
        lazy_tractogram : ``LazyTractogram`` object
            New lazy tractogram.

        '''
        if not callable(data_func):
            raise TypeError("`data_func` must be a coroutine.")

        lazy_tractogram = cls()
        lazy_tractogram._data = data_func

        try:
            first_item = next(data_func())

            # Set data_per_streamline using data_func
            def _gen(key):
                return lambda: (t.data_for_streamline[key] for t in data_func())

            data_per_streamline_keys = first_item.data_for_streamline.keys()
            for k in data_per_streamline_keys:
                lazy_tractogram._data_per_streamline[k] = _gen(k)

            # Set data_per_point using data_func
            def _gen(key):
                return lambda: (t.data_for_points[key] for t in data_func())

            data_per_point_keys = first_item.data_for_points.keys()
            for k in data_per_point_keys:
                lazy_tractogram._data_per_point[k] = _gen(k)

        except StopIteration:
            pass

        return lazy_tractogram

    @property
    def streamlines(self):
        streamlines_gen = iter([])
        if self._streamlines is not None:
            streamlines_gen = self._streamlines()
        elif self._data is not None:
            streamlines_gen = (t.streamline for t in self._data())

        # Check if we need to apply an affine.
        if not np.all(self._affine_to_apply == np.eye(4)):
            def _apply_affine():
                for s in streamlines_gen:
                    yield apply_affine(self._affine_to_apply, s)

            return _apply_affine()

        return streamlines_gen

    @streamlines.setter
    def streamlines(self, value):
        if value is not None and not callable(value):
            raise TypeError("`streamlines` must be a coroutine.")

        self._streamlines = value

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        if value is None:
            value = {}

        self._data_per_streamline = LazyTractogram.LazyDict(value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        if value is None:
            value = {}

        self._data_per_point = LazyTractogram.LazyDict(value)

    @property
    def data(self):
        if self._data is not None:
            return self._data()

        def _gen_data():
            data_per_streamline_generators = {}
            for k, v in self.data_per_streamline.items():
                data_per_streamline_generators[k] = iter(v)

            data_per_point_generators = {}
            for k, v in self.data_per_point.items():
                data_per_point_generators[k] = iter(v)

            for s in self.streamlines:
                data_for_streamline = {}
                for k, v in data_per_streamline_generators.items():
                    data_for_streamline[k] = next(v)

                data_for_points = {}
                for k, v in data_per_point_generators.items():
                    data_for_points[k] = next(v)

                yield TractogramItem(s, data_for_streamline, data_for_points)

        return _gen_data()

    def __getitem__(self, idx):
        raise NotImplementedError('`LazyTractogram` does not support indexing.')

    def __iter__(self):
        i = 0
        for i, tractogram_item in enumerate(self.data, start=1):
            yield tractogram_item

        # Keep how many streamlines there are in this tractogram.
        self._nb_streamlines = i

    def __len__(self):
        # Check if we know how many streamlines there are.
        if self._nb_streamlines is None:
            warn("Number of streamlines will be determined manually by looping"
                 " through the streamlines. If you know the actual number of"
                 " streamlines, you might want to set it beforehand via"
                 " `self.header.nb_streamlines`."
                 " Note this will consume any generators used to create this"
                 " `LazyTractogram` object.", UsageWarning)
            # Count the number of streamlines.
            self._nb_streamlines = sum(1 for _ in self.streamlines)

        return self._nb_streamlines

    def copy(self):
        """ Returns a copy of this `LazyTractogram` object. """
        tractogram = LazyTractogram(self._streamlines,
                                    self._data_per_streamline,
                                    self._data_per_point)
        tractogram._nb_streamlines = self._nb_streamlines
        tractogram._data = self._data
        tractogram._affine_to_apply = self._affine_to_apply.copy()
        return tractogram

    def apply_affine(self, affine):
        """ Applies an affine transformation to the streamlines.

        The transformation will be applied just before returning the
        streamlines.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation that will be applied on each streamline.

        Returns
        -------
        lazy_tractogram : ``LazyTractogram`` object
            Reference to this instance of ``LazyTractogram``.

        """
        # Update the affine that will be applied when returning streamlines.
        self._affine_to_apply = np.dot(affine, self._affine_to_apply)

        # Update the affine that brings back the streamlines to RASmm.
        self._affine_to_rasmm = np.dot(self._affine_to_rasmm,
                                       np.linalg.inv(affine))
        return self

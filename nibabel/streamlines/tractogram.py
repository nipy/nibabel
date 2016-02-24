import copy
import numbers
import numpy as np
import collections
from warnings import warn

from nibabel.affines import apply_affine

from .array_sequence import ArraySequence


def is_data_dict(obj):
    """ Tells if obj is a :class:`DataDict`. """
    return hasattr(obj, 'store')


def is_lazy_dict(obj):
    """ Tells if obj is a :class:`LazyDict`. """
    return is_data_dict(obj) and callable(obj.store.values()[0])


class DataDict(collections.MutableMapping):
    """ Dictionary that makes sure data are 2D array.

    This container behaves like a standard dictionary but it makes sure its
    elements are ndarrays. In addition, it makes sure the amount of data
    contained in those ndarrays matches the number of streamlines of the
    :class:`Tractogram` object provided at the instantiation of this
    dictionary.
    """
    def __init__(self, tractogram, *args, **kwargs):
        self.tractogram = tractogram
        self.store = dict()

        # Use update to set the keys.
        if len(args) == 1:
            if isinstance(args[0], DataDict):
                self.update(**args[0])
            elif args[0] is None:
                return
            else:
                self.update(dict(*args, **kwargs))
        else:
            self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        try:
            return self.store[key]
        except KeyError:
            pass  # Maybe it is an integer.
        except TypeError:
            pass  # Maybe it is an object for advanced indexing.

        # Try to interpret key as an index/slice in which case we
        # perform (advanced) indexing on every element of the dictionnary.
        try:
            idx = key
            new_dict = type(self)(None)
            for k, v in self.items():
                new_dict[k] = v[idx]

            return new_dict
        except TypeError:
            pass

        # That means key was not an index/slice after all.
        return self.store[key]  # Will raise the proper error.

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class DataPerStreamlineDict(DataDict):
    """ Dictionary that makes sure data are 2D array.

    This container behaves like a standard dictionary but it makes sure its
    elements are ndarrays. In addition, it makes sure the amount of data
    contained in those ndarrays matches the number of streamlines of the
    :class:`Tractogram` object provided at the instantiation of this
    dictionary.
    """
    def __setitem__(self, key, value):
        value = np.asarray(list(value))

        if value.ndim == 1 and value.dtype != object:
            # Reshape without copy
            value.shape = ((len(value), 1))

        if value.ndim != 2:
            raise ValueError("data_per_streamline must be a 2D array.")

        # We make sure there is the right amount of values
        # (i.e. same as the number of streamlines in the tractogram).
        if self.tractogram is not None and len(value) != len(self.tractogram):
            msg = ("The number of values ({0}) should match the number of"
                   " streamlines ({1}).")
            raise ValueError(msg.format(len(value), len(self.tractogram)))

        self.store[key] = value


class DataPerPointDict(DataDict):
    """ Dictionary making sure data are :class:`ArraySequence` objects.

    This container behaves like a standard dictionary but it makes sure its
    elements are :class:`ArraySequence` objects. In addition, it makes sure
    the amount of data contained in those :class:`ArraySequence` objects
    matches the the number of points of the :class:`Tractogram` object
    provided at the instantiation of this dictionary.
    """

    def __setitem__(self, key, value):
        value = ArraySequence(value)

        # We make sure we have the right amount of values (i.e. same as
        # the total number of points of all streamlines in the tractogram).
        if (self.tractogram is not None and
                len(value._data) != len(self.tractogram.streamlines._data)):
            msg = ("The number of values ({0}) should match the total"
                   " number of points of all streamlines ({1}).")
            nb_streamlines_points = self.tractogram.streamlines._data
            raise ValueError(msg.format(len(value._data),
                                        len(nb_streamlines_points)))

        self.store[key] = value


class LazyDict(DataDict):
    """ Dictionary of generator functions.

    This container behaves like an dictionary but it makes sure its elements
    are callable objects and assumed to be generator function yielding values.
    When getting the element associated to a given key, the element (i.e. a
    generator function) is first called before being returned.
    """
    def __init__(self, tractogram, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], LazyDict):
            # Copy the generator functions.
            self.tractogram = tractogram
            self.store = dict()
            self.update(**args[0].store)
            return

        super(LazyDict, self).__init__(tractogram, *args, **kwargs)

    def __getitem__(self, key):
        return self.store[key]()

    def __setitem__(self, key, value):
        if value is not None and not callable(value):
            raise TypeError("`value` must be a generator function or None.")

        self.store[key] = value


class TractogramItem(object):
    """ Class containing information about one streamline.

    :class:`TractogramItem` objects have three main properties: `streamline`,
    `data_for_streamline`, and `data_for_points`.

    Parameters
    ----------
    streamline : ndarray shape (N, 3)
        Points of this streamline represented as an ndarray of shape (N, 3)
        where N is the number of points.
    data_for_streamline : dict
        Dictionary containing some data associated to this particular
        streamline. Each key `k` is mapped to a ndarray of shape (Pk,), where
        `Pt` is the dimension of the data associated with key `k`.
    data_for_points : dict
        Dictionary containing some data associated to each point of this
        particular streamline. Each key `k` is mapped to a ndarray of
        shape (Nt, Mk), where `Nt` is the number of points of this streamline
        and `Mk` is the dimension of the data associated with key `k`.
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

    Tractogram objects have three main properties: `streamlines`,
    `data_per_streamline` and `data_per_point`.

    Streamlines of a tractogram can be in any coordinate system of your
    choice as long as you provide the correct `affine_to_rasmm` matrix, at
    construction time, that brings the streamlines back to *RAS+*, *mm* space,
    where the coordinates (0,0,0) corresponds to the center of the voxel
    (opposed to a corner).

    """
    def __init__(self, streamlines=None,
                 data_per_streamline=None,
                 data_per_point=None,
                 affine_to_rasmm=np.eye(4)):
        """
        Parameters
        ----------
        streamlines : list of ndarray of shape (Nt, 3) (optional)
            Sequence of T streamlines. One streamline is an ndarray of
            shape (Nt, 3) where Nt is the number of points of streamline t.
        data_per_streamline : dict of list of ndarray of shape (P,) (optional)
            Sequence of T ndarrays of shape (P,) where T is the number of
            streamlines defined by `streamlines`, P is the number of
            properties associated to each streamline.
        data_per_point : dict of list of ndarray of shape (Nt, M) (optional)
            Sequence of T ndarrays of shape (Nt, M) where T is the number
            of streamlines defined by `streamlines`, Nt is the number of
            points for a particular streamline t and M is the number of
            scalars associated to each point (excluding the three
            coordinates).
        affine_to_rasmm : ndarray of shape (4, 4)
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel.
        """
        self.streamlines = streamlines
        self.data_per_streamline = data_per_streamline
        self.data_per_point = data_per_point
        self._affine_to_rasmm = affine_to_rasmm

    @property
    def streamlines(self):
        return self._streamlines

    @streamlines.setter
    def streamlines(self, value):
        self._streamlines = ArraySequence(value)

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = DataPerStreamlineDict(self, value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = DataPerPointDict(self, value)

    def get_affine_to_rasmm(self):
        """ Returns the affine bringing this tractogram to RAS+mm. """
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

        if isinstance(idx, (numbers.Integral, np.integer)):
            return TractogramItem(pts, data_per_streamline, data_per_point)

        return Tractogram(pts, data_per_streamline, data_per_point)

    def __len__(self):
        return len(self.streamlines)

    def copy(self):
        """ Returns a copy of this :class:`Tractogram` object. """
        return copy.deepcopy(self)

    def apply_affine(self, affine, lazy=False):
        """ Applies an affine transformation on the points of each streamline.

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        affine : ndarray of shape (4, 4)
            Transformation that will be applied to every streamline.
        lazy_load : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been transformed according
            to the given affine transformation. If the `lazy` option is true,
            it returns a :class:`LazyTractogram` object, otherwise it returns a
            reference to this :class:`Tractogram` object with updated
            streamlines.
        """
        if lazy:
            lazy_tractogram = LazyTractogram.from_tractogram(self)
            lazy_tractogram.apply_affine(affine)
            return lazy_tractogram

        if len(self.streamlines) == 0:
            return self

        BUFFER_SIZE = 10000000  # About 128 Mb since pts shape is 3.
        for start in range(0, len(self.streamlines._data), BUFFER_SIZE):
            end = start + BUFFER_SIZE
            pts = self.streamlines._data[start:end]
            self.streamlines._data[start:end] = apply_affine(affine, pts)

        # Update the affine that brings back the streamlines to RASmm.
        self._affine_to_rasmm = np.dot(self._affine_to_rasmm,
                                       np.linalg.inv(affine))

        return self

    def to_world(self, lazy=False):
        """ Brings the streamlines to world space (i.e. RAS+ and mm).

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        lazy_load : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been sent to world space.
            If the `lazy` option is true, it returns a :class:`LazyTractogram`
            object, otherwise it returns a reference to this
            :class:`Tractogram` object with updated streamlines.
        """
        return self.apply_affine(self._affine_to_rasmm, lazy=lazy)


class LazyTractogram(Tractogram):
    """ Class containing information about streamlines.

    Tractogram objects have four main properties: `header`, `streamlines`,
    `scalars` and `properties`. Tractogram objects are iterable and
    produce tuple of `streamlines`, `scalars` and `properties` for each
    streamline.

    Notes
    -----
    If provided, `scalars` and `properties` must yield the same number of
    values as `streamlines`.
    """
    def __init__(self, streamlines=None,
                 data_per_streamline=None,
                 data_per_point=None):
        """
        Parameters
        ----------
        streamlines : generator function yielding, optional
            Generator function yielding streamlines. One streamline is an
            ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept along side every
            streamline, and its associated value is a generator function
            yielding that information via ndarrays of shape ($P_i$,) where
            $P_i$ is the number scalar values to store for that particular
            information $i$.
        data_per_point : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept along side every
            point of every streamline, and its associated value is a generator
            function yielding that information via ndarrays of shape
            ($N_t$, $M_i$) where $N_t$ is the number of points for a particular
            streamline $t$ and $M_i$ is the number scalar values to store for
            that particular information $i$.
        """
        super(LazyTractogram, self).__init__(streamlines,
                                             data_per_streamline,
                                             data_per_point)
        self._nb_streamlines = None
        self._data = None
        self._affine_to_apply = np.eye(4)

    @classmethod
    def from_tractogram(cls, tractogram):
        """ Creates a :class:`LazyTractogram` object from a :class:`Tractogram` object.

        Parameters
        ----------
        tractogram : :class:`Tractgogram` object
            Tractogram from which to create a :class:`LazyTractogram` object.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        lazy_tractogram = cls(lambda: tractogram.streamlines.copy())

        # Set data_per_streamline using data_func
        def _gen(key):
            return lambda: iter(tractogram.data_per_streamline[key])

        for k in tractogram.data_per_streamline:
            lazy_tractogram._data_per_streamline[k] = _gen(k)

        # Set data_per_point using data_func
        def _gen(key):
            return lambda: iter(tractogram.data_per_point[key])

        for k in tractogram.data_per_point:
            lazy_tractogram._data_per_point[k] = _gen(k)

        lazy_tractogram._nb_streamlines = len(tractogram)
        lazy_tractogram._affine_to_rasmm = tractogram.get_affine_to_rasmm()
        return lazy_tractogram

    @classmethod
    def create_from(cls, data_func):
        """ Creates an instance from a generator function.

        The generator function must yield :class:`TractogramItem` objects.

        Parameters
        ----------
        data_func : generator function yielding :class:`TractogramItem` objects
            Generator function that whenever it is called starts yielding
            :class:`TractogramItem` objects that will be used to instantiate a
            :class:`LazyTractogram`.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        if not callable(data_func):
            raise TypeError("`data_func` must be a generator function.")

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
            raise TypeError("`streamlines` must be a generator function.")

        self._streamlines = value

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = LazyDict(self, value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = LazyDict(self, value)

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
                 " `LazyTractogram` object.", Warning)
            # Count the number of streamlines.
            self._nb_streamlines = sum(1 for _ in self.streamlines)

        return self._nb_streamlines

    def copy(self):
        """ Returns a copy of this :class:`LazyTractogram` object. """
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
        lazy_tractogram : :class:`LazyTractogram` object
            Reference to this instance of :class:`LazyTractogram`.
        """
        # Update the affine that will be applied when returning streamlines.
        self._affine_to_apply = np.dot(affine, self._affine_to_apply)

        # Update the affine that brings back the streamlines to RASmm.
        self._affine_to_rasmm = np.dot(self._affine_to_rasmm,
                                       np.linalg.inv(affine))
        return self

    def to_world(self):
        """ Brings the streamlines to world space (i.e. RAS+ and mm).

        The transformation will be applied just before returning the
        streamlines.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            Reference to this instance of :class:`LazyTractogram`.
        """
        return self.apply_affine(self._affine_to_rasmm)

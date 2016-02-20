import numpy as np


def is_array_sequence(obj):
    """ Return True if `obj` is an array sequence. """
    try:
        return obj.is_array_sequence
    except AttributeError:
        return False


class ArraySequence(object):
    """ Sequence of ndarrays having variable first dimension sizes.

    This is a container allowing to store multiple ndarrays where each ndarray
    might have different first dimension size but a *common* size for the
    remaining dimensions.

    More generally, an instance of :class:`ArraySequence` of length $N$ is
    composed of $N$ ndarrays of shape $(d_1, d_2, ... d_D)$ where $d_1$
    can vary in length between arrays but $(d_2, ..., d_D)$ have to be the
    same for every ndarray.
    """

    BUFFER_SIZE = 87382 * 4  # About 4 Mb if item shape is 3 (e.g. 3D points).

    def __init__(self, iterable=None):
        """
        Parameters
        ----------
        iterable : None or iterable or :class:`ArraySequence`, optional
            If None, create an empty :class:`ArraySequence` object.
            If iterable, create a :class:`ArraySequence` object initialized
            from array-like objects yielded by the iterable.
            If :class:`ArraySequence`, create a view (no memory is allocated).
            For an actual copy use :meth:`.copy` instead.
        """
        # Create new empty `ArraySequence` object.
        self._is_view = False
        self._data = np.array(0)
        self._offsets = np.array([], dtype=np.intp)
        self._lengths = np.array([], dtype=np.intp)

        if iterable is None:
            return

        if is_array_sequence(iterable):
            # Create a view.
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths
            self._is_view = True
            return

        # Add elements of the iterable.
        offsets = []
        lengths = []
        # Initialize the `ArraySequence` object from iterable's item.
        offset = 0
        for i, e in enumerate(iterable):
            e = np.asarray(e)
            if i == 0:
                new_shape = (ArraySequence.BUFFER_SIZE,) + e.shape[1:]
                self._data = np.empty(new_shape, dtype=e.dtype)

            end = offset + len(e)
            if end >= len(self._data):
                # Resize needed, adding `len(e)` items plus some buffer.
                nb_points = len(self._data)
                nb_points += len(e) + ArraySequence.BUFFER_SIZE
                self._data.resize((nb_points,) + self.common_shape)

            offsets.append(offset)
            lengths.append(len(e))
            self._data[offset:offset + len(e)] = e
            offset += len(e)

        self._offsets = np.asarray(offsets)
        self._lengths = np.asarray(lengths)

        # Clear unused memory.
        if self._data.ndim != 0:
            self._data.resize((offset,) + self.common_shape)

    @property
    def is_array_sequence(self):
        return True

    @property
    def common_shape(self):
        """ Matching shape of the elements in this array sequence. """
        if self._data.ndim == 0:
            return ()

        return self._data.shape[1:]

    def append(self, element):
        """ Appends :obj:`element` to this array sequence.

        Parameters
        ----------
        element : ndarray
            Element to append. The shape must match already inserted elements
            shape except for the first dimension.

        Notes
        -----
        If you need to add multiple elements you should consider
        `ArraySequence.extend`.
        """
        if self._data.ndim == 0:
            self._data = np.asarray(element).copy()
            self._offsets = np.array([0])
            self._lengths = np.array([len(element)])
            return

        if element.shape[1:] != self.common_shape:
            msg = "All dimensions, except the first one, must match exactly"
            raise ValueError(msg)

        self._offsets = np.r_[self._offsets, len(self._data)]
        self._lengths = np.r_[self._lengths, len(element)]
        self._data = np.append(self._data, element, axis=0)

    def extend(self, elements):
        """ Appends all `elements` to this array sequence.

        Parameters
        ----------
        elements : list of ndarrays or :class:`ArraySequence` object
            If list of ndarrays, each ndarray will be concatenated along the
            first dimension then appended to the data of this ArraySequence.
            If :class:`ArraySequence` object, its data are simply appended to
            the data of this ArraySequence.

        Notes
        -----
            The shape of the elements to be added must match the one of the
            data of this :class:`ArraySequence` except for the first dimension.
        """
        if len(elements) == 0:
            return

        if self._data.ndim == 0:
            elem = np.asarray(elements[0])
            self._data = np.zeros((0, elem.shape[1]), dtype=elem.dtype)

        next_offset = self._data.shape[0]

        if is_array_sequence(elements):
            self._data.resize((self._data.shape[0] + sum(elements._lengths),
                               self._data.shape[1]))

            offsets = []
            for offset, length in zip(elements._offsets, elements._lengths):
                offsets.append(next_offset)
                chunk = elements._data[offset:offset + length]
                self._data[next_offset:next_offset + length] = chunk
                next_offset += length

            self._lengths = np.r_[self._lengths, elements._lengths]
            self._offsets = np.r_[self._offsets, offsets]

        else:
            self._data = np.concatenate([self._data] + list(elements), axis=0)
            lengths = list(map(len, elements))
            self._lengths = np.r_[self._lengths, lengths]
            self._offsets = np.r_[self._offsets,
                                  np.cumsum([next_offset] + lengths)[:-1]]

    def copy(self):
        """ Creates a copy of this :class:`ArraySequence` object. """
        # We do not simply deepcopy this object since we might have a chance
        # to use less memory. For example, if the array sequence being copied
        # is the result of a slicing operation on a array sequence.
        seq = ArraySequence()
        total_lengths = np.sum(self._lengths)
        seq._data = np.empty((total_lengths,) + self._data.shape[1:],
                             dtype=self._data.dtype)

        next_offset = 0
        offsets = []
        for offset, length in zip(self._offsets, self._lengths):
            offsets.append(next_offset)
            chunk = self._data[offset:offset + length]
            seq._data[next_offset:next_offset + length] = chunk
            next_offset += length

        seq._offsets = np.asarray(offsets)
        seq._lengths = self._lengths.copy()

        return seq

    def __getitem__(self, idx):
        """ Gets sequence(s) through advanced indexing.

        Parameters
        ----------
        idx : int or slice or list or ndarray
            If int, index of the element to retrieve.
            If slice, use slicing to retrieve elements.
            If list, indices of the elements to retrieve.
            If ndarray with dtype int, indices of the elements to retrieve.
            If ndarray with dtype bool, only retrieve selected elements.

        Returns
        -------
        ndarray or :class:`ArraySequence`
            If `idx` is an int, returns the selected sequence.
            Otherwise, returns a :class:`ArraySequence` object which is view
            of the selected sequences.
        """
        if isinstance(idx, (int, np.integer)):
            start = self._offsets[idx]
            return self._data[start:start + self._lengths[idx]]

        elif isinstance(idx, (slice, list)):
            seq = ArraySequence()
            seq._data = self._data
            seq._offsets = self._offsets[idx]
            seq._lengths = self._lengths[idx]
            seq._is_view = True
            return seq

        elif (isinstance(idx, np.ndarray) and
                (np.issubdtype(idx.dtype, np.integer) or
                 np.issubdtype(idx.dtype, np.bool))):
            seq = ArraySequence()
            seq._data = self._data
            seq._offsets = self._offsets[idx]
            seq._lengths = self._lengths[idx]
            seq._is_view = True
            return seq

        raise TypeError("Index must be either an int, a slice, a list of int"
                        " or a ndarray of bool! Not " + str(type(idx)))

    def __iter__(self):
        if len(self._lengths) != len(self._offsets):
            raise ValueError("ArraySequence object corrupted:"
                             " len(self._lengths) != len(self._offsets)")

        for offset, lengths in zip(self._offsets, self._lengths):
            yield self._data[offset: offset + lengths]

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        return repr(list(self))

    def save(self, filename):
        """ Saves this :class:`ArraySequence` object to a .npz file. """
        np.savez(filename,
                 data=self._data,
                 offsets=self._offsets,
                 lengths=self._lengths)

    @classmethod
    def from_filename(cls, filename):
        """ Loads a :class:`ArraySequence` object from a .npz file. """
        content = np.load(filename)
        seq = cls()
        seq._data = content["data"]
        seq._offsets = content["offsets"]
        seq._lengths = content["lengths"]
        return seq

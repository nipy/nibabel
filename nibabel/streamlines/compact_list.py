import numpy as np


class CompactList(object):
    """ Class for compacting list of ndarrays with matching shape except for
    the first dimension.
    """

    BUFFER_SIZE = 87382*4  # About 4 Mb if item shape is 3 (e.g. 3D points).

    def __init__(self, iterable=None):
        """
        Parameters
        ----------
        iterable : iterable (optional)
            If specified, create a :class:`CompactList` object initialized from
            iterable's items, otherwise it will be empty.

        Notes
        -----
        If `iterable` is a :class:`CompactList` object, a view is returned and no
        memory is allocated. For an actual copy use the `.copy()` method.
        """
        # Create new empty `CompactList` object.
        self._data = np.array(0)
        self._offsets = np.array([], dtype=int)
        self._lengths = np.array([], dtype=int)

        if isinstance(iterable, CompactList):
            # Create a view.
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths

        elif iterable is not None:
            offsets = []
            lengths = []
            # Initialize the `CompactList` object from iterable's item.
            offset = 0
            for i, e in enumerate(iterable):
                e = np.asarray(e)
                if i == 0:
                    new_shape = (CompactList.BUFFER_SIZE,) + e.shape[1:]
                    self._data = np.empty(new_shape, dtype=e.dtype)

                end = offset + len(e)
                if end >= len(self._data):
                    # Resize needed, adding `len(e)` new items plus some buffer.
                    nb_points = len(self._data)
                    nb_points += len(e) + CompactList.BUFFER_SIZE
                    self._data.resize((nb_points,) + self.shape)

                offsets.append(offset)
                lengths.append(len(e))
                self._data[offset:offset+len(e)] = e
                offset += len(e)

            self._offsets = np.asarray(offsets)
            self._lengths = np.asarray(lengths)

            # Clear unused memory.
            if self._data.ndim != 0:
                self._data.resize((offset,) + self.shape)

    @property
    def shape(self):
        """ Returns the matching shape of the elements in this compact list. """
        if self._data.ndim == 0:
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
        if self._data.ndim == 0:
            self._data = np.asarray(element).copy()
            self._offsets = np.array([0])
            self._lengths = np.array([len(element)])
            return

        if element.shape[1:] != self.shape:
            raise ValueError("All dimensions, except the first one,"
                             " must match exactly")

        self._offsets = np.r_[self._offsets, len(self._data)]
        self._lengths = np.r_[self._lengths, len(element)]
        self._data = np.append(self._data, element, axis=0)

    def extend(self, elements):
        """ Appends all `elements` to this compact list.

        Parameters
        ----------
        elements : list of ndarrays or :class:`CompactList` object
            If list of ndarrays, each ndarray will be concatenated along the
            first dimension then appended to the data of this CompactList.
            If :class:`CompactList` object, its data are simply appended to
            the data of this CompactList.

        Notes
        -----
            The shape of the elements to be added must match the one of the data
            of this CompactList except for the first dimension.

        """
        if self._data.ndim == 0:
            elem = np.asarray(elements[0])
            self._data = np.zeros((0, elem.shape[1]), dtype=elem.dtype)

        next_offset = self._data.shape[0]

        if isinstance(elements, CompactList):
            self._data.resize((self._data.shape[0]+sum(elements._lengths),
                               self._data.shape[1]))

            offsets = []
            for offset, length in zip(elements._offsets, elements._lengths):
                offsets.append(next_offset)
                self._data[next_offset:next_offset+length] = elements._data[offset:offset+length]
                next_offset += length

            self._lengths = np.r_[self._lengths, elements._lengths]
            self._offsets = np.r_[self._offsets, offsets]

        else:
            self._data = np.concatenate([self._data] + list(elements), axis=0)
            lengths = list(map(len, elements))
            self._lengths = np.r_[self._lengths, lengths]
            self._offsets = np.r_[self._offsets, np.cumsum([next_offset] + lengths)[:-1]]

    def copy(self):
        """ Creates a copy of this :class:`CompactList` object. """
        # We do not simply deepcopy this object since we might have a chance
        # to use less memory. For example, if the compact list being copied
        # is the result of a slicing operation on a compact list.
        clist = CompactList()
        total_lengths = np.sum(self._lengths)
        clist._data = np.empty((total_lengths,) + self._data.shape[1:],
                               dtype=self._data.dtype)

        next_offset = 0
        offsets = []
        for offset, length in zip(self._offsets, self._lengths):
            offsets.append(next_offset)
            clist._data[next_offset:next_offset+length] = self._data[offset:offset+length]
            next_offset += length

        clist._offsets = np.asarray(offsets)
        clist._lengths = self._lengths.copy()

        return clist

    def __getitem__(self, idx):
        """ Gets element(s) through indexing.

        Parameters
        ----------
        idx : int, slice or list
            Index of the element(s) to get.

        Returns
        -------
        ndarray object(s)
            When `idx` is an int, returns a single ndarray.
            When `idx` is either a slice or a list, returns a list of ndarrays.
        """
        if isinstance(idx, int) or isinstance(idx, np.integer):
            start = self._offsets[idx]
            return self._data[start:start+self._lengths[idx]]

        elif isinstance(idx, slice) or isinstance(idx, list):
            clist = CompactList()
            clist._data = self._data
            clist._offsets = self._offsets[idx]
            clist._lengths = self._lengths[idx]
            return clist

        elif isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.integer):
            clist = CompactList()
            clist._data = self._data
            clist._offsets = self._offsets[idx]
            clist._lengths = self._lengths[idx]
            return clist

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            clist = CompactList()
            clist._data = self._data
            clist._offsets = [self._offsets[i]
                              for i, take_it in enumerate(idx) if take_it]
            clist._lengths = [self._lengths[i]
                              for i, take_it in enumerate(idx) if take_it]
            return clist

        raise TypeError("Index must be either an int, a slice, a list of int"
                        " or a ndarray of bool! Not " + str(type(idx)))

    def __iter__(self):
        if len(self._lengths) != len(self._offsets):
            raise ValueError("CompactList object corrupted:"
                             " len(self._lengths) != len(self._offsets)")

        for offset, lengths in zip(self._offsets, self._lengths):
            yield self._data[offset: offset+lengths]

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        return repr(list(self))


def save_compact_list(filename, clist):
    """ Saves a :class:`CompactList` object to a .npz file. """
    np.savez(filename,
             data=clist._data,
             offsets=clist._offsets,
             lengths=clist._lengths)


def load_compact_list(filename):
    """ Loads a :class:`CompactList` object from a .npz file. """
    content = np.load(filename)
    clist = CompactList()
    clist._data = content["data"]
    clist._offsets = content["offsets"]
    clist._lengths = content["lengths"]
    return clist

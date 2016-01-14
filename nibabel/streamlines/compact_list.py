import numpy as np


class CompactList(object):
    """ Class for compacting list of ndarrays with matching shape except for
    the first dimension.
    """

    BUFFER_SIZE = 349525  # About 4 Mb if item shape is 3 (e.g. 3D points).

    def __init__(self, iterable=None):
        """
        Parameters
        ----------
        iterable : iterable (optional)
            If specified, create a ``CompactList`` object initialized from
            iterable's items. Otherwise, create an empty ``CompactList``.

        Notes
        -----
        If `iterable` is a ``CompactList`` object, a view is returned and no
        memory is allocated. For an actual copy use the `.copy()` method.
        """
        # Create new empty `CompactList` object.
        self._data = np.array(0)
        self._offsets = []
        self._lengths = []

        if isinstance(iterable, CompactList):
            # Create a view.
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths

        elif iterable is not None:
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

                self._offsets.append(offset)
                self._lengths.append(len(e))
                self._data[offset:offset+len(e)] = e
                offset += len(e)

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
            self._offsets.append(0)
            self._lengths.append(len(element))
            return

        if element.shape[1:] != self.shape:
            raise ValueError("All dimensions, except the first one,"
                             " must match exactly")

        self._offsets.append(len(self._data))
        self._lengths.append(len(element))
        self._data = np.append(self._data, element, axis=0)

    def extend(self, elements):
        """ Appends all `elements` to this compact list.

        Parameters
        ----------
        elements : list of ndarrays, ``CompactList`` object
            Elements to append. The shape must match already inserted elements
            shape except for the first dimension.

        """
        if self._data.ndim == 0:
            elem = np.asarray(elements[0])
            self._data = np.zeros((0, elem.shape[1]), dtype=elem.dtype)

        next_offset = self._data.shape[0]

        if isinstance(elements, CompactList):
            self._data.resize((self._data.shape[0]+sum(elements._lengths),
                               self._data.shape[1]))

            for offset, length in zip(elements._offsets, elements._lengths):
                self._offsets.append(next_offset)
                self._lengths.append(length)
                self._data[next_offset:next_offset+length] = elements._data[offset:offset+length]
                next_offset += length

        else:
            self._data = np.concatenate([self._data] + list(elements), axis=0)
            lengths = list(map(len, elements))
            self._lengths.extend(lengths)
            self._offsets.extend(np.cumsum([next_offset] + lengths).tolist()[:-1])

    def copy(self):
        """ Creates a copy of this ``CompactList`` object. """
        # We do not simply deepcopy this object since we might have a chance
        # to use less memory. For example, if the compact list being copied
        # is the result of a slicing operation on a compact list.
        clist = CompactList()
        total_lengths = np.sum(self._lengths)
        clist._data = np.empty((total_lengths,) + self._data.shape[1:],
                               dtype=self._data.dtype)

        idx = 0
        for offset, length in zip(self._offsets, self._lengths):
            clist._offsets.append(idx)
            clist._lengths.append(length)
            clist._data[idx:idx+length] = self._data[offset:offset+length]
            idx += length

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

        elif isinstance(idx, slice):
            clist = CompactList()
            clist._data = self._data
            clist._offsets = self._offsets[idx]
            clist._lengths = self._lengths[idx]
            return clist

        elif isinstance(idx, list):
            clist = CompactList()
            clist._data = self._data
            clist._offsets = [self._offsets[i] for i in idx]
            clist._lengths = [self._lengths[i] for i in idx]
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
    """ Saves a `CompactList` object to a .npz file. """
    np.savez(filename,
             data=clist._data,
             offsets=clist._offsets,
             lengths=clist._lengths)


def load_compact_list(filename):
    """ Loads a `CompactList` object from a .npz file. """
    content = np.load(filename)
    clist = CompactList()
    clist._data = content["data"]
    clist._offsets = content["offsets"].tolist()
    clist._lengths = content["lengths"].tolist()
    return clist

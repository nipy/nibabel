import numpy as np


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

        Notes
        -----
        If `iterable` is a ``CompactList`` object, a view is returned and no
        memory is allocated. For an actual copy use the `.copy()` method.
        """
        # Create new empty `CompactList` object.
        self._data = None
        self._offsets = []
        self._lengths = []

        if isinstance(iterable, CompactList):
            # Create a view.
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths

        elif iterable is not None:
            # Initialize the `CompactList` object from iterable's item.
            BUFFER_SIZE = 10000000  # About 128 Mb if item shape is 3.

            offset = 0
            for i, e in enumerate(iterable):
                e = np.asarray(e)
                if i == 0:
                    self._data = np.empty((BUFFER_SIZE,) + e.shape[1:],
                                          dtype=e.dtype)

                end = offset + len(e)
                if end >= len(self._data):
                    # Resize is needed (at least `len(e)` items will be added).
                    self._data.resize((len(self._data) + len(e)+BUFFER_SIZE,)
                                      + self.shape)

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
        if self._data is None:
            elem = np.asarray(elements[0])
            self._data = np.zeros((0, elem.shape[1]), dtype=elem.dtype)

        if isinstance(elements, CompactList):
            self._data = np.concatenate([self._data, elements._data], axis=0)
            lengths = elements._lengths
        else:
            self._data = np.concatenate([self._data] + list(elements), axis=0)
            lengths = map(len, elements)

        idx = self._offsets[-1] + self._lengths[-1] if len(self) > 0 else 0
        self._lengths.extend(lengths)
        self._offsets.extend(np.cumsum([idx] + lengths).tolist()[:-1])

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
            When `idx` is a int, returns a single ndarray.
            When `idx` is either a slice or a list, returns a list of ndarrays.
        """
        if isinstance(idx, int) or isinstance(idx, np.integer):
            start = self._offsets[idx]
            return self._data[start:start+self._lengths[idx]]

        elif type(idx) is slice:
            compact_list = CompactList()
            compact_list._data = self._data
            compact_list._offsets = self._offsets[idx]
            compact_list._lengths = self._lengths[idx]
            return compact_list

        elif type(idx) is list:
            compact_list = CompactList()
            compact_list._data = self._data
            compact_list._offsets = [self._offsets[i] for i in idx]
            compact_list._lengths = [self._lengths[i] for i in idx]
            return compact_list

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

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


def save_compact_list(filename, compact_list):
    """ Saves a `CompactList` object to a .npz file. """
    np.savez(filename,
             data=compact_list._data,
             offsets=compact_list._offsets,
             lengths=compact_list._lengths)


def load_compact_list(filename):
    """ Loads a `CompactList` object from a .npz file. """
    content = np.load(filename)
    compact_list = CompactList()
    compact_list._data = content["data"]
    compact_list._offsets = content["offsets"].tolist()
    compact_list._lengths = content["lengths"].tolist()
    return compact_list

'''Containers for storing "elements" which have both a core data value as well
as some additional meta data. When indexing into these containers it is this
core value that is returned, which allows for much cleaner and more readable
access to nested structures.

Each object stored in these containers must have an attribute `value` which
provides the core data value for the element. To get the element object itself
the `get_elem` method must be used.
'''
from collections import MutableMapping, MutableSequence

from .externals import OrderedDict
from .externals.six import iteritems


class Elem(object):
    '''Basic element type has a `value` and a `meta` attribute.'''
    def __init__(self, value, meta=None):
        self.value = value
        self.meta = {} if meta is None else meta


class InvalidElemError(Exception):
    '''Raised when trying to add an object without a `value` attribute to an
    `ElemDict`.'''


class ElemDict(MutableMapping):
    '''Ordered dict-like where each value is an "element", which is defined as
    any object which has a `value` attribute.

    When looking up an item in the dict, it is this `value` attribute that
    is returned. To get the element itself use the `get_elem` method.
    '''

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError("At most one arg expected, got %d" % len(args))
        self._elems = OrderedDict()
        if len(args) == 1:
            if hasattr(args[0], 'items'):
                it = iteritems(args[0])
            else:
                it = args[0]
            for key, val in it:
                if isinstance(val, dict):
                    val = self.__class__(val)
                self[key] = val
        for key, val in iteritems(kwargs):
            if isinstance(val, dict):
                val = self.__class__(val)
            self[key] = val

    def __getitem__(self, key):
        return self._elems[key].value

    def __setitem__(self, key, val):
        if not hasattr(val, 'value'):
            raise InvalidElemError()
        self._elems[key] = val

    def __delitem__(self, key):
        del self._elems[key]

    def __iter__(self):
        return iter(self._elems)

    def __len__(self):
        return len(self._elems)

    def __repr__(self):
        return ('ElemDict(%s)' %
                ', '.join(['%r=%r' % x for x in self.items()]))

    def get_elem(self, key):
        return self._elems[key]


class ElemList(MutableSequence):
    '''A list-like container where each value is an "element", which is
    defined as any object which has a `value` attribute.

    When looking up an item in the list, it is this `value` attribute that
    is returned. To get the element itself use the `get_elem` method.
    '''
    def __init__(self, data=None):
        self._elems = list()
        if data is not None:
            for elem in data:
                self.append(elem)

    def _tuple_from_slice(self, slc):
        '''Get (start, end, step) tuple from slice object.
        '''
        (start, end, step) = slc.indices(len(self))
        # Replace (0, -1, 1) with (0, 0, 1) (misfeature in .indices()).
        if step == 1:
          if end < start:
            end = start
          step = None
        if slc.step == None:
          step = None
        return (start, end, step)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ElemList(self._elems[idx])
        else:
            return self._elems[idx].value

    def __setitem__(self, idx, val):
        if isinstance(idx, slice):
            (start, end, step) = self._tuple_from_slice(idx)
            if step != None:
                # Extended slice
                indices = range(start, end, step)
                if len(val) != len(indices):
                    raise ValueError(('attempt to assign sequence of size %d' +
                                      ' to extended slice of size %d') %
                                     (len(value), len(indices)))
                for j, assign_val in enumerate(val):
                    self.insert(indices[j], assign_val)
            else:
                # Normal slice
                for j, assign_val in enumerate(val):
                    self.insert(start + j, assign_val)
        else:
            self.insert(idx, val)

    def __delitem__(self, idx):
        del self._elems[idx]

    def __len__(self):
        return len(self._elems)

    def __repr__(self):
        return ('ElemList([%s])' % ', '.join(['%r' % x for x in self]))

    def insert(self, idx, val):
        if not hasattr(val, 'value'):
            raise InvalidElemError()
        self._elems.insert(idx, val)

    def append(self, val):
        list_idx = len(self._elems)
        self.insert(list_idx, val)

    def get_elem(self, idx):
        return self._elems[idx]
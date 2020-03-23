# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''Containers that provide easy access to the values of nested elements

These containers are for storing "elements" which have both a core data value
as well as some additional meta data. When indexing into these containers it is
this core value that is returned, which allows for much cleaner and more
readable access to nested structures.

Each object stored in these containers must have an attribute `value` which
provides the core data value for the element. To get the element object itself
the `get_elem` method must be used.
'''
from collections import MutableMapping, MutableSequence

from .externals import OrderedDict


class MetaElem(object):
    '''Basic element type has a `value` and a `meta` attribute.'''
    def __init__(self, value, meta=None):
        self.value = value
        self.meta = {} if meta is None else meta


class InvalidElemError(Exception):
    '''The object being added to the container doesn't have a `value` attribute
    '''
    def __init__(self, invalid_val):
        message = ("Provided value '%s' of type %s does not have a 'value' "
                   "attribute" % (invalid_val, type(invalid_val)))
        super(InvalidElemError, self).__init__(message)


class ElemDict(MutableMapping):
    '''Ordered dict-like providing easy access to nested elements

    Each value added to the dict must in turn have a `value` attribute, which
    is what is returned by subsequent calls to `__getitem__`. To get the
    element itself use the `get_elem` method.
    '''

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError("At most one arg expected, got %d" % len(args))
        self._elems = OrderedDict()
        if len(args) == 1:
            arg = args[0]
            if hasattr(arg, 'get_elem'):
                it = ((k, arg.get_elem(k)) for k in arg)
            elif hasattr(arg, 'items'):
                it = arg.items()
            else:
                it = arg
            for key, val in it:
                self[key] = val
        for key, val in kwargs.items():
            self[key] = val

    def __getitem__(self, key):
        return self._elems[key].value

    def __setitem__(self, key, val):
        if not hasattr(val, 'value'):
            raise InvalidElemError(val)
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

    def update(self, other):
        if hasattr(other, 'get_elem'):
            for key in other:
                self[key] = other.get_elem(key)
        else:
            for key, elem in other.items():
                self[key] = elem

    def get_elem(self, key):
        return self._elems[key]


class ElemList(MutableSequence):
    '''A list-like container providing easy access to nested elements

    Each value added to the list must in turn have a `value` attribute, which
    is what is returned by subsequent calls to `__getitem__`. To get the
    element itself use the `get_elem` method.
    '''
    def __init__(self, data=None):
        self._elems = list()
        if data is None:
            return
        if isinstance(data, self.__class__):
            for idx in range(len(data)):
                self.append(data.get_elem(idx))
        else:
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
        if slc.step is None:
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
            if step is None:
                # Normal slice
                for j, assign_val in enumerate(val):
                    self.insert(start + j, assign_val)
                return
            # Extended slice
            indices = range(start, end, step)
            if len(val) != len(indices):
                raise ValueError(('attempt to assign sequence of size %d' +
                                  ' to extended slice of size %d') %
                                 (len(value), len(indices)))
            for j, assign_val in enumerate(val):
                self.insert(indices[j], assign_val)
        else:
            self.insert(idx, val)

    def __delitem__(self, idx):
        del self._elems[idx]

    def __len__(self):
        return len(self._elems)

    def __repr__(self):
        return ('ElemList([%s])' % ', '.join(['%r' % x for x in self]))

    def __add__(self, other):
        result = self.__class__(self)
        if isinstance(other, self.__class__):
            for idx in range(len(other)):
                result.append(other.get_elem(idx))
        else:
            for e in other:
                result.append(e)
        return result

    def __radd__(self, other):
        result = self.__class__(other)
        for idx in range(len(self)):
            result.append(self.get_elem(idx))
        return result

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            for idx in range(len(other)):
                self.append(other.get_elem(idx))
        else:
            for e in other:
                self.append(e)
        return self

    def insert(self, idx, val):
        if not hasattr(val, 'value'):
            raise InvalidElemError(val)
        self._elems.insert(idx, val)

    def get_elem(self, idx):
        return self._elems[idx]

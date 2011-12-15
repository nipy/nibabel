""" State stamps

A state stamp is something that defines the state of an object. Let's say we have
an object ``X`` is in a state $S$.

Let's call the state stamp finder ``get_state_stamp``.  This could be the result
of ``get_state_stamp = Stamper()`` below, for example.

The *state stamp* of ``X`` is some value ``g`` such that ``get_state_stamp(X) ==
g`` if and only if ``X`` is in state $S$ - however defined.

``get_state_stamp(Y) == g`` should in general not be true if ``Y`` is a
different class for ``X``.

Thus the state stamp guarantees a particular state of ``X`` - as defined by you,
dear programmer. Conversely, if ``get_state_stamp(X) != g`` this does not
guarantee they are different.  It may be that you (dear programmer) don't know
if they are different, and do not want to spend resources on working it out.
For example, if ``X`` is a huge array, you might want to return the
``Unknown()`` state stamp.

The state stamp ``Unknown()`` is the state stamp such that ``get_state_stamp(X) ==
Unknown()`` is always False.

If you have objects you want compared, you can do one of:

* define a ``state_stamp`` method, taking a single argument ``caller`` which is
  the callable from which the method has been called. You can then return
  something which is unique for the states you want to be able to distinguish.
  Don't forget that (usually) stamps from objects of different types should
  compare unequal.
* subclass the ``Stamper`` class, and extend the ``__call__`` method to
  handle a new object type.  The ``NdaStamper`` class below is an example.

It's up to the object how to do the stamping.  In general, don't test what the
stamp is, test whether it compares equal in the situations you are expecting, so
that the object can change it's mind about how it will do the stamping without
you having to rewrite the tests.
"""

import hashlib

import numpy as np

from .py3k import bytes, unicode


class Unknown(object):
    """ state stamp that never matches

    Examples
    --------
    >>> u = Unknown()
    >>> u == u
    False
    >>> p = Unknown()
    >>> u == p
    False

    Notes
    -----
    You would think this could be a singleton, but not so, because:

    >>> u = Unknown()
    >>> (1, u) == (1, u)
    True

    Why?  Because comparisons within sequences in CPython, use
    ``PyObject_RichCompareBool`` for the elements. See around line 572 in
    ``objects/tupleobject.c`` and around line 607 in ``objects/object.c`` in
    cpython 69528:fecf9e6d7630.  This does an identity check equivalent to ``u
    is u``; if this passes it does not do a further equality check (``u == u``).
    For that reason, if you want to make sure nothing matches ``Unknown()``
    within sequences, you need a fresh instances.
    """
    _is_unknown = True

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __repr__(self):
        return 'Unknown()'


def is_unknown(obj):
    """ Return True if `obj` is an Unknown instance

    Examples
    --------
    >>> is_unknown(Unknown())
    True
    >>> is_unknown(object())
    False
    """
    try:
        return obj._is_unknown
    except AttributeError:
        return False


class Stamper(object):
    r""" Basic state stamp collector

    Instantiate and call on objects to get state stamps

    Examples
    --------
    >>> asker = Stamper()
    >>> asker(1) == asker(1)
    True
    >>> asker(1) == asker(2)
    False
    >>> asker('a string') == asker('a string')
    True
    >>> asker(1.0) == asker(1.0)
    True
    >>> asker(1) == asker(1.0) # different types
    False
    >>> asker(object()) == asker(object()) # not known -> False
    False

    List and tuples

    >>> L = [1, 2]
    >>> asker(L) == asker([1, 2])
    True
    >>> L[0] = 3
    >>> asker(L) == asker([1, 2])
    False
    >>> T = (1, 2)
    >>> asker(T) == asker((1, 2))
    True
    >>> asker(T) == asker([1, 2])
    False
    >>> asker([1, object()]) == asker([1, object()])
    False

    If your object implements ``state_stamper``, you can customized the
    behavior.

    >>> class D(object):
    ...     def state_stamper(self, cstate):
    ...          return 28
    >>> asker(D()) == asker(D())
    True
    """
    def __init__(self, funcs = None):
        """ Initialize stamper with optional functions ``funcs``

        Parameters
        ----------
        funcs : sequence of callables, optional
            callables that will be called to process otherwise unknown objects.
            The signature for the callable is ``f(obj, caller)`` where `obj` is
            the object being stamped, and ``caller`` will be the
            ``Stamper``-like object from which the function will be called.

        Examples
        --------
        >>> st = Stamper()
        >>> st((1, object())) == st((1, object()))
        False
        >>> def func(obj, caller):
        ...     return type(obj), 28
        >>> st2 = Stamper((func,))
        >>> st2((1, object())) == st2((1, object()))
        True
        """
        if funcs is None:
            funcs = []
        self.funcs = list(funcs)
        # In case custom objects want an intermediate store
        self.call_state = {}

    def __call__(self, obj):
        r""" Get state stamp for object `obj`

        Parmeters
        ---------
        obj : object
            Object for which to extract state stamp

        Returns
        -------
        stamp_state : object
            state stamp.  This is an object that compares equal to another
            object in the same `state`
        """
        # Reset call state, in case someone wants to use it
        self.call_state = {}
        # None passes through
        if obj is None:
            return None
        tobj = type(obj)
        # Pass through classes before doing method check on instance
        if tobj == type: # class
            return type, obj
        try:
            return obj.state_stamper(self)
        except AttributeError:
            pass
        # Immutable objects are their own state stamps
        if tobj in (str, unicode, bytes, int, float):
            return tobj, obj
        if tobj is dict:
            return dict, self(obj.items())
        # Recurse into known sequence types
        if tobj in (list, tuple):
            return tobj(self(v) for v in obj)
        # Try any additional functions we know about
        for func in self.funcs:
            res = func(obj, self)
            if not res is None and not is_unknown(res):
                return res
        return Unknown()


class NdaStamper(Stamper):
    r""" Collect state stamps, using byte buffers for smallish ndarrays

    >>> nda_asker = NdaStamper()

    The standard Stamper behavior

    >>> nda_asker(1) == nda_asker(1)
    True

    Can also deal with small arrays by hashing byte contents:

    >>> arr = np.zeros((3,), dtype=np.int16)
    >>> nda_asker(arr) == nda_asker(arr)
    True

    Depending on the threshold for the number of bytes:

    >>> small_asker = NdaStamper(byte_thresh=5)
    >>> small_asker(arr)
    Unknown()
    """
    def __init__(self, funcs = None, byte_thresh = 2**16):
        self.byte_thresh = byte_thresh
        if funcs is None:
            funcs = []
        def _proc_array(obj, cstate):
            if type(obj) is np.ndarray and obj.nbytes <= byte_thresh:
                return (type(obj),
                        tuple(obj.shape),
                        obj.dtype,
                        hashlib.md5(obj.tostring()).digest())
        super(NdaStamper, self).__init__(list(funcs) + [_proc_array])

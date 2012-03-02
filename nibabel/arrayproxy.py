# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Array proxy base class

The API is - at minimum:

* The object has an attribute ``shape``
* that the object returns the data array from ``np.asarray(obj)``
* that modifying no object outside ``obj`` will affect the result of
  ``np.asarray(obj)``.  Specifically, if you pass a header into the the
  __init__, then modifying the original header will not affect the result of the
  array return.

You might also want to implement ``state_stamper``
"""

from .volumeutils import allopen


class ArrayProxy(object):
    """
    The array proxy allows us to freeze the passed fileobj and header such that
    it returns the expected data array.

    This fairly generic implementation allows us to deal with Analyze and its
    variants, including Nifti1, and with the MGH format, apparently.

    It requires a ``header`` object with methods:
    * copy
    * get_data_shape
    * data_from_fileobj

    Other image types might need to implement their own implementation of this
    API.  See :mod:`minc` for an example.
    """
    def __init__(self, file_like, header):
        self.file_like = file_like
        self.header = header.copy()
        self._data = None
        self._shape = header.get_data_shape()

    @property
    def shape(self):
        return self._shape

    def __array__(self):
        ''' Cached read of data from file '''
        if self._data is None:
            self._data = self._read_data()
        return self._data

    def _read_data(self):
        fileobj = allopen(self.file_like)
        data = self.header.data_from_fileobj(fileobj)
        if isinstance(self.file_like, basestring):  # filename
            fileobj.close()
        return data

    def state_stamper(self, caller):
        """ Return stamp for current state of `self`

        The result somewhat uniquely identifies the state of the array proxy.
        It assumes that the underlying ``self.file_like`` does not get modified.
        Specifically, if you open a file-like object, pass into an arrayproxy
        (call it ``ap``) and get the stamp (say with ``Stamper()(ap)``, then
        this stamp will uniquely identify the result of ``np.asarry(ap)`` only
        if the file-like object has not changed.

        Parameters
        ----------
        caller : callable
            callable object from which this method was called.

        Returns
        -------
        stamp : object
            object unique to this state of `self`

        Notes
        -----
        The stamp changes if the array to be returned has been cached
        (``_data`` attribute). This is because this makes it possible to change
        the array outside the proxy object, because further calls to
        ``__array__`` returns a refernce to ``self._data``, and the reference
        allows the caller to modify the array in-place.
        """
        return (self.__class__,
                self.file_like,
                caller(self.header),
                caller(self._data))

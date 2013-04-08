# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Context manager openers for various fileobject types
"""

from os.path import splitext
import gzip
import bz2


class Opener(object):
    """ Class to accept, maybe open, and context-manage file-likes / filenames

    Provides context manager to close files that the constructor opened for you.

    Parameters
    ----------
    fileish : str or file-like
        if str, then open with suitable opening method. If file-like, accept as
        is
    \*args : positional arguments
        passed to opening method when `fileish` is str.  ``mode``, if not
        specified, is `rb`.  ``compresslevel``, if relevant, and not specified,
        is set from class variable ``default_compresslevel``
    \*\*kwargs : keyword arguments
        passed to opening method when `fileish` is str.  Change of defaults as
        for \*args
    """
    gz_def = (gzip.open, ('mode', 'compresslevel'))
    bz2_def = (bz2.BZ2File, ('mode', 'buffering', 'compresslevel'))
    compress_ext_map = {
        '.gz': gz_def,
        '.bz2': bz2_def,
        None: (open, ('mode', 'buffering'))
    }
    #: default compression level when writing gz and bz2 files
    default_compresslevel = 1

    def __init__(self, fileish, *args, **kwargs):
        if self._is_fileobj(fileish):
            self.fobj = fileish
            self.me_opened = False
            self._name = None
            return
        _, ext = splitext(fileish)
        if ext in self.compress_ext_map:
            is_compressor = True
            opener, arg_names = self.compress_ext_map[ext]
        else:
            is_compressor = False
            opener, arg_names = self.compress_ext_map[None]
        # Get full arguments to check for mode and compresslevel
        full_kwargs = kwargs.copy()
        n_args = len(args)
        full_kwargs.update(dict(zip(arg_names[:n_args], args)))
        # Set default mode
        if not 'mode' in full_kwargs:
            kwargs['mode'] = 'rb'
        if is_compressor and not 'compresslevel' in kwargs:
            kwargs['compresslevel'] = self.default_compresslevel
        self.fobj = opener(fileish, *args, **kwargs)
        self._name = fileish
        self.me_opened = True

    def _is_fileobj(self, obj):
        """ Is `obj` a file-like object?
        """
        return hasattr(obj, 'read') and hasattr(obj, 'write')

    @property
    def closed(self):
        return self.fobj.closed

    @property
    def name(self):
        """ Return ``self.fobj.name`` or self._name if not present

        self._name will be None if object was created with a fileobj, otherwise
        it will be the filename.
        """
        try:
            return self.fobj.name
        except AttributeError:
            return self._name

    @property
    def mode(self):
        return self.fobj.mode

    def read(self, *args, **kwargs):
        return self.fobj.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self.fobj.write(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.fobj.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.fobj.tell(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.fobj.close(*args, **kwargs)

    def close_if_mine(self):
        """ Close ``self.fobj`` iff we opened it in the constructor
        """
        if self.me_opened:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_if_mine()

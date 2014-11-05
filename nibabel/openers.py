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

# The largest memory chunk that gzip can use for reads
GZIP_MAX_READ_CHUNK = 100 * 1024 * 1024 # 100Mb


def _gzip_open(fileish, *args, **kwargs):
    # open gzip files with faster reads on large files using larger chunks
    # See https://github.com/nipy/nibabel/pull/210 for discussion
    gzip_file = gzip.open(fileish, *args, **kwargs)
    gzip_file.max_read_chunk = GZIP_MAX_READ_CHUNK
    return gzip_file


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
    gz_def = (_gzip_open, ('mode', 'compresslevel'))
    bz2_def = (bz2.BZ2File, ('mode', 'buffering', 'compresslevel'))
    compress_ext_map = {
        '.gz': gz_def,
        '.bz2': bz2_def,
        None: (open, ('mode', 'buffering'))  # default
    }
    #: default compression level when writing gz and bz2 files
    default_compresslevel = 1
    #: whether to ignore case looking for compression extensions
    compress_ext_icase = True

    def __init__(self, fileish, *args, **kwargs):
        if self._is_fileobj(fileish):
            self.fobj = fileish
            self.me_opened = False
            self._name = None
            return
        opener, arg_names = self._get_opener_argnames(fileish)
        # Get full arguments to check for mode and compresslevel
        full_kwargs = kwargs.copy()
        n_args = len(args)
        full_kwargs.update(dict(zip(arg_names[:n_args], args)))
        # Set default mode
        if not 'mode' in full_kwargs:
            kwargs['mode'] = 'rb'
        if 'compresslevel' in arg_names and 'compresslevel' not in kwargs:
            kwargs['compresslevel'] = self.default_compresslevel
        self.fobj = opener(fileish, *args, **kwargs)
        self._name = fileish
        self.me_opened = True

    def _get_opener_argnames(self, fileish):
        _, ext = splitext(fileish)
        if self.compress_ext_icase:
            ext = ext.lower()
            for key in self.compress_ext_map:
                if key is None:
                    continue
                if key.lower() == ext:
                    return self.compress_ext_map[key]
        elif ext in self.compress_ext_map:
            return self.compress_ext_map[ext]
        return self.compress_ext_map[None]

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

    def fileno(self):
        return self.fobj.fileno()

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

    def __iter__(self):
        return iter(self.fobj)

    def close_if_mine(self):
        """ Close ``self.fobj`` iff we opened it in the constructor
        """
        if self.me_opened:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_if_mine()

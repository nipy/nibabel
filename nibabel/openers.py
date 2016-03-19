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

import bz2
import gzip
import sys
from os.path import splitext


# The largest memory chunk that gzip can use for reads
GZIP_MAX_READ_CHUNK = 100 * 1024 * 1024  # 100Mb


class BufferedGzipFile(gzip.GzipFile):
    """GzipFile able to readinto buffer >= 2**32 bytes.

    This class only differs from gzip.GzipFile
    in Python 3.5.0.

    This works around a known issue in Python 3.5.
    See https://bugs.python.org/issue25626
    """

    # This helps avoid defining readinto in Python 2.6,
    #   where it is undefined on gzip.GzipFile.
    # It also helps limit the exposure to this code.
    if sys.version_info[:3] == (3, 5, 0):
        def __init__(self, fileish, mode='rb', compresslevel=9,
                     buffer_size=2**32 - 1):
            super(BufferedGzipFile, self).__init__(fileish, mode=mode,
                                                   compresslevel=compresslevel)
            self.buffer_size = buffer_size

        def readinto(self, buf):
            """Uses self.buffer_size to do a buffered read."""
            n_bytes = len(buf)
            if n_bytes < 2 ** 32:
                return super(BufferedGzipFile, self).readinto(buf)

            # This works around a known issue in Python 3.5.
            # See https://bugs.python.org/issue25626
            mv = memoryview(buf)
            n_read = 0
            max_read = 2 ** 32 - 1  # Max for unsigned 32-bit integer
            while (n_read < n_bytes):
                n_wanted = min(n_bytes - n_read, max_read)
                n_got = super(BufferedGzipFile, self).readinto(
                    mv[n_read:n_read + n_wanted])
                n_read += n_got
                if n_got != n_wanted:
                    break
            return n_read


def _gzip_open(fileish, *args, **kwargs):
    gzip_file = BufferedGzipFile(fileish, *args, **kwargs)

    # Speedup for #209; attribute not present in in Python 3.5
    # open gzip files with faster reads on large files using larger
    # See https://github.com/nipy/nibabel/pull/210 for discussion
    if hasattr(gzip_file, 'max_chunk_read'):
        gzip_file.max_read_chunk = GZIP_MAX_READ_CHUNK

    return gzip_file


class Opener(object):
    """ Class to accept, maybe open, and context-manage file-likes / filenames

    Provides context manager to close files that the constructor opened for
    you.

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
        if 'mode' not in full_kwargs:
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


class ImageOpener(Opener):
    """ Opener-type class to collect extra compressed extensions

    A trivial sub-class of opener to which image classes can add extra
    extensions with custom openers, such as compressed openers.

    To add an extension, add a line to the class definition (not __init__):

        ImageOpener.compress_ext_map[ext] = func_def

    ``ext`` is a file extension beginning with '.' and should be included in
    the image class's ``valid_exts`` tuple.

    ``func_def`` is a `(function, (args,))` tuple, where `function accepts a
    filename as the first parameter, and `args` defines the other arguments
    that `function` accepts. These arguments must be any (unordered) subset of
    `mode`, `compresslevel`, and `buffering`.
    """
    # Add new extensions to this dictionary
    compress_ext_map = Opener.compress_ext_map.copy()

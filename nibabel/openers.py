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

from bz2 import BZ2File
import gzip
import warnings
from os.path import splitext
from distutils.version import StrictVersion

# is indexed_gzip present and modern?
try:
    import indexed_gzip as igzip
    version = igzip.__version__

    HAVE_INDEXED_GZIP = True

    # < 0.7 - no good
    if StrictVersion(version) < StrictVersion('0.7.0'):
        warnings.warn(f'indexed_gzip is present, but too old (>= 0.7.0 required): {version})')
        HAVE_INDEXED_GZIP = False
    # >= 0.8 SafeIndexedGzipFile renamed to IndexedGzipFile
    elif StrictVersion(version) < StrictVersion('0.8.0'):
        IndexedGzipFile = igzip.SafeIndexedGzipFile
    else:
        IndexedGzipFile = igzip.IndexedGzipFile
    del igzip, version

except ImportError:
    # nibabel.openers.IndexedGzipFile is imported by nibabel.volumeutils
    # to detect compressed file types, so we give a fallback value here.
    IndexedGzipFile = gzip.GzipFile
    HAVE_INDEXED_GZIP = False


def _gzip_open(filename, mode='rb', compresslevel=9, keep_open=False):

    # use indexed_gzip if possible for faster read access.  If keep_open ==
    # True, we tell IndexedGzipFile to keep the file handle open. Otherwise
    # the IndexedGzipFile will close/open the file on each read.
    if HAVE_INDEXED_GZIP and mode == 'rb':
        gzip_file = IndexedGzipFile(filename, drop_handles=not keep_open)

    # Fall-back to built-in GzipFile
    else:
        gzip_file = gzip.GzipFile(filename, mode, compresslevel)

    return gzip_file


class Opener(object):
    r""" Class to accept, maybe open, and context-manage file-likes / filenames

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
        is set from class variable ``default_compresslevel``. ``keep_open``, if
        relevant, and not specified, is ``False``.
    \*\*kwargs : keyword arguments
        passed to opening method when `fileish` is str.  Change of defaults as
        for \*args
    """
    gz_def = (_gzip_open, ('mode', 'compresslevel', 'keep_open'))
    bz2_def = (BZ2File, ('mode', 'buffering', 'compresslevel'))
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
        # Default compression level
        if 'compresslevel' in arg_names and 'compresslevel' not in kwargs:
            kwargs['compresslevel'] = self.default_compresslevel
        # Default keep_open hint
        if 'keep_open' in arg_names:
            kwargs.setdefault('keep_open', False)
        # Clear keep_open hint if it is not relevant for the file type
        else:
            kwargs.pop('keep_open', None)
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

    def readinto(self, *args, **kwargs):
        return self.fobj.readinto(*args, **kwargs)

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

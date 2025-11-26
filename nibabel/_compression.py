# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Constants and types for dealing transparently with compression"""

from __future__ import annotations

import bz2
import gzip
import typing as ty

from .optpkg import optional_package

if ty.TYPE_CHECKING:
    import io

    import indexed_gzip  # type: ignore[import]

    # >= py314
    try:
        from compression import zstd  # type: ignore[import]
    # < py314
    except ImportError:
        from backports import zstd  # type: ignore[import]

    HAVE_INDEXED_GZIP = True
    HAVE_ZSTD = True
else:
    indexed_gzip, HAVE_INDEXED_GZIP, _ = optional_package('indexed_gzip')
    zstd, HAVE_ZSTD, _ = optional_package(('compression.zstd',
                                           'backports.zstd', 'pyzstd'))

# Collections of types for isinstance or exception matching
COMPRESSED_FILE_LIKES: tuple[type[io.IOBase], ...] = (
    bz2.BZ2File,
    gzip.GzipFile,
)
COMPRESSION_ERRORS: tuple[type[BaseException], ...] = (
    OSError,  # BZ2File
    gzip.BadGzipFile,
)

if HAVE_INDEXED_GZIP:
    COMPRESSED_FILE_LIKES += (indexed_gzip.IndexedGzipFile,)
    COMPRESSION_ERRORS += (indexed_gzip.ZranError,)
    from indexed_gzip import IndexedGzipFile  # type: ignore[import-not-found]
else:
    IndexedGzipFile = gzip.GzipFile

if HAVE_ZSTD:
    COMPRESSED_FILE_LIKES += (zstd.ZstdFile,)
    COMPRESSION_ERRORS += (zstd.ZstdError,)



class DeterministicGzipFile(gzip.GzipFile):
    """Deterministic variant of GzipFile

    This writer does not add filename information to the header, and defaults
    to a modification time (``mtime``) of 0 seconds.
    """

    def __init__(
        self,
        filename: str | None = None,
        mode: Mode | None = None,
        compresslevel: int = 9,
        fileobj: io.FileIO | None = None,
        mtime: int = 0,
    ):
        if mode is None:
            mode = 'rb'
        modestr: str = mode

        # These two guards are adapted from
        # https://github.com/python/cpython/blob/6ab65c6/Lib/gzip.py#L171-L174
        if 'b' not in modestr:
            modestr = f'{mode}b'
        if fileobj is None:
            if filename is None:
                raise TypeError('Must define either fileobj or filename')
            # Cast because GzipFile.myfileobj has type io.FileIO while open returns ty.IO
            fileobj = self.myfileobj = ty.cast('io.FileIO', open(filename, modestr))
        super().__init__(
            filename='',
            mode=modestr,
            compresslevel=compresslevel,
            fileobj=fileobj,
            mtime=mtime,
        )

def gzip_open(
    filename: str,
    mode: Mode = 'rb',
    compresslevel: int = 9,
    mtime: int = 0,
    keep_open: bool = False,
) -> gzip.GzipFile:
    if not HAVE_INDEXED_GZIP or mode != 'rb':
        gzip_file = DeterministicGzipFile(filename, mode, compresslevel, mtime=mtime)

    # use indexed_gzip if possible for faster read access.  If keep_open ==
    # True, we tell IndexedGzipFile to keep the file handle open. Otherwise
    # the IndexedGzipFile will close/open the file on each read.
    else:
        gzip_file = IndexedGzipFile(filename, drop_handles=not keep_open)

    return gzip_file


def zstd_open(
    filename: str,
    mode: Mode = 'r',
    *,
    level_or_option: int | dict | None = None,
    zstd_dict: zstd.ZstdDict | None = None,
) -> zstd.ZstdFile:
    return zstd.ZstdFile(filename, mode, level_or_option=level_or_option, zstd_dict=zstd_dict)

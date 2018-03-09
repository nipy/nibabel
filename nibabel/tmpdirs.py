# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Contexts for *with* statement providing temporary directories
'''
from __future__ import division, print_function, absolute_import
import os
import shutil
from tempfile import template, mkdtemp


class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.

    Upon exiting the context, the directory and everthing contained
    in it are removed.

    Examples
    --------
    >>> import os
    >>> with TemporaryDirectory() as tmpdir:
    ...     fname = os.path.join(tmpdir, 'example_file.txt')
    ...     with open(fname, 'wt') as fobj:
    ...         _ = fobj.write('a string\\n')
    >>> os.path.exists(tmpdir)
    False
    """

    def __init__(self, suffix="", prefix=template, dir=None):
        self.name = mkdtemp(suffix, prefix, dir)
        self._closed = False

    def __enter__(self):
        return self.name

    def cleanup(self):
        if not self._closed:
            shutil.rmtree(self.name)
            self._closed = True

    def __exit__(self, exc, value, tb):
        self.cleanup()
        return False


class InTemporaryDirectory(TemporaryDirectory):
    ''' Create, return, and change directory to a temporary directory

    Examples
    --------
    >>> import os
    >>> my_cwd = os.getcwd()
    >>> with InTemporaryDirectory() as tmpdir:
    ...     _ = open('test.txt', 'wt').write('some text')
    ...     assert os.path.isfile('test.txt')
    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))
    >>> os.path.exists(tmpdir)
    False
    >>> os.getcwd() == my_cwd
    True
    '''

    def __enter__(self):
        self._pwd = os.getcwd()
        os.chdir(self.name)
        return super(InTemporaryDirectory, self).__enter__()

    def __exit__(self, exc, value, tb):
        os.chdir(self._pwd)
        return super(InTemporaryDirectory, self).__exit__(exc, value, tb)


class InGivenDirectory(object):
    """ Change directory to given directory for duration of ``with`` block

    Useful when you want to use `InTemporaryDirectory` for the final test, but
    you are still debugging.  For example, you may want to do this in the end:

    >>> with InTemporaryDirectory() as tmpdir:
    ...     # do something complicated which might break
    ...     pass

    But indeed the complicated thing does break, and meanwhile the
    ``InTemporaryDirectory`` context manager wiped out the directory with the
    temporary files that you wanted for debugging.  So, while debugging, you
    replace with something like:

    >>> with InGivenDirectory() as tmpdir: # Use working directory by default
    ...     # do something complicated which might break
    ...     pass

    You can then look at the temporary file outputs to debug what is happening,
    fix, and finally replace ``InGivenDirectory`` with ``InTemporaryDirectory``
    again.
    """

    def __init__(self, path=None):
        """ Initialize directory context manager

        Parameters
        ----------
        path : None or str, optional
            path to change directory to, for duration of ``with`` block.
            Defaults to ``os.getcwd()`` if None
        """
        if path is None:
            path = os.getcwd()
        self.path = os.path.abspath(path)

    def __enter__(self):
        self._pwd = os.path.abspath(os.getcwd())
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        os.chdir(self.path)
        return self.path

    def __exit__(self, exc, value, tb):
        os.chdir(self._pwd)

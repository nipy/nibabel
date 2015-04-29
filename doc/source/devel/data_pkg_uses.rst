.. _data-pkg-uses:

########################################
Data package usecases and implementation
########################################

********
Usecases
********

We are here working from :doc:`data_pkg_discuss`

Prundles
========

See :ref:`prundle`.

An *local path* format prundle is a directory on the local file system with prundle data stored in files in a
on the local filesystem.

Examples
========

We'll call our package `dang` - data package new generation.

Create local-path prundle
-------------------------

::

    >>> import os
    >>> import tempfile
    >>> pth = tempfile.mkdtemp() # temporary directory

Make a pinstance object::

    >>> from dang import Pinstance
    >>> pri = Prundle(name='my-package')
    >>> pri.pkg_name
    'my-package'
    >>> pri.meta
    {}

Now we make a prundle.   First a directory to contain it::

    >>> import os
    >>> import tempfile
    >>> pth = tempfile.mkdtemp() # temporary directory

    >>> from dang.prundle import LocalPathPrundle
    >>> prun = LocalPathPrundle(pri, pth)

At the moment there's nothing in the directory.  The 'write' method will write
the meta information - here just the package name::

    >>> prun.write() # writes meta.ini file
    >>> os.listdir(pth)
    ['meta.ini']

The local path prundle data is just the set of files in the temporary directory
named in ``pth`` above.

Now we've written the package, we can get it by a single call that reads in the
``meta.ini`` file::

    >>> prun_back = LocalPathPrundle.from_path(pth)
    >>> prun_back.pkg_name
    'my-package'

Getting prundle data
--------------------

The file-system prundle formats can return content by file names.

For example, for the local path ``prun`` distribution objects we have seen so
far, the following should work::

    >>> fobj = prun.get_fileobj('a_file.txt')

In fact, local path distribution objects also have a ``path`` attribute::

    >>> fname = os.path.join(prun.path, 'a_file.txt')

The ``path`` attribute might not make sense for objects with greater
abstraction over the file-system - for example objects encapsulating web
content.

*********
Discovery
*********

So far, in order to create a prundle object, we have to know where the prundle
is (the path).

We want to be able to tell the system where prundles are - and the system will
then be able to return a prundle on request - perhaps by package name.  The
system here is answering a :ref:`prundle-discovery` query.

We will then want to ask our packaging system whether it knows about the
prundle we are interested in.

Discovery sources
=================

A discovery source is an object that can answer a discovery query.
Specifically, it is an object with a ``discover`` method, like this::

    >>> import dang
    >>> dsrc = dang.get_source('local-system')
    >>> dquery_result = dsrc.discover('my-package', version='0')
    >>> dquery_result[0].pkg_name
    'my-package'
    >>> dquery_result = dsrc.discover('implausible-pkg', version='0')
    >>> len(dquery_result)
    0

The discovery version number spec may allow comparison operators, as for
``distutils.version.LooseVersion``::

    >>> res = dsrc.discover(name='my-package', version='>=0')
    >>> prun = rst[0]
    >>> prun.pkg_name
    'my-package'
    >>> prun.meta['version']
    '0'

Default discovery sources
=========================

We've used the ``local-system`` discovery source in this call::

    >>> dsrc = dpkg.get_source('local-system')

The ``get_source`` function is a convenience function that returns default
discovery sources by name.  There are at least two named discovery sources,
``local-system``, and ``local-user``.  ``local-system`` is a discovery source
for packages that are installed system-wide (``/usr/share/data`` type
installation in \*nix).  ``local-user`` is for packages installed for this user
only (``/home/user/data`` type installations in \*nix).

Discovery source pools
======================

We'll typically have more than one source from which we'd like to query.  The
obvious case is where we want to look for both system and local sources.  For
this we have a *source pool* which simply returns the first known distribution
from a list of sources.  Something like this::

    >>> local_sys = dpkg.get_source('local-system')
    >>> local_usr = dpkg.get_source('local-user')
    >>> src_pool = dpkg.SourcePool((local_usr, local_sys))
    >>> dq_res = src_pool.discover('my-package', version='0')
    >>> dq_res[0].pkg_name
    'my-package'

We'll often want to do exactly this, so we'll add this source pool to those
that can be returned from our ``get_source`` convenience function::

    >>> src_pool = dpkg.get_source('local-pool')

Register a prundle
==================

In order to register a prundle, we need a prundle object and a
discovery source::

    >>> from dang.prundle import LocalPathPrundle
    >>> prun = LocalPathDistribution.from_path(path=/a/path')
    >>> local_usr = dang.get_source('local-user')
    >>> local_usr.register(prun)

Let us then write the source to disk::

    >>> local_usr.write()

Now, when we start another process as the same user, we can do this::

    >>> import dang
    >>> local_usr = dang.get_source('local-user')
    >>> prun = local_usr.discover('my-package', '0')[0]

**************
Implementation
**************

Here are some notes.  We had the hope that we could implement something that
would be simple enough that someone using the system would not need our code,
but could work from the specification.

Local path prundles
===================

These are directories accessible on the local filesystem.  The directory needs
to give information about the prundle name and optionally, version, tag,
revision id and maybe other metadata.  An ``ini`` file is probably enough for
this - something like a ``meta.ini`` file in the directory with::

    [DEFAULT]
    name = my-package
    version = 0

might be enough to get started.

Discovery sources
=================

The discovery source has to be able to return prundle objects for the
prundles it knows about::

    [my-package]
    0 = /some/path
    0.1 = /another/path
    [another-package]
    0 = /further/path

Registering a package
=====================

So far we have a local path distribution, that is a directory with some files
in it, and our own ``meta.ini`` file, containing the package name and version.
How does this package register itself to the default sources?  Of course, we
could use ``dpkg`` as above::

    >>> dst = dpkg.LocalPathDistribution.from_path(path='/a/path')
    >>> local_usr = dpkg.get_source('local-user')
    >>> local_usr.register(dst)
    >>> local_usr.save()

but we wanted to be able to avoid using ``dpkg``.  To do this, there might be
a supporting script, in the distribution directory, called ``register_me.py``,
of form given in :download:`register_me.py`.

Using discovery sources without dpkg
====================================

The local discovery sources are ini files, so it would be easy to read and use
these outside the dpkg system, as long as the locations of the ini files are
well defined.  Here is the code from ``register_me.py`` defining these files::

    import os
    import sys

    if sys.platform == 'win32':
        _home_dpkg_sdir = '_dpkg'
        _sys_drive, _ = os.path.splitdrive(sys.prefix)
    else:
        _home_dpkg_sdir = '.dpkg'
        _sys_drive = '/'
    # Can we get the user directory?
    _home = os.path.expanduser('~')
    if _home == '~': # if not, the user ini file is undefined
        HOME_INI = None
    else:
        HOME_INI = os.path.join(_home, _home_dpkg_sdir, 'local.dsource')
    SYS_INI = os.path.join(_sys_drive, 'etc', 'dpkg', 'local.dsource')

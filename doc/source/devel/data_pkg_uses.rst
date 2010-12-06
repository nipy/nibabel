.. _data-pkg-uses:

#####################
Data package usecases
#####################

This is a moderate conflation of usecases with some more design definitions and
discussions.

*************
Distributions
*************

Distribution definition
=======================

A distribution object is an object referring to a package distribution.  It
has a name and a version.  It may be able to return contents of the
distribution.  It identifies the type of the distribution.

Distribution types
==================

The most common type of distribution is a *file-system* distribution, that is,
a distribution that can return content from passed filenames, as if it were a
file-system.  The examples below should make this clearer.

An *local path* distribution is a file-system distribution with content stored
in files in a directory accessible on the local filesystem.  We could also call
this a *local path* distribution type.

Examples
========

Create distribution
-------------------

::

    >>> import os
    >>> import dpkg
    >>> dst = dpkg.LocalPathDistribution.initialize(name='my-package', path=/a/path')
    >>> dst.name
    'my-package'
    >>> dst.version
    '0'
    >>> dst.path == os.path.abspath(/a/path')
    True
    >>> os.listdir(/a/path')
    ['meta.ini']

The local path distribution here is just the set of files in the ``a/path`` directory.

The call to the ``initialize`` class method above creates the directory if it
does not exist, and writes a bare ``meta.ini`` file to the directory, with the
given ``name``, and default version of ``0``.

Use local path distribution
---------------------------

::

    >>> dst = dpkg.LocalPathDistribution.from_path(path=/a/path')
    >>> dst.name
    'my-package'

Getting content
---------------

The file-system distribution types can return content by file names.

For example, for the local path ``dst`` distribution objects we have seen so
far, the following should work::

    >>> fobj = dst.get_fileobj('a_file.txt')

In fact, local path distribution objects also have a ``path`` attribute::

    >>> fname = os.path.join(dst.path, 'a_file.txt')

The ``path`` attribute might not make sense for objects with greater abstraction
over the file-system - for example objects encapsulating web content.

*********
Discovery
*********

So far we only have distribution objects.  In order for a program to use a
distribution object it has to know where the distribution is.

We will then want to ask our packaging system whether it knows about the
distribution we are interested in.  This is a *discovery query*.

Discovery sources
=================

A discovery source is an object that can answer a discovery query.
Specifically, it is an object with a ``discover`` method, like this::

    >>> dsrc = dpkg.get_source('local-system')
    >>> dquery_result = dsrc.discover('my-package', version='0')
    >>> dquery_result.distribution.name
    'my-package'
    >>> dquery_result = dsrc.discover('implausible-pkg', version='0')
    >>> dquery_result.distribution is None
    True

The ``discover`` method returns a discovery query result.  This result contains
a distribution object if it knows about the distribution with the given name and
version; the distribution in the query is None otherwise.

The discovery version number spec may allow comparison operators, as for
``distutils.version.LooseVersion``::

    >>> res = dsrc.discover(name='my-package', version='>=0')
    >>> dst = rst.distribution
    >>> dst.name
    'my-package'
    >>> dst.version
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
    >>> dq_res.distribution.name
    'my-package'

We'll often want to do exactly this, so we'll add this source pool to those that
can be returned from our ``get_source`` convenience function::

    >>> src_pool = dpkg.get_source('local-pool')

Register a distribution
=======================

In order to register a distribution, we need a distribution object and a
discovery source::

    >>> dst = dpkg.LocalPathDistribution.from_path(path=/a/path')
    >>> local_usr = dpkg.get_source('local-user')
    >>> local_usr.register(dst)

Let us then write the source to disk::

    >>> local_usr.save()

Now, when we start another process as the same user, we can do this::

    >>> import dpkg
    >>> local_usr = dpkg.get_source('local-user')
    >>> dst = local_usr.discover('my-package', '0')

**************
Implementation
**************

Here are some notes.  We had the hope that we could implement something that
would be simple enough that someone using the system would not need our code,
but could work from the specification.  In practice we hope to be able get away
with something that uses ``ini`` format files as base storage, because these are
fairly standard and have support in the python standard library since way back.

Local path distributions
========================

As implied above, these are directories accessible on the local filesystem.
The directory needs to give information about the distribution name and version.
An ``ini`` file is probably enough for this - something like a ``meta.ini`` file
in the directory with::

    [DEFAULT]
    name = my-package
    version = 0

might be enough to get started.

Discovery sources
=================

The discovery source has to be able to return distribution objects for the
distributions it knows about.  A discovery source might only be able to handle
local path distributions, in which case all it needs to know about a
distribution is the (name, version, path).  So, a local path discovery source
could be stored on disk as an ``ini`` file as well::

    [my-package]
    0 = /some/path
    0.1 = /another/path
    [another-package]
    0 = /further/path

Registering a package
=====================

So far we have a local path distribution, that is a directory with some files in
it, and our own ``meta.ini`` file, containing the package name and version.  How
does this package register itself to the default sources?  Of course, we could
use ``dpkg`` as above::

    >>> dst = dpkg.LocalPathDistribution.from_path(path='/a/path')
    >>> local_usr = dpkg.get_source('local-user')
    >>> local_usr.register(dst)
    >>> local_usr.save()

but we wanted to be able to avoid using ``dpkg``.  To do this, there might be a
supporting script, in the distribution directory, called ``register_me.py``, of
form given in :download:`register_me.py`.

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

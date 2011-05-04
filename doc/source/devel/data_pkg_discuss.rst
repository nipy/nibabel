.. _data-package-discuss:

Principles of data package NG
=============================

Motivation
++++++++++

When developing or using nipy, many data files can be useful.

#. *small test data* - very small data files required for routine code testing.
   By small we mean less than 100K, and probably much less.  They have to be
   small because we keep them in the main code repository, and you therefore
   always get them with any download.
#. *large test data*.  These files can be much larger, and don't come in the
   standard repository.  We use them for tests, but we skip the tests if the
   data are not present.
#. *template data* - data files required for some algorithms to function,
   such as templates or atlases
#. *example data* - data files for running examples.

We need some standard way to provide the larger data sets.  To do this, we are
here defining the idea of a *data package*.  This document is a draft
specification of what a *data package* looks like and how to use it.

Separation of ideas
+++++++++++++++++++

This section needs some healthy beating to make the ideas clearer.  However, in
the interests of the 0SAGA_ software model, here are some ideas that may be
separable.

Package
-------

This ideas is rather difficult to define, but is a bit like a data project, that
is a set of information that the packager believed had something in common.  The
package then is an abstract idea, and what is in the package could change
completely over course of the life of the package.  The package then is a little
bit like a namespace, having itself no content other than a string (the package
name) and the data it contains.

Package name
------------

This is a string that gives a name to the package.

Package instantiation
---------------------

By *instantiation* I mean some particular actual set of data for a particular
package.  By actual, I mean stuff that can be read as bytes.  As we add and
remove data from the package, the *instantiation* changes.  This is the same
kind of idea as a *revision* in version control. An instantiation does not need
to be released, any more than a particular revision of some software needs to be
released.  This is what datapkg_ refers to a as a *distribution*.

Package instantiation identifier
--------------------------------

The *instantiation identifier* is a string that identifies a particular
instantiation of the data package.  This is the equivalent of the revision
number in subversion_, or the commit hash in newer systems like git_ or
mercurial_.

Package instantiation label
---------------------------

A *label* is a string that refers to particular state (instantiation) of the
package.  It will probably therefore also refer to a particular *instantiation
identifier*.  It is like a tag or a branch name in git_, that is, it is a
memorable string that refers to a state of the data.  An example might be a
numbered version.  So, a particular package may have an instantiation uniquely
identified by a hash ``af5bd6``.  We might decide to label this instantiation
``release-0.3`` (the equivalent of applying a git_ tag).  ``release-0.3`` is the
label and ``af5bd6`` is the identifier.

Package release
---------------

A release might be a package instantiation that one person has:

#. Labeled
#. Made available to other people

Label source
------------

Instantiation identifiers could for example be hashes of the package
instantiation (package contents), so they could be globally unique to the
contents.  *labels* are just names that someone has attached to a particular
identifier.   If there is more than one person providing versions of a
particular package, there may not be agreement on the identifier that a
particular label is attached to.  For example, I might think that
``release-0.3`` of ``some-package`` refers to package state identified by the
indentifier ``af5bd6``, but you might think that ``release-0.3`` of
``some-package`` refers to some other package state.  In this case you and are
are both a *label sources* for the package.  The state that particular label
refers to can depend then on the source from which the label came.

Package discovery
-----------------

We *discover* a package when we ask a system (local or remote) whether they have
a package at a given instantiation or range of instantiations.  That implies two
discoveries - *local discovery* (is the package instantiation on my local
system, if so where is it and how do I get it?); and *remote discovery* (is the
package instantiation on your expensive server and if so, where is it and how do
I get it?).  For the Debian distributions, the ``sources.list`` file identifies
sources from which we can query for software packages.  Those would be sources
for *remote discovery* in this language.

Package query
-------------

We query a package when we know that a particular system (local or remote) has
an instantiation of the package, and we want to get some information contained
in the package.

Package installation
--------------------

We install a package when we get some instantiation and place it on local
storage, such that we can *discover* the package on our own (local) system.

Data and metadata
-----------------

Data
    is the stuff contained in a particular package.

Metadata
    is data about the package itself.  It might include information about what
    data is in the database.

Registry
--------

Something that can be queried to *discover* a package instantiation.

Desiderata
++++++++++

We want to build a package system that is very simple ('S' in 0SAGA_).  For the
moment, the main problems we want to solve are: creation of a package
instantiation, installation of package instantiations, local discovery of
package instantiations.  For now we are not going to try and solve queries.

At least local discovery should be so simple that is can be implemented in any
language, and should not require a particular tool to be installed.  We hope we
can write a spec that makes all of (creation, installation, local discovery)
clearly defined, so that it would be simple to write an implementation.
Obviously we're going to end up writing our own implementation, or adapting
someone else's.  datapkg_ looks like the best candidate at the moment.

Issues
++++++

From a brief scan of the `debian package management documentation
<http://www.debian.org/doc/manuals/debian-reference/ch02.en.html>`_.

* Dependency management

Authentication and validation
+++++++++++++++++++++++++++++

* Authentication - using signatures to confirm that you made this package.
* Verification - verify that the contents have not been corrupted or changed
  since the original instantiation.

For dependency and validation, see the `Debian secure apt`_ page. One related
proposal would be:

.. _Debian secure apt: http://wiki.debian.org/SecureApt

* Each package instantiation would carry a table of checksums for the files
  within.  Someone using this instantiation would check the checksums to confirm
  that they had the intended content.
* Authentication would involve some kind of signing of the table of checksums,
  as in the ``Release.gpg`` file in Debian distributions (`Debian secure apt`_
  again).  This involves taking a checksum of the table of checksums, then using
  our trusted private key to encrypt this checksum, generating a digital
  signature.  The signature is the thing we provide to the user.  The user then
  gets our public key or has it already; they use the key to decrypt the
  signature to get the checksum, and they check the resulting checksum against
  the actual checksum of the checksum table.  The property of the public/private
  key pair is that it is very hard to go backwards. To explain, here's an
  example. Imagine someone we don't like has made a version of the package
  instantiation, but wants to persuade the world that we made it.  Their
  contents will have different checksums, and therefore a different checksum for
  the checksum table.  Let's say the checksum of the new checksum table is *X*.
  They know that you, the user, will use your own copy of our public key, and
  they can't get at that.  Their job then, is to make a new encrypted checksum
  (the signature) that will decrypt with our real public key, to equal *X*.
  That's going backwards from the desired result *X* to the signature, and that
  is very hard, if they don't have our private key.

Differences from code packages
++++++++++++++++++++++++++++++

The obvious differences are:

#. Data packages can be very large
#. We have less need for full history tracking (probably)

The size of data packages probably mean that using git_ itself will not work
well.  git_ stores (effectively) all previous versions of the files in the
repository, as zlib compressed blobs.  The working tree is an uncompressed
instantiation of the current state.  Thus, if we have, over time, had 4
different versions of a large file with little standard diff relationship to one
Discovery
another, the repository will have four zlib compressed versions of the file in
the ``.git/objects`` database, and one uncompressed version in the working tree.
The files in data packages may or may not compress well.

In contrast to the full git_ model, we may want to avoid duplicates of the data.
We probably won't by default want to keep all previous versions of the data
together at least locally.

We probably do want to be able to keep track of which files are the same across
different instantiations of the package, in the case where we already have one
instantiation on local disk, and we are asking for another, with some shared
files.  We might well want to avoid downloading duplicate data in that case.

Maybe the way to think of it is of the different costs that become important as
files get larger.  So the cost for holding a full history becomes very large,
whereas the benefit decreases a little bit (compared to code).

Some usecases
+++++++++++++

Discovery
---------

::
    from ourpkg import default_registry

    my_pkg_path = default_registry.pathfor('mypkg', '0.3')
    if mypkg_path is None:
        raise RuntimeError('It looks like mypkg version 0.3 is not installed')



.. include:: ../links_names.txt

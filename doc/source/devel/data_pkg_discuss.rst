.. _data-package-discuss:

##########################
Principles of data package
##########################

*******
Summary
*******

This is a discussion of data packages, as they are currently implemented in
nibabel / nipy.

This API proved to be very uncomfortable, and we intend to replace it fairly
soon.  See ``data_packages.rst`` in the `nibabel wiki`_ for our current
thinking, not yet implemented.

**********
Motivation
**********

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

*******************
Separation of ideas
*******************

This section needs some healthy beating to make the ideas clearer.  However, in
the interests of the 0SAGA_ software model, here are some ideas that may be
separable.

Package
=======

This idea is rather difficult to define, but is a bit like a data project, that
is a set of information that the packager believed had something in common.  The
package then is an abstract idea, and what is in the package could change
completely over course of the life of the package.  The package then is a little
bit like a namespace, having itself no content other than a string (the package
name) and the data it contains.

Package name
============

This is a string that gives a name to the package.

Package instantiation
=====================

By *instantiation* we mean some particular actual set of data for a particular
package.  By actual, we mean stuff that can be read as bytes.  As we add and
remove data from the package, the *instantiation* changes.  In version control,
the instantiation would be the particular state of the working tree at any
moment, whether this has been committed or not.

It might not be enjoyable, but we'll call a package instantiation a *pinstance*.

Pinstance revision
==================

A revision is an instantiation of the working tree that has a unique label - the
*revision id*.

Pinstance revision id
=====================

The *revision id* is a string that identifies a particular *pinstance*.  This is
the equivalent of the revision number in subversion_, or the commit hash in
systems like git_ or mercurial_. There is only one pinstance for any given
revision id, but there can be more than one revision id for a pinstance.  For
example, you might have a revision of id '200', delete a file, restore the file,
call this revision id '201', but they might both refer to the same instantiation
of the package.  Or they might not, that's up to you, the author of the package.

Pinstance tag
=============

A *tag* is a memorable string that refers to a particular pinstance.  It differs
from a revision id only in that there is not likely to be a tag for every
revision.  It's possible to imagine pinstances without a revision id but with a
tag, but perhaps it's reasonable to restrict tags to refer to revisions.  A
*tag* is equivalent to a tag name in git or mercurial - a memorable string that
refers to a static state of the data.  An example might be a numbered version.
So, a package may have a revision uniquely identified by a revision id
``af5bd6``.  We might decide to label this revision ``release-0.3`` (the
equivalent of applying a git tag).  ``release-0.3`` is the tag and ``af5bd6`` is
the revision id.  Different sources of the same package might possibly produce
different tags [#tag-sources]_

Pinstance version
=================

A *pinstance* might also have a version.  A version is just a tag that can be
compared using some algorithm.

.. _prundle:

Package provider bundle
=======================

Maybe we could call this a "prundle".

The *provider bundle* is something that can deliver the bytes of a particular
pinstance.  For example, if you have a package named "interesting-images", you
might have a revision of that package identified by revision id "f745dc2" and
tagged with "version-0.2".  There might be a *provider bundle* of that
instantiation that is a zipfile ``interesting-images-version-0.2.zip``.  There
might also be a directory on an http server with the same contents
``http://my.server.org/packages/interesting-images/version-9.2``.  The zipfile
and the http directory would both be *provider bundles* of the particular
instantiation.  When I unpack the zipfile onto my hard disk, I might have a
directory ``/my/home/packages/interesting-images/version-0.2``.  Now this path
is a provider bundle.

Provider bundle format
======================

In the example above, the zipfile, the http directory and the local path are
three different provider bundle formats delivering the same package
instantiation.  Let's call those formats:

* zipfile format
* url-path format
* local-path format

Pinstance release
=================

A release might be a package instantiation that one person has:

#. tagged
#. made available as one or more *provider bundles*

.. _prundle-discovery:

Prundle discovery
=================

We *discover* a package bundle when we ask a system (local or remote) whether
they have a package bundle at a given revision, tag, or bundle format.  That
implies two discoveries - *local discovery* (is the package bundle on my local
system, if so where is it?); and *remote discovery* (is the package bundle on
your expensive server and if so, how do I get it?).  For the Debian
distributions, the ``sources.list`` file identifies sources from which we can
query for software packages.  Those would be sources for *remote discovery* in
our language.

Prundle discovery source
========================

A *prundle discovery source* is somewhere that can answer prundle discovery
queries.

One such thing might be a prundle registry, where an element in the registry
contains information about a particular prundle.  At a first pass this might
contain:

* package name
* bundle format
* revision id (optional)
* tag (optional)

Maybe it should also contain information about where the information came from.

Pinstance metadata query
========================

We query a pinstance when we know that a particular system (local or remote) has
a package bundle of the pinstance we want. Then we get some information about
that pinstance.

By definition, different prundles relating to the same pinstance have the same
metadata.

Pinstance metadata query source
===============================

A *pinstance metadata query source* is somewhere that can answer pinstance
metadata queries.

Obviously a source may well be both a *prundle discovery source* and a
*pinstance metadata query source*.

Pinstance installation
======================

We install a pinstance when we get some prundle containing the pinstance and
place it on local storage, such that we can *discover* the prundle on our own
(local) system.  That is we take some prundle and convert it to a *local-path*
format bundle *and* we register this local-path format bundle to a *discovery
source*.

Data and metadata
=================

Pinstance data
    is the bytes as they are arranged in a particular pinstance.

Pinstance metadata
    is data about the pinstance.  It might include information about what data
    is in the package.

Prundle metadata
    Information about the particular prundle format.

***********************
Comparative terminology
***********************

In which we compare the package terminology above to the terminology of Debian
packaging.

Compared to Debian packaging
============================

* A Debian distribution is a label - such as 'unstable' or 'lenny' - that refers to a
  set of package revisions that go together.  We have no equivalent.
* A Debian *repository* is a set of packages within a distribution that go
  together - e.g. 'main' or 'contrib'.  We probably don't have an equivalent
  (unless we consider Debian's repository as being like a very large package
  in our language).
* A Debian source is a URI giving a location from which you can collect one or
  more repositories. For example, the line: "http://www.example.com/packages
  stable main contrib" in a "sources.list" file refers to the *source*
  "http://www.example.com/packages" providing *distribution* "stable" and
  *repositories* (within stable) of "main" and "contrib".  In our language the
  combination of URI, distribution and repository would refer to a *prundle
  discovery source* - that is - something that will answer queries about
  bundles.
* package probably means the same for us as for Debian - a name - like
  "python-numpy" - that refers to a set of files that go together and should be
  installed together.
* Debian packages have versions to reflect the different byte contents.  For
  example there might be a .deb file (see below) "some-package-0.11_3-i386.deb"
  for one distribution, and another (with different contents) for another
  distribution - say "some-package-0.12_9-i386.deb".  The "0.11_3" and "0.12_9"
  parts of the deb filename are what we would call *package instantiation tags*.
* A Debian deb file is an archive in a particular format that unpacks to provide
  the files for a particular package version.  We'd call the deb file a *package
  bundle*, that is in *bundle format* "deb-format".

**********
Desiderata
**********

We want to build a package system that is very simple ('S' in 0SAGA_).  For the
moment, the main problems we want to solve are: creation of a package
instantiation, installation of package instantiations, local discovery of
package instantiations.  For now we are not going to try and solve queries.

At least local discovery should be so simple that it can be implemented in any
language, and should not require a particular tool to be installed.  We hope we
can write a spec that makes all of (creation, installation, local discovery)
clearly defined, so that it would be simple to write an implementation.
Obviously we're going to end up writing our own implementation, or adapting
someone else's.  datapkg_ looks like the best candidate at the moment.

******
Issues
******

From a brief scan of the `debian package management documentation
<https://www.debian.org/doc/manuals/debian-reference/ch02.en.html>`_.

Dependency management
=====================

(no plan at the moment)

Authentication and validation
=============================

* Authentication - using signatures to confirm that you made this package.
* Verification - verify that the contents have not been corrupted or changed
  since the original instantiation.

For dependency and validation, see the `Debian secure apt`_ page. One related
proposal would be:

.. _Debian secure apt: https://wiki.debian.org/SecureApt

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

******************************
Differences from code packages
******************************

The obvious differences are:

#. Data packages can be very large
#. We have less need for full history tracking (probably)

The size of data packages probably mean that using git_ itself will not work
well.  git_ stores (effectively) all previous versions of the files in the
repository, as zlib compressed blobs.  The working tree is an uncompressed
instantiation of the current state.  Thus, if we have, over time, had 4
different versions of a large file with little standard diff relationship to one
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

*************
Some usecases
*************

Discovery
=========

::

    from ourpkg import default_registry

    my_pkg_path = default_registry.pathfor('mypkg', '0.3')
    if mypkg_path is None:
        raise RuntimeError('It looks like mypkg version 0.3 is not installed')


.. rubric:: Footnotes

.. [#tag-sources]

    Revsion ids could for example be hashes of the package instantiation
    (package contents), so they could be globally unique to the contents,
    whereever the contents was when the identifier was made.  However, *tags*
    are just names that someone has attached to a particular revsion id.  If
    there is more than one person providing versions of a particular package,
    there may not be agreement on the revsion that a particular tag is attached
    to.  For example, I might think that ``release-0.3`` of ``some-package``
    refers to package state identified by revsion id ``af5bd6``, but you might
    think that ``release-0.3`` of ``some-package`` refers to some other package
    state.  In this case you and are are both a *tag sources* for the package.
    The state that particular tag refers to can depend then on the source from
    which the tag came.

.. include:: ../links_names.txt

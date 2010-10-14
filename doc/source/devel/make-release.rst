.. _release-guide:

***********************************
A guide to making a nibabel release
***********************************

A guide for developers who are doing a nibabel release

* Edit :file:`info.py` and bump the version number

.. _release-tools::

Release tools
=============

In :file:`nibabel/tools`, among other files, you will find
the following utilities::

    nibabel/tools/
    |- build_release
    |- release
    |- compile.py
    |- make_tarball.py
    |- toollib.py

.. _release-checklist:

Release checklist
=================

* Make sure all tests pass.

* Review the open list of `issues <http://github.com/nipy/nibabel/issues>`_

* Run :file:`build_release` from the :file:`tools` directory::

    cd tools
    ./build_release

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors::

      git log "$@" | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  Then manually go over the *git log* to make sure the release notes are
  as complete as possible and that every contributor was recognized.

* Tag the release.

* Once everything looks good, run :file:`release` from the
  :file:`tools` directory.

* Announce to the mailing lists.

Releases
========

After each release the master branch should be tagged
with an annotated (or/and signed) tag, naming the intended
next version, plus an 'upstream/' prefix and 'dev' suffix.
For example 'upstream/1.0.0.dev' means "development start
for upcoming version 1.0.0.

This tag is used in the Makefile rules to create development
snapshot releases to create proper versions for those. The
will name the last available annotated tag, the number of
commits since that, and an abbrevated SHA1. See the docs of
``git describe`` for more info.

Please take a look at the Makefile rules ``devel-src``,
``devel-dsc`` and ``orig-src``.

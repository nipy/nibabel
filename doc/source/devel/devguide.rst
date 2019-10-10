.. -*- mode: rst; fill-column: 79 -*- .. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the NiBabel package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_devguide:

****************************
NiBabel Developer Guidelines
****************************

NiBabel source code
===================

.. toctree::
    :maxdepth: 2

    ../gitwash/index

Documentation
=============

Code Documentation
------------------

All documentation should be written using Numpy documentation conventions:

  https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard


Git Repository
==============

Layout
------

The main release branch is called ``master``. This is a merge-only branch.
Features finished or updated by some developer are merged from the
corresponding branch into ``master``. At a certain point the current state of
``master`` is tagged -- a release is done.

Only usable feature should end-up in ``master``. Ideally ``master`` should be
releasable at all times.

Additionally, there are distribution branches. They are prefixed ``dist/``
and labeled after the packaging target (e.g. *debian* for a Debian package).
If necessary, there can be multiple branches for each distribution target.

``dist/debian/proper``
  Official Debian packaging

``dist/debian/dev``
  Debian packaging of unofficial development snapshots. They do not go into the
  main Debian archive, but might be distributed through other channels (e.g.
  NeuroDebian).

Releases are merged into the packaging branches, packaging is updated if
necessary and the branch gets tagged when a package version is released.
Maintenance (as well as backport) releases or branches off from the respective
packaging tag.

There might be additonal branches for each developer, prefixed with intials.
Alternatively, several GitHub (or elsewhere) clones might be used.


Commits
-------

Please prefix all commit summaries with one (or more) of the following labels.
This should help others to easily classify the commits into meaningful
categories:

  * *BF* : bug fix
  * *RF* : refactoring
  * *NF* : new feature
  * *BW* : addresses backward-compatibility
  * *OPT* : optimization
  * *BK* : breaks something and/or tests fail
  * *PL* : making pylint happier
  * *DOC*: for all kinds of documentation related commits
  * *TEST*: for adding or changing tests

Merges
------

For easy tracking of what changes were absorbed during merge, we
advise that you enable merge summaries within git:

  git-config merge.summary true

See :ref:`configure-git` for more detail.

Changelog
=========

The changelog is located in the toplevel directory of the source tree in the
`Changelog` file. The content of this file should be formated as restructured
text to make it easy to put it into manual appendix and on the website.

This changelog should neither replicate the VCS commit log nor the
distribution packaging changelogs (e.g. debian/changelog). It should be
focused on the user perspective and is intended to list rather macroscopic
and/or important changes to the module, like feature additions or bugfixes in
the algorithms with implications to the performance or validity of results.

It may list references to 3rd party bugtrackers, in case the reported bugs
match the criteria listed above.

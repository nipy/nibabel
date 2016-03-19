.. _release-guide:

***********************************
A guide to making a nibabel release
***********************************

This is a guide for developers who are doing a nibabel release.

The general idea of these instructions is to go through the following steps:

* Make sure that the code is in the right state for release;
* update release-related docs such as the Changelog;
* update various documents giving dependencies, dates and so on;
* check all standard and release-specific tests pass;
* make the *release commit* and release tag;
* check Windows binary builds and slow / big memory tests;
* push source and windows builds to pypi;
* push docs;
* push release commit and tag to github;
* announce.

We leave pushing the tag to the last possible moment, because it's very bad
practice to change a git tag once it has reached the public servers (in our
case, github).  So we want to make sure of the contents of the release before
pushing the tag.

.. _release-checklist:

Release checklist
=================

* Review the open list of `nibabel issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Review and update the release notes.  Review and update the ``Changelog``
  file.  Get a partial list of contributors with something like::

      git log 2.0.0.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``2.0.0`` was the last release tag name.

  Then manually go over ``git shortlog 2.0.0..`` to make sure the release
  notes are as complete as possible and that every contributor was recognized.

* Look at ``doc/source/index.rst`` and add any authors not yet acknowledged.

* Update new authors and add thansk in ``doc/source/index.rst`` and consider
  any updates to the ``AUTHOR`` file.

* Use the opportunity to update the ``.mailmap`` file if there are any
  duplicate authors listed from ``git shortlog -nse``.

* Check the copyright year in ``doc/source/conf.py``

* Refresh the ``REAMDE.rst`` text from the ``LONG_DESCRIPTION`` in ``info.py``
  by running ``make refresh-readme``.

  Check the output of::

    rst2html.py README.rst > ~/tmp/readme.html

  because this will be the output used by pypi_

* Check the dependencies listed in ``nibabel/info.py`` (e.g.
  ``NUMPY_MIN_VERSION``) and in ``doc/source/installation.rst``.  They should
  at least match. Do they still hold?  Make sure `nibabel on travis`_ is
  testing the minimum dependencies specifically.

* Do a final check on the `nipy buildbot`_.  Use the ``try_branch.py``
  scheduler available in nibotmi_ to test particular schedulers.

* If you have travis-ci_ building set up for your own repo you might want to
  push the code in it's current state to a branch that will build, e.g::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test
    git push your-github-user pre-release-test -u

* Clean::

    make distclean

* Make sure all tests pass (from the nibabel root directory)::

    nosetests --with-doctest nibabel

* Make sure all tests pass from sdist::

    make sdist-tests

  and the three ways of installing (from tarball, repo, local in repo)::

    make check-version-info

  The last may not raise any errors, but you should detect in the output
  lines of this form::

    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'archive substitution', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nibabel', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nibabel/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'installation', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nibabel', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /Users/mb312/dev_trees/nibabel/nibabel/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'repository', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/Users/mb312/dev_trees/nibabel/nibabel', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}

* Check the ``setup.py`` file is picking up all the library code and scripts,
  with::

    make check-files

  Look for output at the end about missed files, such as::

    Missed script files:  /Users/mb312/dev_trees/nibabel/bin/nib-dicomfs, /Users/mb312/dev_trees/nibabel/bin/nifti1_diagnose.py

  Fix ``setup.py`` to carry across any files that should be in the
  distribution.

* You may have virtualenvs for different Python versions.  Check the tests
  pass for different configurations. The long-hand way looks like this::

    workon python26
    make distclean
    make sdist-tests
    deactivate

  etc for the different virtualenvs.

* Check on different platforms, particularly windows and PPC. Look at the
  `nipy buildbot`_ automated test runs for this.

* Check the documentation doctests::

    make -C doc doctest

  This should also be tested by `nibabel on travis`_.

* Check everything compiles without syntax errors::

    python -m compileall .

* Edit ``nibabel/info.py`` to set ``_version_extra`` to ``''``; commit.
  Then::

    make source-release

* Make sure you are set up to use the ``try_branch.py`` - see
  https://github.com/nipy/nibotmi/blob/master/install.rst#trying-a-set-of-changes-on-the-buildbots

* Make sure all your changes are committed or removed, because
  ``try_branch.py`` pushes up the changes in the working tree;

* Force build of your release candidate branch with the slow and big-memory
  tests on the ``zibi`` buildslave::

    try_branch.py nibabel-py2.7-osx-10.10

  Check the build web-page for errors:

  * https://nipy.bic.berkeley.edu/builders/nibabel-py2.7-osx-10.10

* Force builds of your local branch on the win32 and amd64 binaries on
  buildbot::

    try_branch.py nibabel-bdist32-27
    try_branch.py nibabel-bdist32-34
    try_branch.py nibabel-bdist32-35
    try_branch.py nibabel-bdist64-27

  Check the builds completed without error on their respective web-pages:

  * https://nipy.bic.berkeley.edu/builders/nibabel-bdist32-27
  * https://nipy.bic.berkeley.edu/builders/nibabel-bdist32-34
  * https://nipy.bic.berkeley.edu/builders/nibabel-bdist32-35
  * https://nipy.bic.berkeley.edu/builders/nibabel-bdist64-27

  Then get the built binaries in:

  * https://nipy.bic.berkeley.edu/nibabel-dist

  When you've done the release to pypi, you can upload them to pypi with the
  admin files interface.

  If you are already on a Windows machine, you could have done the manual
  command to build instead: ``python setup.py bdist_wininst``.

* Once everything looks good, you are ready to upload the source release to
  PyPi.  See `setuptools intro`_.  Make sure you have a file
  ``\$HOME/.pypirc``, of form::

    [distutils]
    index-servers =
        pypi

    [pypi]
    username:your.pypi.username
    password:your-password

    [server-login]
    username:your.pypi.username
    password:your-password

* When ready::

    python setup.py register
    python setup.py sdist --formats=gztar,zip upload

* Tag the release with tag of form ``2.0.0``::

    git tag -am "Something about this release' 2.0.0

* Push the tag and any other changes to trunk with::

    git push --tags

* Now the version number is OK, push the docs to github pages with::

    make upload-html

* Finally (for the release uploads) upload the Windows binaries you built with
  ``try_branch.py`` above;

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/2.0.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say -
    '2.0.1.dev' until the next maintenance release - say '2.0.1'.  Commit.
    Don't forget to push upstream with something like::

      git push upstream-remote maint/2.0.x --set-upstream

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor``
    by 1.  Thus the development series ('trunk') will have a version number
    here of '2.1.0.dev' and the next full release will be '2.1.0'.

    Next merge the maintenance branch with the "ours" strategy.  This just
    labels the maintenance `info.py` edits as seen but discarded, so we can
    merge from maintenance in future without getting spurious merge conflicts::

       git merge -s ours maint/2.0.x


  If this is just a maintenance release from ``maint/2.0.x`` or similar, just
  tag and set the version number to - say - ``2.0.2.dev``.

* Push the main branch::

    git push upstream-remote main-master

* Make next development release tag

    After each release the master branch should be tagged
    with an annotated (or/and signed) tag, naming the intended
    next version, plus an 'upstream/' prefix and 'dev' suffix.
    For example 'upstream/1.0.0.dev' means "development start
    for upcoming version 1.0.0.

    This tag is used in the Makefile rules to create development snapshot
    releases to create proper versions for those. The version derives its name
    from the last available annotated tag, the number of commits since that,
    and an abbreviated SHA1. See the docs of ``git describe`` for more info.

    Please take a look at the Makefile rules ``devel-src``, ``devel-dsc`` and
    ``orig-src``.

* Announce to the mailing lists.

.. _setuptools intro: https://pythonhosted.org/an_example_pypi_project/setuptools.html

.. include:: ../links_names.txt

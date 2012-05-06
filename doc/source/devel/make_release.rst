.. _release-guide:

***********************************
A guide to making a nibabel release
***********************************

A guide for developers who are doing a nibabel release

.. _release-tools:

Release tools
=============

There are some release utilities that come with nibabel_.  nibabel should
install these as the ``nisext`` package, and the testing stuff is understandably
in the ``testers`` module of that package.  nibabel has Makefile targets for their
use.  The relevant targets are::

    make check-version-info
    make sdist-tests

The first installs the code from a git archive, from the repository, and for
in-place use, and runs the ``get_info()`` function to confirm that installation
is working and information parameters are set correctly.

The second (``sdist-tests``) makes an sdist source distribution archive,
installs it to a temporary directory, and runs the tests of that install.

If you have a version of nibabel trunk past February 11th 2011, there will also
be a functional make target::

    make bdist-egg-tests

This builds an egg (which is a zip file), hatches it (unzips the egg) and runs
the tests from the resulting directory.

.. _release-checklist:

Release checklist
=================

* Review the open list of `nibabel issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git log 1.2.0.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``1.2.0`` was the last release tag name.

  Then manually go over ``git shortlog 1.2.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any duplicate
  authors listed from ``git shortlog``.

* Check the ``long_description`` in ``nibabel/info.py``.  Check it matches the
  ``README`` in the root directory.

* Do a final check on the `nipy buildbot`_

* If you have travis-ci_ building set up you might want to push the code in it's
  current state to a branch that will build, e.g::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test
    git push origin pre-release-test

* Clean::

    make distclean

* Make sure all tests pass (from the nibabel root directory)::

    cd ..
    nosetests --with-doctest nibabel
    cd nibabel # back to the root directory

* Make sure all tests pass from sdist::

    make sdist-tests

  and bdist_egg::

    make bdist-egg-tests

  and the three ways of installing (from tarball, repo, local in repo)::

    make check-version-info

  The last may not raise any errors, but you should detect in the output
  lines of this form::

    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'archive substitution', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nibabel', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nibabel/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'installation', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nibabel', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    Files not taken across by the installation:
    []
    /Users/mb312/dev_trees/nibabel/nibabel/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'repository', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/Users/mb312/dev_trees/nibabel/nibabel', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}

* You probably have virtualenvs for different python versions.  Check the tests
  pass for different configurations.  If you have pytox_ and a network
  connnection, and lots of pythons installed, you might be able to do::

    tox

  and get tests for python 2.5, 2.6, 2.7, 3.2.  I (MB) have my own set of
  virtualenvs installed and I've set them up to run with::

    tox -e python25,python26,python27,python32,np-1.2.1

  The trick was only to define these ``testenv`` sections in ``tox.ini``.

  These two above run with::

    make tox-fresh
    make tox-stale

  respectively.

  The long-hand not-tox way looks like this::

    workon python26
    make sdist-tests
    deactivate

  etc for the different virtualenvs.

* Check on different platforms, particularly windows and PPC.  I have wine
  installed on my Mac, and git bash installed under wine.  I run bash and the
  tests like this::

    wineconsole bash
    # in wine bash
    make sdist-tests

  For the PPC I have to log into an old Mac G5 in Berkeley at
  ``jerry.bic.berkeley.edu``.  Here's an example session::

    ssh jerry.bic.berkeley.edu
    cd dev_trees/nibabel
    git co main-master
    git pull
    make sdist-tests

* Check the documentation doctests::

    cd doc
    make doctest
    cd ..

* Check everything compiles without syntax errors::

    python -m compileall .

* The release should now be ready.

* Edit :file:`nibabel/info.py` to set ``_version_extra`` to ``''``; commit.
  Then::

    make source-release

* Once everything looks good, upload the source release to PyPi.  See
  `setuptools intro`_::

    python setup.py register
    python setup.py sdist --formats=gztar,zip upload

  From somewhere - maybe a windows machine - upload the windows installer for
  convenience::

    python setup.py bdist_wininst upload

* Tag the release with tag of form ``1.1.0``::

    git tag -am 'Second main release' 1.1.0

* Now the version number is OK, push the docs to sourceforge with::

    make upload-htmldoc-mysfusername

  where ``mysfusername`` is obviously your own sourceforge username.

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/1.0.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '1.0.1.dev'
    until the next maintenance release - say '1.0.1'.  Commit. Don't forget to
    push upstream with something like::

      git push upstream maint/1.0.0 --set-upstream

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor`` by 1.
    Thus the development series ('trunk') will have a version number here of
    '1.1.0.dev' and the next full release will be '1.1.0'.

  If this is just a maintenance release from ``maint/1.0.x`` or similar, just
  tag and set the version number to - say - ``1.0.2.dev``.

* Push tags::

    git push --tags

* Make next development release tag

    After each release the master branch should be tagged
    with an annotated (or/and signed) tag, naming the intended
    next version, plus an 'upstream/' prefix and 'dev' suffix.
    For example 'upstream/1.0.0.dev' means "development start
    for upcoming version 1.0.0.

    This tag is used in the Makefile rules to create development snapshot
    releases to create proper versions for those. The version derives its name
    from the last available annotated tag, the number of commits since that, and
    an abbreviated SHA1. See the docs of ``git describe`` for more info.

    Please take a look at the Makefile rules ``devel-src``,
    ``devel-dsc`` and ``orig-src``.

* Announce to the mailing lists.

.. _pytox: http://codespeak.net/tox
.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html
.. _travis-ci: http://travis-ci.org

.. include:: ../links_names.txt

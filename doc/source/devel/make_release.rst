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
  You might want to use the following to list authors by the date of their
  contributions::

    git log --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'

  (From:
  http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit#6482473)

  Consider any updates to the ``AUTHOR`` file.

* Use the opportunity to update the ``.mailmap`` file if there are any
  duplicate authors listed from ``git shortlog -nse``.

* Refresh the ``CITATION.cff`` file::

    uv run tools/prep_citation_cff.py

* Refresh the ``README.rst`` text from the ``LONG_DESCRIPTION`` in ``info.py``
  by running ``make refresh-readme``.

  Check the output of::

    rst2html.py README.rst > ~/tmp/readme.html

  because this will be the output used by pypi_

* Check the dependencies listed in ``pyproject.toml`` and in
  ``doc/source/installation.rst``. They should at least match. Do
  they still hold?  Verify that `nibabel on GitHub actions`_ is testing the minimum
  dependencies specifically.

* Make sure all tests pass (from the nibabel root directory)::

    tox

* Make sure you have travis-ci_ building set up for your own repo. Make a new
  ``release-check`` (or similar) branch, and push the code in its current
  state to a branch that will build, e.g::

    git branch -D release-check # in case branch already exists
    git co -b release-check
    # You might need the --force flag here
    git push your-github-user release-check -u

* Once everything looks good, you are ready to tag and release::

    git tag -s 2.0.0

* Push the tag and any other changes to trunk with::

    git push origin master --tags

* Now the version number is OK, push the docs to github pages with::

    make upload-html

* Set up maintenance / development branches

  If this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/2.0.x

      git push upstream-remote maint/2.0.x --set-upstream

  * Start next development series::

      git co master

       git merge -s ours maint/2.0.x

  If this is just a maintenance release from ``maint/2.0.x`` or similar, just
  tag and set the version number to - say - ``2.0.2.dev``.

* Push the main branch::

    git push upstream-remote master

* Go to: https://github.com/nipy/nibabel/tags and select the new tag, to fill
  in the release notes.  Copy the relevant part of the Changelog into the
  release notes.  Click on "Publish release".  This will cause Zenodo_ to
  generate a new release "upload", including a DOI.  After a few minutes, go
  to https://zenodo.org/deposit and click on the new release upload.  Click on
  the "View" button and click on the DOI badge at the right to display the
  text for adding a DOI badge in various formats. Copy the DOI Markdown text.
  The markdown will look something like this::

    [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.60847.svg)](https://doi.org/10.5281/zenodo.60847)

  Go back to the Github release page for this release, click "Edit release".
  and copy the DOI into the release notes.  Click "Update release".

  See: https://guides.github.com/activities/citable-code

* Announce to the mailing lists.

.. include:: ../links_names.txt

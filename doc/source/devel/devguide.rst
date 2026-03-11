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

Also see :ref:`devindex`

NiBabel source code
===================

.. toctree::
    :maxdepth: 2

    ../gitwash/index

Documentation
=============

Code Documentation
------------------

Please write documentation using Numpy documentation conventions:

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

There might be additional branches for each developer, prefixed with initials.
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

Testing
=======

NiBabel uses tox_ to organize our testing and development workflows.
tox runs tests in isolated environments that we specify,
ensuring that we are able to test across many different environments,
and those environments do not depend on our local configurations.

The simplest way to install the necessary tools is to run::

    uv sync --group=dev

This will install tox_ and pre-commit_ with uv_ plugins into a local
`.venv` environment.

Otherwise, you may install tox using `uv tool`::

    uv tool install --with=tox-uv-bare tox

Or in any environment with::

    python -m pip install tox
    tox

This will run the tests in several configurations, with multiple sets of
optional dependencies.
If you have multiple versions of Python installed in your path, it will
repeat the process for each version of Python iin our supported range.
It may be useful to pick a particular version for rapid development::

    tox -e py311-full-x64

This will run the environment using the Python 3.11 interpreter, with the
full set of optional dependencies that are available for 64-bit
interpreters. If you are using 32-bit Python, replace ``-x64`` with ``-x86``.


Style guide
===========

To ensure code consistency and readability, NiBabel has adopted the following
tools:

* ruff_ - A general linter and autoformatter.
* codespell_ - A spell checker targeted at source code.
* pre-commit_ - A pre-commit hook manager that runs the above and various
  other checks/fixes.

To apply our style checks uniformly, simply run::

    pre-commit run --all-files

Individual checks can be run with::

    pre-commit run ruff-check --all-files
    pre-commit run ruff-format --all-files
    pre-commit run codespell --all-files

While some amount of personal preference is involved in selecting and
configuring auto-formatters, their value lies in largely eliminating the
need to think or argue about style.
With pre-commit turned on, you can write in the style that works for you,
and the NiBabel style will be adopted prior to the commit.

Ruff implements a large number of checks.
Many of them may not be applicable to our code base,
or may produce a large amount of churn for little gain.
Checks may be added or disabled, but the effort may outweigh the benefits.
Please make an affirmative argument for each change,
including an example diff.

Occasionally, codespell has a false positive. To ignore the suggestion, add
the intended word to ``tool.codespell.ignore-words-list`` in ``pyproject.toml``.
However, the ignore list is a blunt instrument and could cause a legitimate
misspelling to be missed. Consider choosing a word that does not trigger
codespell before adding it to the ignore list.

Pre-commit hooks
----------------

NiBabel uses pre-commit_ to help committers validate their changes
before committing.

Install the pre-commit hooks with::

    pre-commit install

If pre-commit is not already installed in your environment,
use one of the following::

    uv sync --group=dev
    # or
    uv tool install --with=pre-commit-uv-bare pre-commit
    # or
    python -m pip install pre-commit

Changelog
=========

The changelog is located in the toplevel directory of the source tree in the
`Changelog` file. The content of this file should be formatted as restructured
text to make it easy to put it into manual appendix and on the website.

This changelog should neither replicate the VCS commit log nor the
distribution packaging changelogs (e.g. debian/changelog). It should be
focused on the user perspective and is intended to list rather macroscopic
and/or important changes to the module, like feature additions or bugfixes in
the algorithms with implications to the performance or validity of results.

It may list references to 3rd party bugtrackers, in case the reported bugs
match the criteria listed above.

.. _mission_and_values:

.. _community_guidelines:

.. _code_of_conduct:

Community guidelines
====================

Please see `our community guidelines
<https://github.com/nipy/nibabel/blob/master/.github/CODE_OF_CONDUCT.md>`_.
Other projects call these guidelines the "code of conduct".

.. _ruff: https://docs.astral.sh/ruff/
.. _uv: https://docs.astral.sh/uv/
.. _codespell: https://github.com/codespell-project/codespell
.. _pre-commit: https://pre-commit.com/
.. _tox: https://tox.wiki/

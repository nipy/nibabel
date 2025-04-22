.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the NiBabel package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _installation:

************
Installation
************

NiBabel is a pure Python package,
and it should be easy to get NiBabel running on any system.
For the most popular platforms and operating systems
there should be packages in the respective native packaging format
(DEB, RPM or installers).
On other systems you can install NiBabel using pip_.

.. This remark below is not yet true; comment to avoid confusion
   To run all of the tests, you may need some extra data packages - see
   :ref:`installing-data`.

Installer and packages
======================

.. _install-pypi:

pip and the Python package index
--------------------------------

If you are not using a Linux package manager, then best way to install NiBabel
is via pip_.  If you don't have pip already, follow the `pip install
instructions`_.

Then open a terminal (``Terminal.app`` on OSX, ``cmd`` or ``Powershell`` on
Windows), and type::

    pip install nibabel

This will download and install NiBabel.

If you really like doing stuff manually, you can install NiBabel by downloading
the source from `NiBabel pypi`_ .  Go to the pypi page and select the source
distribution you want.  Download the distribution, unpack it, and then, from
the unpacked directory, run::

    pip install .

If you get permission errors, this may be because ``pip`` is trying to install
to the system directories.  You can solve this error by using ``sudo``, but we
strongly suggest you either do an install into your "user" directories, like
this::

    pip install --user .

or you work inside a virtualenv_.

.. _install_debian:

Debian/Ubuntu
-------------

Our friends at NeuroDebian_ have packaged NiBabel at `NiBabel NeuroDebian`_.
Please follow the instructions on the NeuroDebian_ website on how to access
their repositories. Once this is done, installing NiBabel is::

  apt-get update
  apt-get install python-nibabel

Install a development version
=============================

If you want to test the latest development version of nibabel, or you'd like to
help by contributing bug-fixes or new features (excellent!), then this section
is for you.

Requirements
------------

.. check these against pyproject.toml / tox.ini

*  Python_ 3.9 or greater
*  NumPy_ 1.22 or greater
*  Packaging_ 20.0 or greater
*  importlib-resources_ 5.12 or greater (or Python 3.12+)
*  SciPy_ 1.8 or greater (optional, for full SPM-ANALYZE support)
*  h5py_ 3.5 or greater (optional, for MINC2 support)
*  PyDICOM_ 2.3.0 or greater (optional, for DICOM support)
*  `Python Imaging Library`_ 8.4 or greater (optional, for PNG conversion in DICOMFS)
*  pytest_ (optional, to run the tests)
*  sphinx_ (optional, to build the documentation)

Get the development sources
---------------------------

You can download a tarball of the latest development snapshot (i.e. the current
state of the *master* branch of the NiBabel source code repository) from the
`NiBabel github`_ page.

If you want to have access to the full NiBabel history and the latest
development code, do a full clone (AKA checkout) of the NiBabel
repository::

  git clone https://github.com/nipy/nibabel.git

Installation
------------

Just install the modules by invoking::

  pip install .

See :ref:`install-pypi` for advice on what to do for permission errors.

Validating your install
-----------------------

For a basic test of your installation, fire up Python and try importing the
module to see if everything is fine.  It should look something like this::

    Python 3.8.5 (default, Sep  4 2020, 07:30:14)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import nibabel
    >>>


To run the nibabel test suite, from the terminal run
``pytest --pyargs nibabel`` or
``python -c "import nibabel; nibabel.test()``.

To run an extended test suite that validates ``nibabel`` for long-running and
resource-intensive cases, please see :ref:`advanced_testing`.

.. include:: links_names.txt

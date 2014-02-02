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

NiBabel is a pure python package at the moment, and it should be easy to get
NiBabel running on any system. For the most popular platforms and operating
systems there should be packages in the respective native packaging format
(DEB, RPM or installers). On other systems you can install NiBabel using
``easy_install`` or by downloading the source package and running the usual
``python setup.py install``.

.. This remark below is not yet true; comment to avoid confusion
   To run all of the tests, you may need some extra data packages - see
   :ref:`installing-data`.

Installer and packages
======================

.. _install_pypi:

The python package index
------------------------

NiBabel is available via `pypi`_.  If you already have setuptools_ or
distribute_ installed, you can run::

    easy_install nibabel

to download nibabel and its dependencies.  Alternatively go to the `nibabel
pypi`_ page and select the source distribution you want.  Download the
distribution, unpack it, and then, from the unpacked directory, run::

    python setup.py install

or (if you need root permission to install on a unix system)::

    sudo python setup.py install

.. _install_debian:

Debian/Ubuntu
-------------

NiBabel is available as a `NeuroDebian package`_. Please follow the instructions
on the NeuroDebian_ website on how access their repositories. Once this is done,
installing NiBabel is::

  apt-get update
  apt-get install python-nibabel

.. _NeuroDebian package: http://neuro.debian.net/pkgs/python-nibabel.html

Install from source
===================

If no installer or package is provided for your platfom, you can install
NiBabel from source.

Requirements
------------

*  Python_ 2.6 or greater
*  NumPy_ 1.2 or greater
*  SciPy_ (for full SPM-ANALYZE support)
*  PyDICOM_ 0.9.7 or greater (for DICOM support)
*  `Python Imaging Library`_ (for PNG conversion in DICOMFS)
*  nose_ 0.11 or greater (to run the tests)
*  sphinx_ (to build the documentation)

Get the sources
---------------

The latest release is always available from `nibabel pypi`_.

Alternatively, you can download a tarball of the latest development snapshot
(i.e. the current state of the *master* branch of the NiBabel source code
repository) from the `nibabel github`_ page.

If you want to have access to the full NiBabel history and the latest
development code, do a full clone (aka checkout) of the NiBabel
repository::

  git clone git://github.com/nipy/nibabel.git

or::

  git clone http://github.com/nipy/nibabel.git

(The first will be faster, the second more likely to work behind a firewall).

Installation
------------

Just install the modules by invoking::

  sudo python setup.py install

If sudo is not configured (or even installed) you might have to use
``su`` instead.

Now fire up Python and try importing the module to see if everything is fine.
It should look similar to this::


    Python 2.7.3 (v2.7.3:70274d53c1dd, Apr  9 2012, 20:52:43)
    [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import nibabel
    >>>

.. include:: links_names.txt

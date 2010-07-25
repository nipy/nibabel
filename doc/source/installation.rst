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

It should be easy to get NiBabel running on any system. For the most popular
platforms and operating systems there are binary packages provided in the
respective native packaging format (DEB, RPM or installers). On all other
systems NiBabel has to be compiled from source -- which should also be pretty
straightforward.


Installer and packages
======================

.. _install_debian:

Debian/Ubuntu
-------------

NiBabel is available as a `NeuroDebian package`_. Please follow the instructions
on the NeuroDebian_ website on how access their repositories. Once this is done,
installing NiBabel is::

  apt-get update
  apt-get install python-nibabel

.. _NeuroDebian package: http://neuro.debian.net/pkgs/python-nibabel-snapshot.html



Install from source
===================

If no installer or package is provided for your platfom, you can install
NiBabel from source. It needs a few things to run properly:

*  Python_ 2.5 or greater
*  NumPy_
*  SciPy_ (for SPM-ANALYZE support)


Get the sources
---------------

The latest release is always available from the `SourceForge download page`_.

Alternatively, you can download a tarball of the latest development snapshot
(i.e. the current state of the *master* branch of the NiBabel source code
repository) from the `nibabel github`_ page.

If you want to have access to the full NiBabel history and the latest
development code, do a full clone (aka checkout) of the NiBabel
repository::

  git clone http://github.com/nipy/nibabel.git


Installation
------------

Just install the modules by invoking::

  sudo python setup.py install

If sudo is not configured (or even installed) you might have to use
``su`` instead.

Now fire up Python and try importing the module to see if everything is fine.
It should look similar to this::

  Python 2.5.5 (r255:77872, Apr 21 2010, 08:44:16) 
  [GCC 4.4.3] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import nibabel
  >>>


.. include:: links_names.txt

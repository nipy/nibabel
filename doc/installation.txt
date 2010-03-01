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

It should be easy to get NiBabel running on any system. For the most
popular platforms and operating systems there are binary packages provided in
the respective native packaging format (DEB, RPM or installers). On all other
systems NiBabel has to be compiled from source -- which should also be pretty
straightforward.


Binary packages
===============

.. _install_debian:

GNU/Linux
---------

NiBabel is available in recent versions of the Debian (since lenny) and
Ubuntu (since gutsy in universe) distributions. The name of the binary package
is ``python-nibabel`` in both cases.

* NiBabel `versions in Debian`_

* NiBabel `versions in Ubuntu`_

Binary packages for some additional Debian and (K)Ubuntu versions are also
available. Please visit `Michael Hanke's APT repository`_ to read about how
you have to setup your system to retrieve the NiBabel package via your package
manager and stay in sync with future releases.

If you are using Debian lenny (or later) or Ubuntu gutsy (or later) or you
have configured your system for `Michael Hanke's APT repository`_ all you
have to do to install NiBabel is this::

  apt-get update
  apt-get install python-nibabel

This should pull all necessary dependencies. If it doesn't, it's a bug that
should be reported.

.. _versions in Debian: http://packages.debian.org/python-nifti
.. _versions in Ubuntu: http://packages.ubuntu.com/python-nifti
.. _Michael Hanke's APT repository: http://apsy.gse.uni-magdeburg.de/main/index.psp?page=hanke/debian&lang=en&sec=1


.. _install_rpm:

Additionally, there are binary packages for several RPM-based distributions,
provided through the `OpenSUSE Build Service`_. To install one of these
packages first download it from the `OpenSUSE software website`_. Please note,
that this site does not only offer OpenSUSE packages, but also binaries for
other distributions, including: CentOS 5, Fedora 9-10, Mandriva 2007-2008, RedHat
Enterprise Linux 5, SUSE Linux Enterprise 10, OpenSUSE 10.2 up to 11.0.  Once
downloaded, open a console and invoke (the example command refers to NiBabel
0.3.1)::

  rpm -i python-nifti-0.20080710.1-4.1.i386.rpm

The OpenSUSE website also offers `1-click-installations`_ for distributions
supporting it.

A more convenient way to install NiBabel and automatically receive software
updates is to included one of the `RPM-package repositories`_ in the system's
package management configuration. For e.g. OpenSUSE 11.0, simply use Yast to add
another repository, using the following URL:

.. _RPM-package repositories: http://download.opensuse.org/repositories/home:/hankem/openSUSE_11.0/

For other distributions use the respective package managers (e.g. Yum) to setup
the repository URL.  The repositories include all core dependencies of NiBabel,
if they are not available from other repositories
of the respective distribution. There are two different repository groups, one
for `Suse and Mandriva-related packages`_ and another one for `Fedora, Redhat
and CentOS-related packages`_.

.. _Suse and Mandriva-related packages: http://download.opensuse.org/repositories/home:/hankem/
.. _Fedora, Redhat and CentOS-related packages: http://download.opensuse.org/repositories/home://hankem://rh5/
.. _1-click-installations: http://software.opensuse.org/search?baseproject=ALL&p=1&q=python-nifti
.. _OpenSUSE software website: http://software.opensuse.org/search?baseproject=ALL&p=1&q=python-nifti
.. _OpenSUSE Build Service: https://build.opensuse.org/


.. _install_win:

Windows
-------

A binary installer for a recent Python version is available from the
nifticlibs Sourceforge_ project site.

There are a few Python distributions for Windows. In theory all of them should
work equally well. However, I only tested the standard Python distribution
from www.python.org (with version 2.5.2).

First you need to download and install Python. Use the Python installer for
this job. Yo do not need to install the Python test suite and utility scripts.
From now on we will assume that Python was installed in `C:\\Python25` and that
this directory has been added to the `PATH` environment variable.

In addition you'll need NumPy_. Download a matching NumPy windows installer for
your Python version (in this case 2.5) from the `SciPy download page`_ and
install it.

Now, you can use the NiBabel windows installer to install NiBabel on your
system.  As always: click *Next* as long as necessary and finally *Finish*.  If
done, verify that everything went fine by opening a command promt and start
Python by typing `python` and hit enter. Now you should see the Python prompt.
Import the nifti module, which should cause no error messages::

  >>> import nibabel
  >>>

.. _SciPy download page: http://scipy.org/Download
.. _NIfTI libraries: http://niftilib.sourceforge.net/
.. _GnuWin32 project: http://gnuwin32.sourceforge.net/


.. _install_macos:

MacOS X
-------

The easiest installation method for OSX is via MacPorts_. MacPorts is a package
management system for MacOS, which is in some respects very similiar to RPM or
APT which are used in most GNU/Linux distributions. However, rather than
installing binary packages, it compiles software from source on the target
machine. 

*The MacPort of NiBabel is kindly maintained by James Kyle <jameskyle@ucla.edu>.*

.. _MacPorts: http://www.macports.org

In the context of NiBabel MacPorts is much easier to handle than the previously
available installer for Macs.  Although the initial overhead to setup MacPorts
on a machine is higher than simply installing NiBabel using the former
installer, MacPorts saves the user a significant amount of time (in the long
run). This is due to the fact that this framework will not only take care of
updating a NiBabel installation automatically whenever a new release is
available. It will also provide many of the optional dependencies of NiBabel
(e.g. NumPy_, nifticlibs) in the same environment and therefore abolishes the
need to manually check dozens of websites for updates and deal with an
unbelievable number of different installation methods.

MacPorts provides a universal binary package installer that is downloadable at
http://www.macports.org/install.php

After downloading, simply mount the dmg image and double click `MacPorts.pkg`.

By default, MacPorts installs to `/opt/local`. After the installation is
completed, you must ensure that your paths are set up correctly in order to
access the programs and utilities installed by MacPorts. For exhaustive details
on editing shell paths please see:

  http://www.debian.org/doc/manuals/reference/ch-install.en.html#s-bashconf

A typical `.bash_profile` set up for MacPorts might look like::

  > export PATH=/opt/local/bin:/opt/local/sbin:$PATH
  > export DYLD_LIBRARY_PATH=/opt/local/lib:$DYLD_LIBRARY_PATH

Be sure to source your .bash_profile or close Terminal.app and reopen it for
these changes to take effect.

Once MacPorts is installed and your environment is properly configured, NiBabel
is installed using a single command::

  > $ sudo port install py25-nibabel

If this is your first time using MacPorts Python 2.5 will be automatically
installed for you. However, an additional step is needed::

  $ sudo port install python_select
  $ sudo python_select python25

MacPorts has the ability of installing several Python versions at a time, the
`python_select` utility ensures that the default Python (located at
`/opt/local/bin/python`) points to your preferred version.

Upon success, open a terminal window and start Python by typing `python` and
hit return. Now try to import the NiBabel module by doing:

  >>> import nibabel
  >>>

If no error messages appear, you have succesfully installed NiBabel.


.. _requirements:
.. _buildfromsource:

Compile from source
===================

If no binary packages are provided for your platfom, you can build NiBabel from
source. It needs a few things to build and run properly:

*  Python_ 2.4 or greater
*  NumPy_

Get the sources
---------------

You can download a tarball of the latest development snapshot (i.e. the
current state of the *master* branch of the NiBabel source code
repository) from the `nibabel github`_ interface.

If you want to have access to the full NiBabel history and the latest
development code, do a full clone (aka checkout) of the NiBabel
repository::

  git clone http://github.com/hanke/nibabel.git

Compiling: General instructions
-------------------------------

Just install the modules by invoking::

  sudo python setup.py install

If sudo is not configured (or even installed) you might have to use
``su`` instead.

Now fire up Python and try importing the module to see if everything is fine.
It should look similar to this::

  Python 2.4.4 (#2, Oct 20 2006, 00:23:25)
  [GCC 4.1.2 20061015 (prerelease) (Debian 4.1.1-16.1)] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import nibabel
  >>>


Building on Windows Systems
---------------------------

On Windows you need to install the Python_ and NumPy_, if not done
yet. Make sure Python is on your path, and then do the usual Python install::

    python setup.py install


MacOS X
-------

If you want to use or even work on the latest development code, you
should also install Git_.  There is a `MacOS installer for Git`_, that
makes this step very easy.

.. _XCode developer tools: http://developer.apple.com/tools/xcode/
.. _MacOS installer for Git: http://code.google.com/p/git-osx-installer/

Otherwise follow the :ref:`general build instructions <buildfromsource>`.

.. include:: links_names.txt

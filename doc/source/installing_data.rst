.. _installing-data:

Installing data packages
========================

nibabel includes some machinery for using optional data packages.  We use data
packages for some of the DICOM tests in nibabel.  There are also data packages
for standard template images, and other packages for components of nipy,
including the main nipy package.

For more details on data package design, see :ref:`data-package-design`. 

We haven't yet made a nice automated way of downloading and installing the
packages.  For the moment you can find packages for the data and template files
at http://nipy.org/data-packages.

Data package installation as an administrator
---------------------------------------------

The installation procedure, for now, is very basic.  For example, let us
say that you want the 'nipy-templates' package at
http://nipy.org/data-packages/nipy-templates-0.1.tar.gz
. You simply download this archive, unpack it, and then run the standard
``python setup.py install`` on it.  On a unix system this might look
like::

   curl -O http://nipy.org/data-packages/nipy-templates-0.1.tar.gz
   tar zxvf nipy-templates-0.1.tar.gz
   cd nipy-templates-0.1
   sudo python setup.py install

On windows, download the file, extract the archive to a folder using the
GUI, and then, using the windows shell or similar::

   cd c:\path\to\extracted\files
   python setup.py install

Non-administrator data package installation
-------------------------------------------

The commands above assume you are installing into the default system
directories.  If you want to install into a custom directory, then (in
python, or ipython, or a text editor) look at the help for
``nipy.utils.data.get_data_path()`` . There are instructions there for
pointing your nipy installation to the installed data.

On unix
~~~~~~~

For example, say you installed with::

   cd nipy-templates-0.1
   python setup.py install --prefix=/home/my-user/some-dir

Then you may want to do make a file ``~/.nipy/config.ini`` with the
following contents::

   [DATA]
   /home/my-user/some-dir/share/nipy

On windows
~~~~~~~~~~

Say you installed with (windows shell)::

   cd nipy-templates-0.1
   python setup.py install --prefix=c:\some\path

Then first, find out your home directory::

   python -c "import os; print os.path.expanduser('~')"

Let's say that was ``c:\Documents and Settings\My User``.  Then, make a
new file called ``c:\Documents and Settings\My User\_nipy\config.ini``
with contents::

   [DATA]
   c:\some\path\share\nipy


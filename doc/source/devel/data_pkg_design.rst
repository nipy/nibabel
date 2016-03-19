.. _data-package-design:

Design of data packages for the nibabel and the nipy suite
==========================================================

See :ref:`data-package-discuss` for a more general discussion of design
issues.

When developing or using nipy, many data files can be useful. We divide the
data files nipy uses into at least 3 categories

#. *test data* - data files required for routine code testing
#. *template data* - data files required for algorithms to function,
   such as templates or atlases
#. *example data* - data files for running examples, or optional tests

Files used for routine testing are typically very small data files. They are
shipped with the software, and live in the code repository. For example, in
the case of ``nipy`` itself, there are some test files that live in the module
path ``nipy.testing.data``.  Nibabel ships data files in
``nibabel.tests.data``.  See :doc:`add_test_data` for discussion.

*template data* and *example data* are example of *data packages*.  What
follows is a discussion of the design and use of data packages.

.. testsetup::

    # Make fake data and template directories
    import os
    from os.path import join as pjoin
    import tempfile
    tmpdir = tempfile.mkdtemp()
    os.environ['NIPY_USER_DIR'] = tmpdir
    for subdir in ('data', 'templates'):
        files_dir = pjoin(tmpdir, 'nipy', subdir)
        os.makedirs(files_dir)
        with open(pjoin(files_dir, 'config.ini'), 'wt') as fobj:
            fobj.write(
    """[DEFAULT]
    version = 0.2
    """)

Use cases for data packages
+++++++++++++++++++++++++++

Using the data package
``````````````````````

The programmer can use the data like this:

.. testcode::

   from nibabel.data import make_datasource

   templates = make_datasource(dict(relpath='nipy/templates'))
   fname = templates.get_filename('ICBM152', '2mm', 'T1.nii.gz')

where ``fname`` will be the absolute path to the template image
``ICBM152/2mm/T1.nii.gz``.

The programmer can insist on a particular version of a ``datasource``:

>>> if templates.version < '0.4':
...     raise ValueError('Need datasource version at least 0.4')
Traceback (most recent call last):
...
ValueError: Need datasource version at least 0.4

If the repository cannot find the data, then:

>>> make_datasource(dict(relpath='nipy/implausible'))
Traceback (most recent call last):
 ...
nibabel.data.DataError: ...

where ``DataError`` gives a helpful warning about why the data was not
found, and how it should be installed.

Warnings during installation
````````````````````````````

The example data and template data may be important, and so we want to warn
the user if NIPY cannot find either of the two sets of data when installing
the package.  Thus::

   python setup.py install

will import nipy after installation to check whether these raise an error:

>>> from nibabel.data import make_datasource
>>> templates = make_datasource(dict(relpath='nipy/templates'))
>>> example_data = make_datasource(dict(relpath='nipy/data'))

and warn the user accordingly, with some basic instructions for how to
install the data.

.. _find-data:

Finding the data
````````````````

The routine ``make_datasource`` will look for data packages that have been
installed.  For the following call:

>>> templates = make_datasource(dict(relpath='nipy/templates'))

the code will:

#. Get a list of paths where data is known to be stored with
   ``nibabel.data.get_data_path()``
#. For each of these paths, search for directory ``nipy/templates``.  If
   found, and of the correct format (see below), return a datasource,
   otherwise raise an Exception

The paths collected by ``nibabel.data.get_data_paths()`` are constructed from
':' (Unix) or ';' separated strings.  The source of the strings (in the order
in which they will be used in the search above) are:

#. The value of the ``NIPY_DATA_PATH`` environment variable, if set
#. A section = ``DATA``, parameter = ``path`` entry in a
   ``config.ini`` file in ``nipy_dir`` where ``nipy_dir`` is
   ``$HOME/.nipy`` or equivalent.
#. Section = ``DATA``, parameter = ``path`` entries in configuration
   ``.ini`` files, where the ``.ini`` files are found by
   ``glob.glob(os.path.join(etc_dir, '*.ini')`` and ``etc_dir`` is
   ``/etc/nipy`` on Unix, and some suitable equivalent on Windows.
#. The result of ``os.path.join(sys.prefix, 'share', 'nipy')``
#. If ``sys.prefix`` is ``/usr``, we add ``/usr/local/share/nipy``. We
   need this because Python >= 2.6 in Debian / Ubuntu does default installs to
   ``/usr/local``.
#. The result of ``get_nipy_user_dir()``

Requirements for a data package
```````````````````````````````

To be a valid NIPY project data package, you need to satisfy:

#. The installer installs the data in some place that can be found using
   the method defined in :ref:`find-data`.

We recommend that:

#. By default, you install data in a standard location such as
   ``<prefix>/share/nipy`` where ``<prefix>`` is the standard Python
   prefix obtained by ``>>> import sys; print sys.prefix``

Remember that there is a distinction between the NIPY project - the
umbrella of neuroimaging in python - and the NIPY package - the main
code package in the NIPY project.  Thus, if you want to install data
under the NIPY *package* umbrella, your data might go to
``/usr/share/nipy/nipy/packagename`` (on Unix).  Note ``nipy`` twice -
once for the project, once for the package.  If you want to install data
under - say - the ``pbrain`` package umbrella, that would go in
``/usr/share/nipy/pbrain/packagename``.

Data package format
```````````````````

The following tree is an example of the kind of pattern we would expect
in a data directory, where the ``nipy-data`` and ``nipy-templates``
packages have been installed::

  <ROOT>
  `-- nipy
      |-- data
      |   |-- config.ini
      |   `-- placeholder.txt
      `-- templates
          |-- ICBM152
          |   `-- 2mm
          |       `-- T1.nii.gz
          |-- colin27
          |   `-- 2mm
          |       `-- T1.nii.gz
          `-- config.ini

The ``<ROOT>`` directory is the directory that will appear somewhere in
the list from ``nibabel.data.get_data_path()``.  The ``nipy`` subdirectory
signifies data for the ``nipy`` package (as opposed to other
NIPY-related packages such as ``pbrain``).  The ``data`` subdirectory of
``nipy`` contains files from the ``nipy-data`` package.  In the
``nipy/data`` or ``nipy/templates`` directories, there is a
``config.ini`` file, that has at least an entry like this::

  [DEFAULT]
  version = 0.2

giving the version of the data package.

.. _data-package-design-install:

Installing the data
```````````````````

We use python distutils to install data packages, and the ``data_files``
mechanism to install the data.  On Unix, with the following command::

   python setup.py install --prefix=/my/prefix

data will go to::

   /my/prefix/share/nipy

For the example above this will result in these subdirectories::

   /my/prefix/share/nipy/nipy/data
   /my/prefix/share/nipy/nipy/templates

because ``nipy`` is both the project, and the package to which the data
relates.

If you install to a particular location, you will need to add that location to
the output of ``nibabel.data.get_data_path()`` using one of the mechanisms
above, for example, in your system configuration::

   export NIPY_DATA_PATH=/my/prefix/share/nipy

Packaging for distributions
```````````````````````````

For a particular data package - say ``nipy-templates`` - distributions
will want to:

#. Install the data in set location.  The default from ``python setup.py
   install`` for the data packages will be ``/usr/share/nipy`` on Unix.
#. Point a system installation of NIPY to these data.

For the latter, the most obvious route is to copy an ``.ini`` file named for
the data package into the NIPY ``etc_dir``.  In this case, on Unix, we will
want a file called ``/etc/nipy/nipy_templates.ini`` with contents::

   [DATA]
   path = /usr/share/nipy

Current implementation
``````````````````````

This section describes how we (the nipy community) implement data packages at
the moment.

The data in the data packages will not usually be under source control.  This
is because images don't compress very well, and any change in the data will
result in a large extra storage cost in the repository.  If you're pretty
clear that the data files aren't going to change, then a repository could work
OK.

The data packages will be available at a central release location.  For now
this will be: http://nipy.org/data-packages/ .

A package, such as ``nipy-templates-0.2.tar.gz`` will have the following sort
of structure::


  <ROOT>
    |-- setup.py
    |-- README.txt
    |-- MANIFEST.in
    `-- templates
        |-- ICBM152
        |   |-- 1mm
        |   |   `-- T1_brain.nii.gz
        |   `-- 2mm
        |       `-- T1.nii.gz
        |-- colin27
        |   `-- 2mm
        |       `-- T1.nii.gz
        `-- config.ini


There should be only one ``nipy/packagename`` directory delivered by a
particular package.  For example, this package installs ``nipy/templates``,
but does not contain ``nipy/data``.

Making a new package tarball is simply:

#. Downloading and unpacking e.g. ``nipy-templates-0.1.tar.gz`` to form the
   directory structure above;
#. Making any changes to the directory;
#. Running ``setup.py sdist`` to recreate the package.

The process of making a release should be:

#. Increment the major or minor version number in the ``config.ini`` file;
#. Make a package tarball as above;
#. Upload to distribution site.

There is an example nipy data package ``nipy-examplepkg`` in the
``examples`` directory of the NIPY repository.

The machinery for creating and maintaining data packages is available at
https://github.com/nipy/data-packaging.

See the ``README.txt`` file there for more information.

.. testcleanup::

    import shutil
    shutil.rmtree(tmpdir)

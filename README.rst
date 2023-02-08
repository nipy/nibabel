.. -*- rest -*-
.. vim:syntax=rst

.. Following contents should be from LONG_DESCRIPTION in nibabel/info.py


.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - Code
     -
      .. image:: https://img.shields.io/badge/code%20style-black-000000.svg
         :target: https://github.com/psf/black
         :alt: code style: black
      .. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
         :target: https://pycqa.github.io/isort/
         :alt: imports: isort
      .. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
         :target: https://github.com/pre-commit/pre-commit
         :alt: pre-commit
      .. image:: https://codecov.io/gh/nipy/nibabel/branch/master/graph/badge.svg
         :target: https://codecov.io/gh/nipy/nibabel
         :alt: codecov badge
      .. image:: https://img.shields.io/librariesio/github/nipy/nibabel
         :target: https://libraries.io/github/nipy/nibabel
         :alt: Libraries.io dependency status for GitHub repo
   * - Status
     -
      .. image:: https://github.com/nipy/nibabel/actions/workflows/stable.yml/badge.svg
         :target: https://github.com/nipy/nibabel/actions/workflows/stable.yml
         :alt: stable tests
      .. image:: https://github.com/nipy/nibabel/actions/workflows/pages/pages-build-deployment/badge.svg
         :target: https://github.com/nipy/nibabel/actions/workflows/pages/pages-build-deployment
         :alt: documentation build
   * - Packaging
     -
      .. image:: https://img.shields.io/pypi/v/nibabel.svg
         :target: https://pypi.python.org/pypi/nibabel/
         :alt: PyPI version
      .. image:: https://img.shields.io/pypi/format/nibabel.svg
         :target: https://pypi.org/project/nibabel
         :alt: PyPI Format
      .. image:: https://img.shields.io/pypi/pyversions/nibabel.svg
         :target: https://pypi.python.org/pypi/nibabel/
         :alt: PyPI - Python Version
      .. image:: https://img.shields.io/pypi/implementation/nibabel.svg
         :target: https://pypi.python.org/pypi/nibabel/
         :alt: PyPI - Implementation
      .. image:: https://img.shields.io/pypi/dm/nibabel.svg
         :target: https://pypistats.org/packages/nibabel
         :alt: PyPI - Downloads
   * - Distribution
     -
      .. image:: https://repology.org/badge/version-for-repo/aur/python:nibabel.svg?header=Arch%20%28%41%55%52%29
         :target: https://repology.org/project/python:nibabel/versions
         :alt: Arch (AUR)
      .. image:: https://repology.org/badge/version-for-repo/debian_unstable/nibabel.svg?header=Debian%20Unstable
         :target: https://repology.org/project/nibabel/versions
         :alt: Debian Unstable package
      .. image:: https://repology.org/badge/version-for-repo/gentoo_ovl_science/nibabel.svg?header=Gentoo%20%28%3A%3Ascience%29
         :target: https://repology.org/project/nibabel/versions
         :alt: Gentoo (::science)
      .. image:: https://repology.org/badge/version-for-repo/nix_unstable/python:nibabel.svg?header=nixpkgs%20unstable
         :target: https://repology.org/project/python:nibabel/versions
         :alt: nixpkgs unstable
   * - License & DOI
     -
      .. image:: https://img.shields.io/pypi/l/nibabel.svg
         :target: https://github.com/nipy/nibabel/blob/master/COPYING
         :alt: License
      .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.591597.svg
         :target: https://doi.org/10.5281/zenodo.591597
         :alt: Zenodo DOI


=======
NiBabel
=======

Read / write access to some common neuroimaging file formats

This package provides read +/- write access to some common medical and
neuroimaging file formats, including: ANALYZE_ (plain, SPM99, SPM2 and later),
GIFTI_, NIfTI1_, NIfTI2_, `CIFTI-2`_, MINC1_, MINC2_, `AFNI BRIK/HEAD`_, MGH_ and
ECAT_ as well as Philips PAR/REC.  We can read and write FreeSurfer_ geometry,
annotation and morphometry files.  There is some very limited support for
DICOM_.  NiBabel is the successor of PyNIfTI_.

.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _AFNI BRIK/HEAD: https://afni.nimh.nih.gov/pub/dist/src/README.attributes
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _NIfTI2: http://nifti.nimh.nih.gov/nifti-2/
.. _CIFTI-2: https://www.nitrc.org/projects/cifti/
.. _MINC1:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC1_File_Format_Reference
.. _MINC2:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti/
.. _GIFTI: https://www.nitrc.org/projects/gifti
.. _MGH: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
.. _ECAT: http://xmedcon.sourceforge.net/Docs/Ecat
.. _Freesurfer: https://surfer.nmr.mgh.harvard.edu
.. _DICOM: http://medical.nema.org/

The various image format classes give full or selective access to header
(meta) information and access to the image data is made available via NumPy
arrays.

Website
=======

Current documentation on nibabel can always be found at the `NIPY nibabel
website <http://nipy.org/nibabel>`_.

Mailing Lists
=============

Please send any questions or suggestions to the `neuroimaging mailing list
<https://mail.python.org/mailman/listinfo/neuroimaging>`_.

Code
====

Install nibabel with::

    pip install nibabel

You may also be interested in:

* the `nibabel code repository`_ on Github;
* documentation_ for all releases and current development tree;
* download the `current release`_ from pypi;
* download `current development version`_ as a zip file;
* downloads of all `available releases`_.

.. _nibabel code repository: https://github.com/nipy/nibabel
.. _Documentation: http://nipy.org/nibabel
.. _current release: https://pypi.python.org/pypi/nibabel
.. _current development version: https://github.com/nipy/nibabel/archive/master.zip
.. _available releases: https://github.com/nipy/nibabel/releases

License
=======

Nibabel is licensed under the terms of the MIT license. Some code included
with nibabel is licensed under the BSD license.  Please see the COPYING file
in the nibabel distribution.

Citing nibabel
==============

Please see the `available releases`_ for the release of nibabel that you are
using.  Recent releases have a Zenodo_ `Digital Object Identifier`_ badge at
the top of the release notes.  Click on the badge for more information.

.. _zenodo: https://zenodo.org
.. _Digital Object Identifier: https://en.wikipedia.org/wiki/Digital_object_identifier

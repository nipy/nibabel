.. -*- rest -*-
.. vim:syntax=rst

.. Use raw location to ensure image shows up on PyPI
.. image:: https://raw.githubusercontent.com/nipy/nibabel/master/doc/pics/logo.png
   :target: https://nipy.org/nibabel
   :alt: NiBabel logo

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - Code
     -
      .. image:: https://img.shields.io/pypi/pyversions/nibabel.svg
         :target: https://pypi.python.org/pypi/nibabel/
         :alt: PyPI - Python Version
      .. image:: https://img.shields.io/badge/code%20style-blue-blue.svg
         :target: https://blue.readthedocs.io/en/latest/
         :alt: code style: blue
      .. image:: https://img.shields.io/badge/imports-isort-1674b1
         :target: https://pycqa.github.io/isort/
         :alt: imports: isort
      .. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
         :target: https://github.com/pre-commit/pre-commit
         :alt: pre-commit

   * - Tests
     -
      .. image:: https://github.com/nipy/NiBabel/actions/workflows/stable.yml/badge.svg
         :target: https://github.com/nipy/NiBabel/actions/workflows/stable.yml
         :alt: stable tests
      .. image:: https://codecov.io/gh/nipy/NiBabel/branch/master/graph/badge.svg
         :target: https://codecov.io/gh/nipy/NiBabel
         :alt: codecov badge

   * - PyPI
     -
      .. image:: https://img.shields.io/pypi/v/nibabel.svg
         :target: https://pypi.python.org/pypi/nibabel/
         :alt: PyPI version
      .. image:: https://img.shields.io/pypi/dm/nibabel.svg
         :target: https://pypistats.org/packages/nibabel
         :alt: PyPI - Downloads

   * - Packages
     -
      .. image:: https://img.shields.io/conda/vn/conda-forge/nibabel
         :target: https://anaconda.org/conda-forge/nibabel
         :alt: Conda package
      .. image:: https://repology.org/badge/version-for-repo/debian_unstable/nibabel.svg?header=Debian%20Unstable
         :target: https://repology.org/project/nibabel/versions
         :alt: Debian Unstable package
      .. image:: https://repology.org/badge/version-for-repo/aur/python:nibabel.svg?header=Arch%20%28%41%55%52%29
         :target: https://repology.org/project/python:nibabel/versions
         :alt: Arch (AUR)
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

.. Following contents should be copied from LONG_DESCRIPTION in nibabel/info.py


Read and write access to common neuroimaging file formats, including:
ANALYZE_ (plain, SPM99, SPM2 and later), GIFTI_, NIfTI1_, NIfTI2_, `CIFTI-2`_,
MINC1_, MINC2_, `AFNI BRIK/HEAD`_, ECAT_ and Philips PAR/REC.
In addition, NiBabel also supports FreeSurfer_'s MGH_, geometry, annotation and
morphometry files, and provides some limited support for DICOM_.

NiBabel's API gives full or selective access to header information (metadata),
and image data is made available via NumPy arrays. For more information, see
NiBabel's `documentation site`_ and `API reference`_.

.. _API reference: https://nipy.org/nibabel/api.html
.. _AFNI BRIK/HEAD: https://afni.nimh.nih.gov/pub/dist/src/README.attributes
.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _CIFTI-2: https://www.nitrc.org/projects/cifti/
.. _DICOM: http://medical.nema.org/
.. _documentation site: http://nipy.org/nibabel
.. _ECAT: http://xmedcon.sourceforge.net/Docs/Ecat
.. _Freesurfer: https://surfer.nmr.mgh.harvard.edu
.. _GIFTI: https://www.nitrc.org/projects/gifti
.. _MGH: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
.. _MINC1:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC1_File_Format_Reference
.. _MINC2:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _NIfTI2: http://nifti.nimh.nih.gov/nifti-2/

Installation
============

To install NiBabel's `current release`_ with ``pip``, run::

   pip install nibabel

To install the latest development version, run::

   pip install git+https://github.com/nipy/nibabel

When working on NiBabel itself, it may be useful to install in "editable" mode::

   git clone https://github.com/nipy/nibabel.git
   pip install -e ./nibabel

For more information on previous releases, see the `release archive`_ or
`development changelog`_.

Please find detailed `download and installation instructions
<installation>`_ in the manual.

.. _current release: https://pypi.python.org/pypi/NiBabel
.. _development changelog: https://nipy.org/nibabel/changelog.html
.. _installation: https://nipy.org/nibabel/installation.html#installation
.. _release archive: https://github.com/nipy/NiBabel/releases

Support
=======

If you have problems installing the software or questions about usage,
documentation or anything else related to NiBabel, you can post to the NiPy
mailing list: neuroimaging@python.org [subscription_, archive_]

The mailing list is the preferred way to announce changes and additions to the
project. You can also search the mailing list archive using the mailing list
archive search located in the sidebar of the NiBabel home page.
We recommend that anyone using NiBabel subscribes to the mailing list.

.. _subscription: https://mail.python.org/mailman/listinfo/neuroimaging
.. _archive: https://mail.python.org/pipermail/neuroimaging

License
=======

NiBabel is free-software (beer and speech) and covered by the
`MIT License <mit>`_. This applies to all source code, documentation,
examples and snippets inside the source distribution (including this website).
Please see the `appendix of the manual <license>`_ for the copyright statement
and the full text of the license.

.. _license: https://nipy.org/nibabel/legal.html#license
.. _mit: https://opensource.org/license17s/MIT

Citation
========

NiBabel releases have a Zenodo_ `Digital Object Identifier`_ (DOI) badge at
the top of the release notes. Click on the badge for more information.

.. _Digital Object Identifier: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _zenodo: https://zenodo.org

.. -*- rest -*-
.. vim:syntax=rst

.. image:: https://coveralls.io/repos/nipy/nibabel/badge.png?branch=master
    :target: https://coveralls.io/r/nipy/nibabel?branch=master

.. Following contents should be from LONG_DESCRIPTION in nibabel/info.py

TESTME

=======
NiBabel
=======

Read / write access to some common neuroimaging file formats

This package provides read +/- write access to some common medical and
neuroimaging file formats, including: ANALYZE_ (plain, SPM99, SPM2 and later),
GIFTI_, NIfTI1_, NIfTI2_, MINC1_, MINC2_, MGH_ and ECAT_ as well as Philips
PAR/REC.  We can read and write FreeSurfer_ geometry, annotation and
morphometry files.  There is some very limited support for DICOM_.  NiBabel is
the successor of PyNIfTI_.

.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _NIfTI2: http://nifti.nimh.nih.gov/nifti-2/
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

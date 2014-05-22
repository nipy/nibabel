.. -*- rest -*-
.. vim:syntax=rest

=======
NiBabel
=======

Read / write access to some common neuroimaging file formats

This package provides read +/- write access to some common medical and
neuroimaging file formats, including: ANALYZE_ (plain, SPM99, SPM2),
GIFTI_, NIfTI1_, NIfTI2_, MINC1_, MINC2_, MGH_ and ECAT_ as well as PAR/REC.
We can read and write Freesurfer_ geometry, and read Freesurfer morphometry and
annotation files.  There is some very limited support for DICOM_.  NiBabel is
the successor of PyNIfTI_.

.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _NIfTI2: http://nifti.nimh.nih.gov/nifti-2/
.. _MINC1:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC1_File_Format_Reference
.. _MINC2:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti/
.. _GIFTI: http://www.nitrc.org/projects/gifti
.. _MGH: http://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
.. _ECAT: http://xmedcon.sourceforge.net/Docs/Ecat
.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu
.. _DICOM: http://medical.nema.org/

The various image format classes give full or selective access to header (meta)
information and access to the image data is made available via NumPy arrays.

Website
=======

Current information can always be found at the NIPY nibabel website::

    http://nipy.org/nibabel

Mailing Lists
=============

Please see the developer's list here::

    http://mail.scipy.org/mailman/listinfo/nipy-devel

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.
* Downloads of all `available releases`_.

.. _main repository: http://github.com/nipy/nibabel
.. _Documentation: http://nipy.org/nibabel
.. _current trunk: http://github.com/nipy/nibabel/archives/master
.. _available releases: http://github.com/nipy/nibabel/downloads

License
=======

Nibabel is licensed under the terms of the MIT license. Some code included with
nibabel is licensed under the BSD license.  Please see the COPYING file in the
nibabel distribution.

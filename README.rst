.. -*- rest -*-
.. vim:syntax=rest

=======
NiBabel
=======

This package provides read and write access to some common medical and
neuroimaging file formats, including: ANALYZE_ (plain, SPM99, SPM2),
GIFTI_, NIfTI1_, MINC_, MGH_ and ECAT_ as well as PAR/REC. There is some very
limited support for DICOM_.  NiBabel is the successor of PyNIfTI_.

.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _MINC: http://wiki.bic.mni.mcgill.ca/index.php/MINC
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti/
.. _GIFTI: http://www.nitrc.org/projects/gifti
.. _MGH: http://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
.. _ECAT: http://xmedcon.sourceforge.net/Docs/Ecat
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
nibabel is licensed under the BSD license.  Please the COPYING file in the
nibabel distribution.

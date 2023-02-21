"""Define static nibabel metadata for nibabel

The long description parameter is used in the nibabel top-level docstring,
and in building the docs.
We exec this file in several places, so it cannot import nibabel or use
relative imports.
"""

# Note: this long_description is the canonical place to edit this text.
# It also appears in README.rst, but it should get there by running
# ``tools/refresh_readme.py`` which pulls in this version.
# We also include this text in the docs by ``..include::`` in
# ``docs/source/index.rst``.
long_description = """
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

NiBabel is free-software (beer and speech) and covered by the `MIT License`_.
This applies to all source code, documentation, examples and snippets inside
the source distribution (including this website). Please see the
`appendix of the manual <license>`_ for the copyright statement and the
full text of the license.

.. _license: https://nipy.org/nibabel/legal.html#license
.. _MIT License: https://opensource.org/license17s/MIT

Citation
========

NiBabel releases have a Zenodo_ `Digital Object Identifier`_ (DOI) badge at
the top of the release notes. Click on the badge for more information.

.. _Digital Object Identifier: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _zenodo: https://zenodo.org
"""   # noqa: E501

#######
Roadmap
#######

The roadmap is intended for larger, fundamental changes to the project that are
likely to take months or years of developer time. Smaller-scoped items will
continue to be tracked on our issue tracker.

The scope of these improvements means that these changes may be controversial,
are likely to involve significant discussion among the core development team,
and may require the creation of one or more BIAPs (niBabel Increased
Awesomeness Proposals).

==========
Background
==========

Nibabel is a workbench that provides a Python API for working with images in
many formats.  It is also a base library for tools implementing higher level
processing.

Nibabel's success depends on:

* How easy it is to express common imaging tasks in the API.
* The range of tasks it can perform.

An expressive, broad API will increase adoption and make it easier to teach.

Expressive API
==============

Axis and tick labels
--------------------

Brain images typically have three or four axes, whose meanings depend on the
way the image was acquired.  Axes have natural labels, expressing meaning,
such as "time" or "slice", and they may have tick labels such as acquisition
time. The scanner captures this information, but typical image formats cannot
store it, so it is easy to lose metadata and make analysis errors; see
:ref:`biap6`.

We plan to expand Nibabel's API to encode axis and tick labels by integrating
the `Xarray package <http://xarray.pydata.org>`_.  Xarray simplifies HDF5
serialization, and visualization.

An API for labels is not useful if we cannot read labels from the scanner
data, or save them with the image.  We plan to:

* Develop HDF5 equivalents of standard image formats, for serialization of
  data with labels.
* Expand the current standard image format, NIfTI, to store labels in a JSON
  addition to image metadata: :ref:`biap3`.
* Read image metadata from DICOM, the standard scanner format.

Reading and attaching DICOM data will start with code integrated from
`Dcmstack <https://github.com/moloney/dcmstack>`_, by Brendan Moloney; see:
:ref:`biap4`.

DICOM metadata is often hidden inside "private" DICOM elements that need
specialized parsers. We want to expand these parsers to preserve full metadata
and build a normalization layer to abstract vendor-specific storage locations
for metadata elements that describe the same thing.

API for surface data
--------------------

Neuroimaging data often refers to locations on the brain surface.  There are
three common formats for such data: GIFTI, CIFTI and Freesurfer.  Nibabel can
read these formats, but lacks a standard API for reading and storing surface
data with metadata; see
`nipy/nibabel#936 <https://github.com/nipy/nibabel/issues/936>`_,
`nilearn/nilearn#2171 <https://github.com/nilearn/nilearn/issues/2171>`_.
We plan to develop a standard API, apply it to the standard formats,
and design an efficient general HDF5 storage container for serializing surface
data and metadata.

Range
=====

Spatial transforms
------------------

Neuroimaging toolboxes include spatial registration methods to align the
objects and features present in two or more images. Registration methods
estimate and store spatial transforms.  There is no standard or compatible
format to store and reuse these transforms, across packages.

Because Nibabel is a workbench, we want to extend its support to read
transforms calculated with AFNI, FreeSurfer, FSL, ITK/ANTs, NiftyReg, and SPM.

We have developed the NiTransforms project for this task; we plan to complete
and integrate NiTransforms into Nibabel.  This will make transforms more
accessible to researchers, and therefore easier to work with, and reason about.

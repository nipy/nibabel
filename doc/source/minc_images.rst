########################
Working with MINC images
########################

This page describes nibabel's support for the MINC1_ (NetCDF) and MINC2_
(HDF5) formats of the MINC_ tools.  Nibabel reads and writes both formats;
MINC2 needs h5py_.

.. testsetup::

    # Work in a temporary directory
    import os
    import tempfile
    pwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)

*******************
Reading and writing
*******************

>>> import os
>>> import numpy as np
>>> import nibabel as nib
>>> from nibabel.testing import data_path
>>> img = nib.load(os.path.join(data_path, 'minc1_4d.mnc'))
>>> img
<nibabel.minc1.Minc1Image object at ...>
>>> img.shape
(2, 10, 20, 20)

The array axes are the MINC dimensions in file order, slowest varying first:
here ``time``, ``zspace``, ``yspace`` and ``xspace``.  The affine maps the
three spatial axes to world coordinates; time, if any, is the first axis, and
a ``vector_dimension`` (for example the components of RGB data) the last.

Saving to ``.mnc`` keeps the format of the image: a
:class:`~nibabel.minc1.Minc1Image` writes MINC1 and a
:class:`~nibabel.minc2.Minc2Image` writes MINC2.  Other images saved as
``.mnc``, such as NIfTI images, become MINC1, the first format nibabel
registers for that extension, as libminc tools write MINC1 unless asked for
MINC2.  To change the format, convert the image:

>>> nib.save(img, 'same_format.mnc')
>>> img2 = nib.Minc2Image.from_image(img)  # doctest: +SKIP
>>> nib.save(img2, 'minc2.mnc')  # doctest: +SKIP

The writers are :func:`nibabel.minc1_write.write_minc1` and
:func:`nibabel.minc2_write.write_minc2`, which also take arrays and an affine
directly, and options that ``nib.save`` passes on: ``slice_scaling``, and for
MINC2 ``compression``, ``chunks`` and ``labels``.

*****************
Keeping metadata
*****************

The readers keep a read-only snapshot of the file's metadata as
``img.minc_meta``, a :class:`~nibabel.minc_meta.MincMeta`:

>>> meta = img.minc_meta
>>> meta.dimorder
('time', 'zspace', 'yspace', 'xspace')
>>> meta.info['study'].attrs['modality']
'MRI__'

It holds the global attributes and history, every dimension variable with
its attributes, coordinates and widths, and the info variables (``patient``,
``study``, ``acquisition`` and any others).  When an image is saved, the
writer copies all of these, in the format being written, and appends a line
to the history.  The geometry (``start``, ``step``, ``direction_cosines``,
``length``, ``spacing``) and the scaling always come from the data and the
affine.  If the data shape or the affine changed since the image was read, a
dimension of the snapshot that no longer matches loses only its geometry:
its exact start, step and direction cosines, and the coordinates and widths
of irregular samples.  ``from_image`` carries the snapshot between
:class:`~nibabel.minc1.Minc1Image` and :class:`~nibabel.minc2.Minc2Image`.

Both formats share one model, libminc's MINC1 model of global attributes and
variables, which libminc also uses for MINC2 files: the attributes of
``/minc-2.0`` are global, ``image``, ``image-min`` and ``image-max`` are in
``/minc-2.0/image/0``, dimension variables in ``/minc-2.0/dimensions`` and all
other variables in ``/minc-2.0/info``.  Attributes that only describe one
format's structure (``parent``, ``children``, the ``rootvariable``,
``signtype``, ``dimorder``, ``length``) are written as that format needs.

*******
Scaling
*******

Integer data are scaled per slice, as libminc does by default: an
``image-min`` and ``image-max`` per index of all but the two fastest varying
spatial dimensions.  Integral data that fit the type are stored unscaled.
``slice_scaling=False`` writes a single range for the volume, and an image
read from a file keeps the scaling of that file, so that its stored values
are written back unchanged.  Floating point and complex data are not scaled.

***************************
Compatibility with libminc
***************************

Status of the features of the MINC1 and MINC2 formats.  "Tested" names the
external readers whose results the tests compare with nibabel's: libminc
2.4.05 of minc-toolkit-v2 1.9.18 (both formats), SimpleITK 2.5 (ITK's
libminc, MINC2) and FreeSurfer's MINC tools (libminc 1.5, MINC1 only).

.. list-table::
   :header-rows: 1
   :widths: 30 9 9 9 9 34

   * - Feature
     - MINC1 read
     - MINC1 write
     - MINC2 read
     - MINC2 write
     - Tested against
   * - Spatial dimensions, start, step, direction cosines
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, SimpleITK, FreeSurfer
   * - Time dimension (start, step, alignment ``start_``)
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, SimpleITK (as components), FreeSurfer
   * - Irregular spacing on any dimension, with widths
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, FreeSurfer; SimpleITK reads the data but keeps only
       start and step
   * - ``vector_dimension``
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, SimpleITK, FreeSurfer
   * - Spatial frequency dimensions (``xfrequency``, ``yfrequency``,
       ``zfrequency``), alone or mixed with spatial ones
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4 (files from ``rawtominc``; ``mincinfo``); mapped to the
       world axis of their name, as libminc does, so the affine maps
       samples to frequency coordinates
   * - ``tfrequency``
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4 (``rawtominc``); the first, non-spatial axis, like time
   * - uint8, int8, int16, uint16, int32, uint32 (``signtype``)
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, SimpleITK, FreeSurfer
   * - float32, float64
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, SimpleITK, FreeSurfer
   * - Float complex (``real``, ``imag``; class ``complex``)
     - n/a
     - n/a
     - yes
     - yes
     - libminc 2.4 (reads the real part); SimpleITK cannot read it
   * - Integer complex
     - n/a
     - n/a
     - no
     - no
     - not supported: libminc defines no valid range or scaling for it, so
       its real values are undefined
   * - Slice-wise and volume-wise scaling, ``valid_range``
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4, SimpleITK, FreeSurfer
   * - No image-min / image-max, no ``signtype``
     - yes
     - n/a
     - yes
     - n/a
     - libminc defaults
   * - Info variables and their attributes, history, ident
     - yes
     - yes
     - yes
     - yes
     - libminc 2.4 (``mincheader``, ``mincinfo -attvalue``), FreeSurfer
   * - Nested MINC2 info groups
     - n/a
     - n/a
     - yes
     - yes
     - left out with a warning when writing MINC1, as libminc's
       conversion leaves them out
   * - Label volumes (names in an enumerated type, class ``label__``)
     - study modality ``label``
     - refused
     - yes
     - yes
     - SimpleITK (values); libminc's MINC1 interface tools cannot read
       them, and libminc cannot write label names to MINC1 (``mincconvert``
       fails), so ``write_minc1`` refuses them (``labels={}`` writes the
       values only)
   * - gzip compression and chunking
     - ``.mnc.gz``
     - no
     - yes
     - yes
     - libminc 2.4, SimpleITK
   * - Resolution levels (``/minc-2.0/image/1``, ``/2``...)
     - n/a
     - refused
     - yes
     - yes (3D)
     - libminc 2.4 (a file written with ``miset_props_multi_resolution``);
       ``resolution_level`` of ``nib.load`` reads a level, ``levels`` of
       ``write_minc2`` writes them, and saving keeps them.  libminc writes
       levels of 3D images only, and none to MINC1, so the writers refuse
       other cases (``levels=0`` writes the full resolution only).  libminc
       makes level ``n`` from level ``n - 1`` with the wrong scaling;
       nibabel makes every level from the full resolution
   * - Record (unlimited) dimension
     - yes
     - yes
     - n/a
     - refused
     - libminc 2.4 (``mincgen``), FreeSurfer; libminc cannot write a
       record dimension to MINC2 (``mincconvert -2`` fails), so
       ``write_minc2`` refuses it (``record_dimension=False`` writes an
       ordinary dimension)
   * - Uniform record volumes (class ``array__``)
     - no
     - no
     - no
     - no
     - not supported: libminc makes them with an empty compound type and
       no tool writes them

Where libminc cannot write a feature to a format, nibabel reads it but
refuses to write it, with an error that names the argument that writes the
data without it.

Writing follows libminc: the MINC1 header of a new file is that of
``nii2mnc`` for the same data but for the valid range (of the type rather
than of the data) and the slice-wise scaling, and saving a file written by
``nii2mnc``, ``mincconvert`` or ``minccalc`` changes no attribute but
``ident``, ``minc_version`` and the appended history line.

*******
Testing
*******

The tests compare every attribute of files before and after a load and save,
in both formats and across them, for nibabel's MINC test files.  They run the
MINC tools on the ``PATH``, and those of a minc-toolkit-v2 Apptainer image
named by the ``MINC_TOOLKIT_SIF`` environment variable, if present; SimpleITK
checks run if SimpleITK is installed.

.. testcleanup::

    os.chdir(pwd)
    import shutil
    shutil.rmtree(tmp_dir)

.. include:: links_names.txt

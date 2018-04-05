#########################
Working with NIfTI images
#########################

This page describes some features of the nibabel implementation of the NIfTI
format.  Generally all these features apply equally to the NIfTI 1 and the
NIfTI 2 format, but we will note the differences when they come up.  NIfTI 1
is much more common than NIfTI 2.

.. testsetup::

    # Work in a temporary directory
    import os
    import tempfile
    pwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)

*************
Preliminaries
*************

We first set some display parameters to print out numpy arrays in a compact
form:

>>> import numpy as np
>>> # Set numpy to print only 2 decimal digits for neatness
>>> np.set_printoptions(precision=2, suppress=True)

********************
Example NIfTI images
********************

>>> import os
>>> import nibabel as nib
>>> from nibabel.testing import data_path

This is the example NIfTI 1 image:

>>> example_ni1 = os.path.join(data_path, 'example4d.nii.gz')
>>> n1_img = nib.load(example_ni1)
>>> n1_img
<nibabel.nifti1.Nifti1Image object at ...>

Here is the NIfTI 2 example image:

>>> example_ni2 = os.path.join(data_path, 'example_nifti2.nii.gz')
>>> n2_img = nib.load(example_ni2)
>>> n2_img
<nibabel.nifti2.Nifti2Image object at ...>

****************
The NIfTI header
****************

The NIfTI 1 header is a small C structure of size 352 bytes.  It contains the
following fields:

>>> n1_header = n1_img.header
>>> print(n1_header)                     # doctest: +SKIP
<class 'nibabel.nifti1.Nifti1Header'> object, endian='<'
sizeof_hdr      : 348
data_type       : b''
db_name         : b''
extents         : 0
session_error   : 0
regular         : b'r'
dim_info        : 57
dim             : [  4 128  96  24   2   1   1   1]
intent_p1       : 0.0
intent_p2       : 0.0
intent_p3       : 0.0
intent_code     : none
datatype        : int16
bitpix          : 16
slice_start     : 0
pixdim          : [   -1.      2.      2.      2.2  2000.      1.      1.      1. ]
vox_offset      : 0.0
scl_slope       : nan
scl_inter       : nan
slice_end       : 23
slice_code      : unknown
xyzt_units      : 10
cal_max         : 1162.0
cal_min         : 0.0
slice_duration  : 0.0
toffset         : 0.0
glmax           : 0
glmin           : 0
descrip         : b'FSL3.3\x00 v2.25 NIfTI-1 Single file format'
aux_file        : b''
qform_code      : scanner
sform_code      : scanner
quatern_b       : -1.94510681403e-26
quatern_c       : -0.996708512306
quatern_d       : -0.081068739295
qoffset_x       : 117.855102539
qoffset_y       : -35.7229423523
qoffset_z       : -7.24879837036
srow_x          : [  -2.      0.      0.    117.86]
srow_y          : [ -0.     1.97  -0.36 -35.72]
srow_z          : [ 0.    0.32  2.17 -7.25]
intent_name     : b''
magic           : b'n+1'

The NIfTI 2 header is similar, but of length 540 bytes, with fewer fields:

>>> n2_header = n2_img.header
>>> print(n2_header)                     # doctest: +SKIP
    <class 'nibabel.nifti2.Nifti2Header'> object, endian='<'
    sizeof_hdr      : 540
    magic           : b'n+2'
    eol_check       : [13 10 26 10]
    datatype        : int16
    bitpix          : 16
    dim             : [ 4 32 20 12  2  1  1  1]
    intent_p1       : 0.0
    intent_p2       : 0.0
    intent_p3       : 0.0
    pixdim          : [   -1.      2.      2.      2.2  2000.      1.      1.      1. ]
    vox_offset      : 0
    scl_slope       : nan
    scl_inter       : nan
    cal_max         : 1162.0
    cal_min         : 0.0
    slice_duration  : 0.0
    toffset         : 0.0
    slice_start     : 0
    slice_end       : 23
    descrip         : b'FSL3.3\x00 v2.25 NIfTI-1 Single file format'
    aux_file        : b''
    qform_code      : scanner
    sform_code      : scanner
    quatern_b       : -1.94510681403e-26
    quatern_c       : -0.996708512306
    quatern_d       : -0.081068739295
    qoffset_x       : 117.855102539
    qoffset_y       : -35.7229423523
    qoffset_z       : -7.24879837036
    srow_x          : [  -2.      0.      0.    117.86]
    srow_y          : [ -0.     1.97  -0.36 -35.72]
    srow_z          : [ 0.    0.32  2.17 -7.25]
    slice_code      : unknown
    xyzt_units      : 10
    intent_code     : none
    intent_name     : b''
    dim_info        : 57
    unused_str      : b''

You can get and set individual fields in the header using dict (mapping-type)
item access.  For example:

>>> n1_header['cal_max']
array(1162., dtype=float32)
>>> n1_header['cal_max'] = 1200
>>> n1_header['cal_max']
array(1200., dtype=float32)

Check the attributes of the header for ``get_`` / ``set_`` methods to get and
set various combinations of NIfTI header fields.

The ``get_`` / ``set_`` methods should check and apply valid combinations of
values from the header, whereas you can do anything you like with the dict /
mapping item access.  It is safer to use the ``get_`` / ``set_`` methods and
use the mapping item access only if the ``get_`` / ``set_`` methods will not
do what you want.

*****************
The NIfTI affines
*****************

Like other nibabel image types, NIfTI images have an affine relating the voxel
coordinates to world coordinates in RAS+ space:

>>> n1_img.affine
array([[ -2.  ,   0.  ,   0.  , 117.86],
       [ -0.  ,   1.97,  -0.36, -35.72],
       [  0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]])

Unlike other formats, the NIfTI header format can specify this affine in one
of three ways |--| the *sform* affine, the *qform* affine and the *fall-back
header* affine.

Nibabel uses an :ref:`algorithm <choosing-image-affine>` to chose which of
these three it will use for the overall image ``affine``.

The sform affine
================

The header stores the three first rows of the 4 by 4 affine in the header
fields ``srow_x``, ``srow_y``, ``srow_z``. The header does not store the
fourth row because it is always ``[0, 0, 0, 1]`` (see
:doc:`coordinate_systems`).

You can get the sform affine specifically with the ``get_sform()`` method of
the image or the header.

For example:

>>> print(n1_header['srow_x'])
[ -2.     0.     0.   117.86]
>>> print(n1_header['srow_y'])
[ -0.     1.97  -0.36 -35.72]
>>> print(n1_header['srow_z'])
[ 0.    0.32  2.17 -7.25]
>>> print(n1_header.get_sform())
[[ -2.     0.     0.   117.86]
 [ -0.     1.97  -0.36 -35.72]
 [  0.     0.32   2.17  -7.25]
 [  0.     0.     0.     1.  ]]

This affine is valid only if the ``sform_code`` is not zero.

>>> print(n1_header['sform_code'])
1

The different sform code values specify which RAS+ space the sform affine
refers to, with these interpretations:

==== ========= ===========
Code Label     Meaning
==== ========= ===========
0    unknown   sform not defined
1    scanner   RAS+ in scanner coordinates
2    aligned   RAS+ aligned to some other scan
3    talairach RAS+ in Talairach atlas space
4    mni       RAS+ in MNI atlas space
==== ========= ===========

In our case the code is 1, meaning "scanner" alignment.

You can get the affine and the code using the ``coded=True`` argument to
``get_sform()``:

>>> print(n1_header.get_sform(coded=True))
(array([[ -2.  ,   0.  ,   0.  , 117.86],
       [ -0.  ,   1.97,  -0.36, -35.72],
       [  0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]]), 1)

You can set the sform with the ``set_sform()`` method of the header and
the image.

>>> n1_header.set_sform(np.diag([2, 3, 4, 1]))
>>> n1_header.get_sform()
array([[2., 0., 0., 0.],
       [0., 3., 0., 0.],
       [0., 0., 4., 0.],
       [0., 0., 0., 1.]])

Set the affine and code using the ``code`` parameter to ``set_sform()``:

>>> n1_header.set_sform(np.diag([3, 4, 5, 1]), code='mni')
>>> n1_header.get_sform(coded=True)
(array([[3., 0., 0., 0.],
       [0., 4., 0., 0.],
       [0., 0., 5., 0.],
       [0., 0., 0., 1.]]), 4)

The qform affine
================

This affine can be calculated from a combination of the voxel sizes (entries 1
through 4 of the ``pixdim`` field), a sign flip called ``qfac`` stored in
entry 0 of ``pixdim``, and a `quaternion <wikipedia quaternion_>`_ that can be
reconstructed from fields ``quatern_b``, ``quatern_c``, ``quatern_d``.

See the code for the :meth:`get_qform() method
<nibabel.nifti1.Nifti1Header.get_qform>` for details.

You can get and set the qform affine using the equivalent methods to those for
the sform: ``get_qform()``, ``set_qform()``.

>>> n1_header.get_qform(coded=True)
(array([[ -2.  ,   0.  ,   0.  , 117.86],
       [ -0.  ,   1.97,  -0.36, -35.72],
       [  0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]]), 1)

The qform also has a corresponding ``qform_code`` with the same interpretation
as the `sform_code`.

The fall-back header affine
===========================

This is the affine of last resort, constructed only from the ``pixdim`` voxel
sizes.  The `NIfTI specification <nifti1>`_ says that this should set the
first voxel in the image as [0, 0, 0] in world coordinates, but we nibabblers
follow SPM_ in preferring to set the central voxel to have [0, 0, 0] world
coordinate. The NIfTI spec also implies that the image should be assumed to be
in RAS+ *voxel* orientation for this affine (see :doc:`coordinate_systems`).
Again like SPM, we prefer to assume LAS+ voxel orientation by default.

You can always get the fall-back affine with ``get_base_affine()``:

>>> n1_header.get_base_affine()
array([[ -2. ,   0. ,   0. , 127. ],
       [  0. ,   2. ,   0. , -95. ],
       [  0. ,   0. ,   2.2, -25.3],
       [  0. ,   0. ,   0. ,   1. ]])

.. _choosing-image-affine:

Choosing the image affine
=========================

Given there are three possible affines defined in the NIfTI header, nibabel
has to chose which of these to use for the image ``affine``.

The algorithm is defined in the ``get_best_affine()`` method.  It is:

#. If ``sform_code`` != 0 ('unknown') use the sform affine; else
#. If ``qform_code`` != 0 ('unknown') use the qform affine; else
#. Use the fall-back affine.

.. _default-sform-qform-codes:

Default sform and qform codes
=============================

If you create a new image, e.g.:

>>> data = np.random.random((20, 20, 20))
>>> xform = np.eye(4) * 2
>>> img = nib.nifti1.Nifti1Image(data, xform)

The sform and qform codes will be initialised to 2 (aligned) and 0 (unknown)
respectively:

>>> img.get_sform(coded=True) # doctest: +NORMALIZE_WHITESPACE
(array([[2., 0., 0., 0.],
       [0., 2., 0., 0.],
       [0., 0., 2., 0.],
       [0., 0., 0., 1.]]), 2)
>>> img.get_qform(coded=True)
(None, 0)

This is based on the assumption that the affine you specify for a newly
created image will align the image to some known coordinate system. According
to the `NIfTI specification <nifti1>`_, the qform is intended to encode a
transformation into scanner coordinates - for a programmatically created
image, we have no way of knowing what the scanner coordinate system is;
furthermore, the qform cannot be used to store an arbitrary affine transform,
as it is unable to encode shears. So the provided affine will be stored in the
sform, and the qform will be left uninitialised.

If you create a new image and specify an existing header, e.g.:

>>> example_ni1 = os.path.join(data_path, 'example4d.nii.gz')
>>> n1_img = nib.load(example_ni1)
>>> new_header = header=n1_img.header.copy()
>>> new_data = np.random.random(n1_img.shape[:3])
>>> new_img = nib.nifti1.Nifti1Image(data, None, header=new_header)

then the newly created image will inherit the same sform and qform codes that
are in the provided header. However, if you create a new image with both an
affine and a header specified, e.g.:

>>> xform = np.eye(4)
>>> new_img = nib.nifti1.Nifti1Image(data, xform, header=new_header)

then the sform and qform codes will *only* be preserved if the provided affine
is the same as the affine in the provided header. If the affines do not match,
the sform and qform codes will be set to their default values of 2 and 0
respectively. This is done on the basis that, if you are changing the affine,
you are likely to be changing the space to which the affine is pointing. So
the original sform and qform codes can no longer be assumed to be valid.

If you wish to set the sform and qform affines and/or codes to some other
value, you can always set them after creation using the ``set_sform`` and
``set_qform`` methods, as described above.

************
Data scaling
************

NIfTI uses a simple scheme for data scaling.

By default, nibabel will take care of this scaling for you, but there may be
times that you want to control the data scaling yourself.  If so, the next
section describes how the scaling works and the nibabel implementation of
same.

There are two scaling fields in the header called ``scl_slope`` and
``scl_inter``.

The output data from a NIfTI image comes from:

#. Loading the binary data from the image file;
#. Casting the numbers to the binary format given in the header and returned
   by ``get_data_dtype()``;
#. Reshaping to the output image shape;
#. Multiplying the result by the header ``scl_slope`` value, if
   both of ``scl_slope`` and ``scl_inter`` are defined;
#. Adding the value header ``scl_inter`` value to the result, if both of
   ``scl_slope`` and ``scl_inter`` are defined;

'Defined' means, the value is not NaN (not a number).

All this gets built into the array proxy when you load a NIfTI image.

When you load an image, the header scaling values automatically get set to NaN
(undefined) to mark the fact that the scaling values have been consumed by the
read.  The scaling values read from the header on load only appear in the
array proxy object.

To see how this works, let's make a new image with some scaling:

>>> array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
>>> affine = np.diag([1, 2, 3, 1])
>>> array_img = nib.Nifti1Image(array_data, affine)
>>> array_header = array_img.header

The default scaling values are NaN (undefined):

>>> array_header['scl_slope']
array(nan, dtype=float32)
>>> array_header['scl_inter']
array(nan, dtype=float32)

You can get the scaling values with the ``get_slope_inter()`` method:

>>> array_header.get_slope_inter()
(None, None)

None corresponds to the NaN scaling value (undefined).

We can set them in the image header, so they get saved to the header when the
image is written.  We can do this by setting the fields directly, or with
``set_slope_inter()``:

>>> array_header.set_slope_inter(2, 10)
>>> array_header.get_slope_inter()
(2.0, 10.0)
>>> array_header['scl_slope']
array(2., dtype=float32)
>>> array_header['scl_inter']
array(10., dtype=float32)

Setting the scale factors in the header has no effect on the image data before
we save and load again:

>>> array_img.get_fdata()
array([[[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]],
<BLANKLINE>
       [[12., 13., 14., 15.],
        [16., 17., 18., 19.],
        [20., 21., 22., 23.]]])

Now we save the image and load it again:

>>> nib.save(array_img, 'scaled_image.nii')
>>> scaled_img = nib.load('scaled_image.nii')

The data array has the scaling applied:

>>> scaled_img.get_fdata()
array([[[10., 12., 14., 16.],
        [18., 20., 22., 24.],
        [26., 28., 30., 32.]],
<BLANKLINE>
       [[34., 36., 38., 40.],
        [42., 44., 46., 48.],
        [50., 52., 54., 56.]]])

The header for the loaded image has had the scaling reset to undefined, to
mark the fact that the scaling has been "consumed" by the load:

>>> scaled_img.header.get_slope_inter()
(None, None)

The original slope and intercept are still accessible in the array proxy
object:

>>> scaled_img.dataobj.slope
2.0
>>> scaled_img.dataobj.inter
10.0

If the header scaling is undefined when we save the image, nibabel will try to
find an optimum slope and intercept to best preserve the precision of the data
in the output data type.  Because nibabel will set the scaling to undefined
when loading the image, or creating a new image, this is the default behavior.

.. testcleanup::

    os.chdir(pwd)
    import shutil
    shutil.rmtree(tmp_dir)

.. include:: links_names.txt

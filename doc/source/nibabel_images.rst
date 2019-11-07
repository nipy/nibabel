##############
Nibabel images
##############

A nibabel image object is the association of three things:

* an N-D array containing the image *data*;
* a (4, 4) *affine* matrix mapping array coordinates to coordinates in some
  RAS+ world coordinate space (:doc:`coordinate_systems`);
* image metadata in the form of a *header*.

****************
The image object
****************

First we load some libraries we are going to need for the examples:

.. testsetup::

    # Work in a temporary directory
    import os
    import tempfile
    pwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)

>>> import os
>>> import numpy as np

There is an example image in the nibabel distribution.

>>> from nibabel.testing import data_path
>>> example_file = os.path.join(data_path, 'example4d.nii.gz')

We load the file to create a nibabel *image object*:

>>> import nibabel as nib
>>> img = nib.load(example_file)

The object ``img`` is an instance of a nibabel image. In fact it is an
instance of a nibabel :class:`nibabel.nifti1.Nifti1Image`:

>>> img
<nibabel.nifti1.Nifti1Image object at ...>

As with any Python object, you can inspect ``img`` to see what attributes it
has.  We recommend using IPython tab completion for this, but here are some
examples of interesting attributes:

``dataobj`` is the object pointing to the image array data:

>>> img.dataobj
<nibabel.arrayproxy.ArrayProxy object at ...>

See :ref:`array-proxies` for more on why this is an array *proxy*.

``affine`` is the affine array relating array coordinates from the image data
array to coordinates in some RAS+ world coordinate system
(:doc:`coordinate_systems`):

>>> # Set numpy to print only 2 decimal digits for neatness
>>> np.set_printoptions(precision=2, suppress=True)

>>> img.affine
array([[ -2.  ,   0.  ,   0.  , 117.86],
       [ -0.  ,   1.97,  -0.36, -35.72],
       [  0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]])

``header`` contains the metadata for this inage.  In this case it is
specifically NIfTI metadata:

>>> img.header
<nibabel.nifti1.Nifti1Header object at ...>

****************
The image header
****************

The header of an image contains the image metadata.  The information in the
header will differ between different image formats.  For example, the header
information for a NIfTI1 format file differs from the header information for a
MINC format file.

Our image is a NIfTI1 format image, and it therefore has a NIfTI1 format
header:

>>> header = img.header
>>> print(header)                           # doctest: +SKIP
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

The header of any image will normally have the following methods:

* ``get_data_shape()`` to get the output shape of the image data array:

  >>> print(header.get_data_shape())
  (128, 96, 24, 2)

* ``get_data_dtype()`` to get the numpy data type in which the image data is
  stored (or will be stored if you save the image):

  >>> print(header.get_data_dtype())
  int16

* ``get_zooms()`` to get the voxel sizes in millimeters:

  >>> print(header.get_zooms())
  (2.0, 2.0, 2.19999..., 2000.0)

  The last value of ``header.get_zooms()`` is the time between scans in
  milliseconds; this is the equivalent of voxel size on the time axis.

********************
The image data array
********************

The image data array is a little more complicated, because the image array can
be stored in the image object as a numpy array or stored on disk for you to
access later via an *array proxy*.

.. _array-proxies:

Array proxies and proxy images
==============================

When you load an image from disk, as we did here, the data is likely to be
accessible via an array proxy.  An array proxy_ is not the array itself but
something that represents the array, and can provide the array when we ask for
it.

Our image does have an array proxy, as we have already seen:

>>> img.dataobj
<nibabel.arrayproxy.ArrayProxy object at ...>

The array proxy allows us to create the image object without immediately
loading all the array data from disk.

Images with an array proxy object like this one are called *proxy images*
because the image data is not yet an array, but the array proxy points to
(proxies) the array data on disk.

You can test if the image has a array proxy like this:

>>> nib.is_proxy(img.dataobj)
True

Array images
============

We can also create images from numpy arrays.  For example:

>>> array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
>>> affine = np.diag([1, 2, 3, 1])
>>> array_img = nib.Nifti1Image(array_data, affine)

In this case the image array data is already a numpy array, and there is no
version of the array on disk.  The ``dataobj`` property of the image is the
array itself rather than a proxy for the array:

>>> array_img.dataobj
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
<BLANKLINE>
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]], dtype=int16)
>>> array_img.dataobj is array_data
True

``dataobj`` is an array, not an array proxy, so:

>>> nib.is_proxy(array_img.dataobj)
False

Getting the image data the easy way
===================================

For either type of image (array or proxy) you can always get the data with the
:meth:`get_fdata() <nibabel.spatialimages.SpatialImage.get_fdata>` method.

For the array image, ``get_fdata()`` just returns the data array, if it's already the required floating point type (default 64-bit float).  If it isn't that type, ``get_fdata()`` casts it to one:

>>> image_data = array_img.get_fdata()
>>> image_data.shape
(2, 3, 4)
>>> image_data.dtype == np.dtype(np.float64)
True

The cast to floating point means the array is not the one attached to the image:

>>> image_data is array_img.dataobj
False

Here's an image backed by a floating point array:

>>> farray_img = nib.Nifti1Image(image_data.astype(np.float64), affine)
>>> farray_data = farray_img.get_fdata()
>>> farray_data.dtype == np.dtype(np.float64)
True

There was no cast, so the array returned is exactly the array attached to the
image:

>>> farray_data is farray_img.dataobj
True

For the proxy image, the ``get_fdata()`` method fetches the array data from
disk using the proxy, and returns the array.

>>> image_data = img.get_fdata()
>>> image_data.shape
(128, 96, 24, 2)

The image ``dataobj`` property is still a proxy object:

>>> img.dataobj
<nibabel.arrayproxy.ArrayProxy object at ...>

.. _proxies-caching:

Proxies and caching
===================

You may not want to keep loading the image data off disk every time
you call ``get_fdata()`` on a proxy image. By default, when you call
``get_fdata()`` the first time on a proxy image, the image object keeps a
cached copy of the loaded array.  The next time you call ``img.get_fdata()``,
the image returns the array from cache rather than loading it from disk again.

>>> data_again = img.get_fdata()

The returned data is the same (cached) copy we returned before:

>>> data_again is image_data
True

See :doc:`images_and_memory` for more details on managing image memory and
controlling the image cache.

.. _image-slicing:

Image slicing
=============

At times it is useful to manipulate an image's shape while keeping it in the
same coordinate system.
The ``slicer`` attribute provides an array-slicing interface to produce new
images with an appropriately adjusted header, such that the data at a given
RAS+ location is unchanged.

>>> cropped_img = img.slicer[32:-32, ...]
>>> cropped_img.shape
(64, 96, 24, 2)

The data is identical to cropping the data block directly:

>>> np.array_equal(cropped_img.get_fdata(), img.get_fdata()[32:-32, ...])
True

However, unused data did not need to be loaded into memory or scaled.
Additionally, the image affine was adjusted so that the X-translation is
32 voxels (64mm) less:

>>> cropped_img.affine
array([[ -2.  ,   0.  ,   0.  ,  53.86],
       [ -0.  ,   1.97,  -0.36, -35.72],
       [  0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]])

>>> img.affine - cropped_img.affine
array([[ 0.,  0.,  0., 64.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])

Another use for the slicer object is to choose specific volumes from a
time series:

>>> vol0 = img.slicer[..., 0]
>>> vol0.shape
(128, 96, 24)

Or a selection of volumes:

>>> img.slicer[..., :1].shape
(128, 96, 24, 1)
>>> img.slicer[..., :2].shape
(128, 96, 24, 2)

It is also possible to use an integer step when slicing, downsampling
the image without filtering.
Note that this *will induce artifacts* in the frequency spectrum
(`aliasing <wikipedia aliasing>`_) along any axis that is down-sampled.

>>> downsampled = vol0.slicer[::2, ::2, ::2]
>>> downsampled.header.get_zooms()
(4.0, 4.0, 4.399998)

Finally, an image can be flipped along an axis, maintaining an appropriate
affine matrix:

>>> nib.orientations.aff2axcodes(img.affine)
('L', 'A', 'S')
>>> ras = img.slicer[::-1]
>>> nib.orientations.aff2axcodes(ras.affine)
('R', 'A', 'S')
>>> ras.affine
array([[  2.  ,   0.  ,   0.  , 117.86],
       [  0.  ,   1.97,  -0.36, -35.72],
       [ -0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]])


******************
Loading and saving
******************

The ``save`` and ``load`` functions in nibabel should do all the work for you:

>>> nib.save(array_img, 'my_image.nii')
>>> img_again = nib.load('my_image.nii')
>>> img_again.shape
(2, 3, 4)

You can also use the ``to_filename`` method:

>>> array_img.to_filename('my_image_again.nii')
>>> img_again = nib.load('my_image_again.nii')
>>> img_again.shape
(2, 3, 4)

You can get and set the filename with ``get_filename()`` and
``set_filename()``:

>>> img_again.set_filename('another_image.nii')
>>> img_again.get_filename()
'another_image.nii'

***************************
Details of files and images
***************************

If an image can be loaded or saved on disk, the image will have an attribute
called ``file_map``.  ``img.file_map`` is a dictionary where the keys are the
names of the files that the image uses to load / save on disk, and the values
are ``FileHolder`` objects, that usually contain the filenames that the image
has been loaded from or saved to.  In the case of a NiFTI1 single file, this
is just a single image file with a ``.nii`` or ``.nii.gz`` extension:

>>> list(img_again.file_map)
['image']
>>> img_again.file_map['image'].filename
'another_image.nii'

Other file types need more than one file to make up the image.  The NiFTI1
pair type is one example.  NIfTI pair images have one file containing the
header information and another containing the image array data:

>>> pair_img = nib.Nifti1Pair(array_data, np.eye(4))
>>> nib.save(pair_img, 'my_pair_image.img')
>>> sorted(pair_img.file_map)
['header', 'image']
>>> pair_img.file_map['header'].filename
'my_pair_image.hdr'
>>> pair_img.file_map['image'].filename
'my_pair_image.img'

The older Analyze format also has a separate header and image file:

>>> ana_img = nib.AnalyzeImage(array_data, np.eye(4))
>>> sorted(ana_img.file_map)
['header', 'image']

It is the contents of the ``file_map`` that gets changed when you use
``set_filename`` or ``to_filename``:

>>> ana_img.set_filename('analyze_image.img')
>>> ana_img.file_map['image'].filename
'analyze_image.img'
>>> ana_img.file_map['header'].filename
'analyze_image.hdr'

.. testcleanup::

    os.chdir(pwd)
    import shutil
    shutil.rmtree(tmp_dir)

.. include:: links_names.txt

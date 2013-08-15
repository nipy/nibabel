########################
The nibabel image object
########################

A nibabel image object is the association of three things:

* an N-D array containing the image *data*
* a (4, 4) *affine* matrix mapping array coordinates to coordinates in some real
  world
* image metadata in the form of a *header*

*****************
Header and affine
*****************

There's an example image contained in the nibabel distribution

>>> import os
>>> import numpy as np
>>> np.set_printoptions(precision=2, suppress=True)

>>> import nibabel as nib
>>> from nibabel.testing import data_path
>>> example_file = os.path.join(data_path, 'example4d.nii.gz')
>>> img = nib.load(example_file)

You can get direct access to the *affine* and the *header* with:

>>> print(img.affine)
[[  -2.      0.      0.    117.86]
 [  -0.      1.97   -0.36  -35.72]
 [   0.      0.32    2.17   -7.25]
 [   0.      0.      0.      1.  ]]
>>> print(img.header)
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
vox_offset      : 416.0
scl_slope       : 1.0
scl_inter       : 0.0
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
quatern_b       : -1.9451068140294884e-26
quatern_c       : -0.9967085123062134
quatern_d       : -0.0810687392950058
qoffset_x       : 117.8551025390625
qoffset_y       : -35.72294235229492
qoffset_z       : -7.248798370361328
srow_x          : [  -2.      0.      0.    117.86]
srow_y          : [ -0.     1.97  -0.36 -35.72]
srow_z          : [ 0.    0.32  2.17 -7.25]
intent_name     : b''
magic           : b'n+1'

****************
Reading the data
****************

We defend the data from you a little, because we want to be able to load images
from disk without automatically loading all the data.  Images loaded like this
are called *proxy images* because the data in the image is not yet an array, but
an placeholder or *proxy* for the array.  You can get the object holding the
image data with:

>>> dataobj = img.dataobj

Because this image has been loaded from disk, ``dataobj`` is not the array
itself, but a *proxy* for the array that lets you get to the array data with:

>>> data = np.asarray(dataobj)
>>> data.shape
(128, 96, 24, 2)

If you created the image from an array in memory, ``dataobj`` will be the
data array:

>>> arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
>>> arr_img = nib.Nifti1Image(arr, np.eye(4))
>>> arr_img.dataobj is arr
True

For either type of image (array or proxy) you can always get the data with:

>>> data = img.get_data()
>>> data.shape
(128, 96, 24, 2)
>>> data = arr_img.get_data()
>>> data.shape
(2, 3, 4)

If you created ``img`` using an array in memory, ``img.get_data()`` just returns
the data array:

>>> data = arr_img.get_data()
>>> data is arr_img.dataobj
True

If you loaded ``img`` from disk, and have a proxy image, then the the
``get_data()`` method gets the data from the proxy, and returns the array.

>>> proxy_img = nib.load(example_file)
>>> type(proxy_img.dataobj)
<class 'nibabel.arrayproxy.ArrayProxy'>
>>> data = proxy_img.get_data()
>>> data.shape
(128, 96, 24, 2)
>>> data is proxy_img.dataobj
False

After a call to ``get_data()``, the proxy image keeps a cached copy of the
loaded array, so the next time you call ``img.get_data()``, we do not have to
load the array off the disk again.

>>> data_again = proxy_img.get_data()
>>> data is data_again
True

If you call ``img.get_data()`` on a proxy image, the image object will get much
larger in memory, because the image now stores a copy of the loaded array.  If
you want to avoid this memory load you can:

* use ``np.asarray(img.dataobj)`` instead of ``img.get_data()`` or
* run ``img.uncache()`` after calling ``img.get_data()``.  This deletes the copy
  of the array inside the image, so the next time you call ``img.get_data()``
  the image has to load the data from disk again.

Here is ``uncache`` in action:

>>> data_again = proxy_img.get_data()
>>> data is data_again
True
>>> proxy_img.uncache()
>>> data_once_more = proxy_img.get_data()
>>> data_once_more is data_again
False

This means you need to be careful when you modify arrays returned by
``get_data()`` on proxy images, because ``uncache`` will then change the result
you get back from ``get_data()``:

>>> data = proxy_img.get_data()
>>> data[0, 0, 0, 0]
0
>>> data[0, 0, 0, 0] = 99
>>> data_again = proxy_img.get_data()
>>> data_again[0, 0, 0, 0]
99
>>> proxy_img.uncache()
>>> data_once_more = proxy_img.get_data()
>>> data_once_more[0, 0, 0, 0]
0

******************
Loading and saving
******************

The ``save`` and ``load`` functions in nibabel should do all the work for you:

>>> img = nib.load(example_file)
>>> img.shape
(128, 96, 24, 2)
>>> import tempfile
>>> temp_fname = tempfile.mktemp('.nii')
>>> nib.save(img, temp_fname)
>>> img_again = nib.load(temp_fname)
>>> img_again.shape
(128, 96, 24, 2)
>>> os.unlink(temp_fname)

You can also use the ``to_filename`` method:

>>> temp_fname = tempfile.mktemp('.nii')
>>> img.to_filename(temp_fname)
>>> img_again = nib.load(temp_fname)
>>> img_again.shape
(128, 96, 24, 2)
>>> os.unlink(temp_fname)

You can get and set the filename with ``get_filename()`` and ``set_filename()``:

>>> img.set_filename('my_image.nii')
>>> img.get_filename()
'my_image.nii'

***************************
Details of files and images
***************************

If an image can be loaded or saved on disk, the image will have an attribute
called ``file_map``.  ``img.file_map`` is a dictionary where the keys are the
names of the files that the image uses to load / save on disk, and the values
are ``FileHolder`` objects, that usually contain the filenames that the image
has been loaded from or saved to.  In the case of a NiFTI1 single file, this is
just a single image file with a ``.nii`` extension:

>>> list(proxy_img.file_map)
['image']
>>> proxy_img.file_map['image'].filename
'/Users/mb312/dev_trees/nibabel/nibabel/tests/data/example4d.nii.gz'

Other file types need more than one file to make up the image.  The NiFTI1 pair
type is one example:

>>> pair_img = nib.Nifti1Pair(arr, np.eye(4))
>>> sorted(pair_img.file_map)
['header', 'image']

The older Analyze format is another:

>>> ana_img = nib.AnalyzeImage(arr, np.eye(4))
>>> sorted(ana_img.file_map)
['header', 'image']

It is ``img.file_map`` that gets changed when you use ``set_filename`` or
``to_filename``:

>>> ana_img.set_filename('another_image.img')
>>> ana_img.file_map['image'].filename
'another_image.img'
>>> ana_img.file_map['header'].filename
'another_image.hdr'

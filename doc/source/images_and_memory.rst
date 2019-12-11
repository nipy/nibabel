#################
Images and memory
#################

We saw in :doc:`nibabel_images` that images loaded from disk are usually
*proxy images*. Proxy images are images that have a ``dataobj`` property that
is not a numpy array, but an *array proxy* that can fetch the array data from
disk.

>>> import os
>>> import numpy as np
>>> from nibabel.testing import data_path
>>> example_file = os.path.join(data_path, 'example4d.nii.gz')

>>> import nibabel as nib
>>> img = nib.load(example_file)
>>> img.dataobj
<nibabel.arrayproxy.ArrayProxy object at ...>

Nibabel does not load the image array from the proxy when you ``load`` the
image.  It waits until you ask for the array data.  The standard way to ask
for the array data is to call the ``get_fdata()`` method:

>>> data = img.get_fdata()
>>> data.shape
(128, 96, 24, 2)

We also saw in :ref:`proxies-caching` that this call to ``get_fdata()`` will
(by default) load the array data into an internal image cache.  The image
returns the cached copy on the next call to ``get_fdata()``:

>>> data_again = img.get_fdata()
>>> data is data_again
True

This behavior is convenient if you want quick and repeated access to the image
array data.  The down-side is that the image keeps a reference to the image
data array, so the array can't be cleared from memory until the image object
gets deleted.  You might prefer to keep loading the array from disk instead of
keeping the cached copy in the image.

This page describes ways of using the image array proxies to save memory and
time.

***************************************************
Using ``in_memory`` to check the state of the cache
***************************************************

You can use the ``in_memory`` property to check if the image has cached the
array.

The ``in_memory`` property is always True for array images, because the image
data is always an array in memory:

>>> array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
>>> affine = np.diag([1, 2, 3, 1])
>>> array_img = nib.Nifti1Image(array_data, affine)
>>> array_img.in_memory
True

For a proxy image, the ``in_memory`` property is False when the array is not
in cache, and True when it is in cache:

>>> img = nib.load(example_file)
>>> img.in_memory
False
>>> data = img.get_fdata()
>>> img.in_memory
True


*****************
Using ``uncache``
*****************

As y'all know, the proxy image has the array in cache, ``get_fdata()`` returns
the cached array:

>>> data_again = img.get_fdata()
>>> data_again is data  # same array returned from cache
True

You can uncache a proxy image with the ``uncache()`` method:

>>> img.uncache()
>>> img.in_memory
False
>>> data_once_more = img.get_fdata()
>>> data_once_more is data  # a new copy read from disk
False

``uncache()`` has no effect if the image is an array image, or if the cache is
already empty.

You need to be careful when you modify arrays returned by ``get_fdata()`` on
proxy images, because ``uncache`` will then change the result you get back
from ``get_fdata()``:

>>> proxy_img = nib.load(example_file)
>>> data = proxy_img.get_fdata()  # array cached and returned
>>> data[0, 0, 0, 0]
0.0
>>> data[0, 0, 0, 0] = 99  # modify returned array
>>> data_again = proxy_img.get_fdata()  # return cached array
>>> data_again[0, 0, 0, 0]  # cached array modified
99.0

So far the proxy image behaves the same as an array image.  ``uncache()`` has
no effect on an array image, but it does have an effect on the returned array
of a proxy image:

>>> proxy_img.uncache()  # cached array discarded from proxy image
>>> data_once_more = proxy_img.get_fdata()  # new copy of array loaded
>>> data_once_more[0, 0, 0, 0]  # array modifications discarded
0.0

*************
Saving memory
*************

Uncache the array
=================

If you do not want the image to keep the array in its internal cache, you can
use the ``uncache()`` method:

>>> img.uncache()

Use the array proxy instead of ``get_fdata()``
==============================================

The ``dataobj`` property of a proxy image is an array proxy.  We can ask the
proxy to return the array directly by passing ``dataobj`` to the numpy
``asarray`` function:

>>> proxy_img = nib.load(example_file)
>>> data_array = np.asarray(proxy_img.dataobj)
>>> type(data_array)
<... 'numpy.ndarray'>

This also works for array images, because ``np.asarray`` returns the array:

>>> array_img = nib.Nifti1Image(array_data, affine)
>>> data_array = np.asarray(array_img.dataobj)
>>> type(data_array)
<... 'numpy.ndarray'>

If you want to avoid caching you can avoid ``get_fdata()`` and always use
``np.asarray(img.dataobj)``.

Use the ``caching`` keyword to ``get_fdata()``
==============================================

The default behavior of the ``get_fdata()`` function is to always fill the
cache, if it is empty.  This corresponds to the default ``'fill'`` value
to the ``caching`` keyword.  So, this:

>>> proxy_img = nib.load(example_file)
>>> data = proxy_img.get_fdata()  # default caching='fill'
>>> proxy_img.in_memory
True

is the same as this:

>>> proxy_img = nib.load(example_file)
>>> data = proxy_img.get_fdata(caching='fill')
>>> proxy_img.in_memory
True

Sometimes you may want to avoid filling the cache, if it is empty. In this
case, you can use ``caching='unchanged'``:

>>> proxy_img = nib.load(example_file)
>>> data = proxy_img.get_fdata(caching='unchanged')
>>> proxy_img.in_memory
False

``caching='unchanged'`` will leave the cache full if it is already full.

>>> data = proxy_img.get_fdata(caching='fill')
>>> proxy_img.in_memory
True
>>> data = proxy_img.get_fdata(caching='unchanged')
>>> proxy_img.in_memory
True

See the :meth:`get_fdata() docstring
<nibabel.spatialimages.SpatialImage.get_fdata>` for more detail.

**********************
Saving time and memory
**********************

You can use the array proxy to get slices of data from disk in an efficient
way.

The array proxy API allows you to do slicing on the proxy.  In most cases this
will mean that you only load the data from disk that you actually need, often
saving both time and memory.

For example, let us say you only wanted the second volume from the example
dataset.  You could do this:

>>> proxy_img = nib.load(example_file)
>>> data = proxy_img.get_fdata()
>>> data.shape
(128, 96, 24, 2)
>>> vol1 = data[..., 1]
>>> vol1.shape
(128, 96, 24)

The problem is that you had to load the whole data array into memory before
throwing away the first volume and keeping the second.

You can use array proxy slicing to do this more efficiently:

>>> proxy_img = nib.load(example_file)
>>> vol1 = proxy_img.dataobj[..., 1]
>>> vol1.shape
(128, 96, 24)

The slicing call in ``proxy_img.dataobj[..., 1]`` will only load the data from
disk that you need to fill the memory of ``vol1``.

.. include:: links_names.txt

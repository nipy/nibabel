.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the NiBabel package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _gettingstarted:

***************
Getting Started
***************

NiBabel supports an ever growing collection of neuroimaging file formats. Every
file format format has its own features and pecularities that need to be taken
care of to get the most out of it. To this end, NiBabel offers both high-level
format-independent access to neuroimages, as well as an API with various levels
of format-specific access to all available information in a particular file
format.  The following examples show some of NiBabel's capabilities and give
you an idea of the API.

For more detail on the API, see :doc:`nibabel_images`.

When loading an image, NiBabel tries to figure out the image format from the
filename. An image in a known format can easily be loaded by simply passing its
filename to the ``load`` function.

To start the code examples, we load some useful libraries:

>>> import os
>>> import numpy as np

Then we fine the nibabel directory containing the example data:

>>> from nibabel.testing import data_path

There is a NIfTI file in this directory called ``example4d.nii.gz``:

>>> example_filename = os.path.join(data_path, 'example4d.nii.gz')

Now we can import nibabel and load the image:

>>> import nibabel as nib
>>> img = nib.load(example_filename)

A NiBabel image knows about its shape:

>>> img.shape
(128, 96, 24, 2)

It also records the data type of the data as stored on disk. In this case the
data on disk are 16 bit signed integers:

>>> img.get_data_dtype() == np.dtype(np.int16)
True

The image has an affine transformation that determines the world-coordinates of
the image elements (see :doc:`coordinate_systems`):

>>> img.affine.shape
(4, 4)

This information is available without the need to load anything of the main
image data into the memory. Of course there is also access to the image data as
a NumPy_ array

>>> data = img.get_fdata()
>>> data.shape
(128, 96, 24, 2)
>>> type(data)
<... 'numpy.ndarray'>

The complete information embedded in an image header is available via a
format-specific header object.

>>> hdr = img.header

In case of this NIfTI_ file it allows accessing all NIfTI-specific information,
e.g.

>>> hdr.get_xyzt_units()
('mm', 'sec')

Corresponding "setter" methods allow modifying a header, while ensuring its
compliance with the file format specifications.

In some situations we need even more flexibility and, for those with great
courage, NiBabel also offers access to the raw header information

>>> raw = hdr.structarr
>>> raw['xyzt_units']
array(10, dtype=uint8)

This lowest level of the API is designed for people who know the file format
well enough to work with its internal data, and comes without any safety-net.

Creating a new image in some file format is also easy. At a minimum it only
needs some image data and an image coordinate transformation (affine):

>>> import numpy as np
>>> data = np.ones((32, 32, 15, 100), dtype=np.int16)
>>> img = nib.Nifti1Image(data, np.eye(4))
>>> img.get_data_dtype() == np.dtype(np.int16)
True
>>> img.header.get_xyzt_units()
('unknown', 'unknown')

In this case, we used the identity matrix as the affine transformation. The
image header is initialized from the provided data array (i.e. shape, dtype)
and all other values are set to resonable defaults.

Saving this new image to a file is trivial:

>>> nib.save(img, os.path.join('build', 'test4d.nii.gz'))  # doctest: +SKIP

This short introduction only gave a quick overview of NiBabel's capabilities.
Please have a look at the :ref:`api` for more details about supported file
formats and their features.

.. include:: links_names.txt

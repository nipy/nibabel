.. -*- rst -*-

#############################################################
Keeping track of whether images have been modified since load
#############################################################

*******
Summary
*******

This is a discussion of a missing feature in nibabel: the ability to keep
track of whether an image object in memory still corresponds to an image file
(or files) on disk.

**********
Motivation
**********

We may need to know whether the image in memory corresponds to the image file
on disk.

For example, we often need to get filenames for images when passing
images to external programs. Imagine a realignment, in this case, in nipy_
(the package)::

   import nipy
   img1 = nibabel.load('meanfunctional.nii')
   img2 = nibabel.load('anatomical.nii')
   realigner = nipy.interfaces.fsl.flirt()
   params = realigner.run(source=img1, target=img2)

In ``nipy.interfaces.fsl.flirt.run`` there may at some point be calls like::

   source_filename = nipy.as_filename(source_img)
   target_filename = nipy.as_filename(target_img)

As the authors of the ``flirt.run`` method, we need to make sure that the
``source_filename`` corresponds to the ``source_img``.

Of course, in the general case, if ``source_img`` has no corresponding
filename (from ``source_img.get_filename()``, then we will have to save a copy
to disk, maybe with a temporary filename, and return that temporary name as
``source_filename``.

In our particular case, ``source_img`` does have a filename
(``meanfunctional.nii``).  We would like to return that as
``source_filename``.  The question is, how can we be sure that the user has
done nothing to ``source_img`` to make it diverge from its original state?
Could ``source_img`` have diverged, in memory, from the state recorded in
``meantunctional.nii``?

If the image and file have not diverged, we return ``meanfunctional.nii`` as
the ``source_filename``, otherwise we will have to do something like::

   import tempfile
   fname = tempfile.mkstemp('.nii')
   img = source_img.to_filename(fname)

and return ``fname`` as ``source_filename``.

Another situation where we might like to pass around image objects that are
known to correspond to images on disk is when working in parallel. A set of
nodes may have fast common access to a filesystem on which the images are
stored.  If a master is farming out images to nodes, a master node
distribution jobs to workers might want to check if the image was identical to
something on file and pass around a lightweight (proxied) image (with the data
not loaded into memory), relying on the node pulling the image from disk when
it uses it.

***********************
Possible implementation
***********************

One implementation is to have ``dirty`` flag, which, if set, would tell
you that the image might not correspond to the disk file.  We set this
flag when anyone asks for the data, on the basis that the user may then
do something to the data and you can't know if they have::

   img = nibabel.load('some_image.nii')
   data = img.get_fdata()
   data[:] = 0
   img2 = nibabel.load('some_image.nii')
   assert not np.all(img2.get_fdata() == img.get_fdata())

The image consists of the data, the affine and a header.  In order to
keep track of the header and affine, we could cache them when loading
the image::

   img = nibabel.load('some_image.nii')
   hdr = img.header
   assert img._cache['header'] == img.header
   hdr.set_data_dtype(np.complex64)
   assert img._cache['header'] != img.header

When we need to know whether the image object and image file correspond, we
could check the current header and current affine (the header may be separate
from the affine for an SPM Analyze image) against their cached copies, if they
are the same and the 'dirty' flag has not been set by a previous call to
``get_fdata()``, we know that the image file does correspond to the image
object.

This may be OK for small bits of memory like the affine and the header,
but would quickly become prohibitive for larger image metadata such as
large nifti header extensions.  We could just always assume that images
with large header extensions are *not* the same as for on disk.

The user might be able to override the result of these checks directly::

   img = nibabel.load('some_image.nii')
   assert img.is_dirty == False
   hdr = img.header
   hdr.set_data_dtype(np.complex64)
   assert img.is_dirty == True
   img.is_dirty == False

The checks are magic behind the scenes stuff that do some safe optimization
(in the sense that we are not re-saving the data if that is not necessary),
but drops back to the default (re-saving the data) if there is any
uncertainty, or the cost is too high to be able to check.

.. include:: ../links_names.txt

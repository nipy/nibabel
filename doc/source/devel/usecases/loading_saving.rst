.. -*- rst -*- 

===============================================
 Loading and saving image files from / to disk
===============================================

Need for lightweight version of image object
============================================

Images can be very large, and therefore take up a large amount of
memory, or processor / system time when loading the image from disk.

We would like to be able to create and look at an image (and not the
image data) without incurring the full cost of the load from disk.

So, the image object can be lightweight::

   import nibabel
   img = nibabel.load('some_image.nii')

In this case, the ``img`` object has not yet loaded the data from disk.
We may use the ``get_data()`` method to get the data, or something fancy
with the ``data`` attribute to delay loading until we access the data.

Keeping track of the image and the disk file
============================================

We may need to know whether the image in memory corresponds to the image
file on disk.

For example, we often need to get filenames for images when passing
images to external programs. Imagine a realignment::

   import nipy
   img1 = nibabel.load('meanfunctional.nii')
   img2 = nibabel.load('anatomical.nii')
   realigner = nipy.interfaces.fsl.flirt()
   params = realigner.run(source=img1, target=img2)

In ``nipy.interfaces.fsl.flirt.run`` there will at some point be calls
like::

   source_filename = nipy.as_filename(source_img)
   target_filename = nipy.as_filename(target_img)

We need to make sure that the ``source_filename`` corresponds to the
``source_img``.  When we pass the source image, this will be true::

   source_img.get_filename() == 'meanfunctional.nii'

We need to know whether the ``source_img`` still corresponds exactly to
``meanfunctional.nii``.  If so, we return ``meanfunctional.nii`` as the
``source_filename``, otherwise we will have to do something like::

   import tempfile
   fname = tempfile.mkstemp('.nii')
   img = source_img.to_filename(fname)

Another application for this scheme is when working in parallel. A set
of nodes may have fast common access to a filesystem on which the images
are stored.  If a master is farming out images to nodes, a master might
want to check if the image was identical to something on file and pass a
lightweight shell round the image (with the data not loaded into
memory), relying on the node pulling the image from disk when it uses
it.

One implementation is to have ``dirty`` flag, which, if set, would tell
you that the image might not correspond to the disk file.  We set this
flag when anyone asks for the data, on the basis that the user may then
do something to the data and you can't know if they have::

   img = nibabel.load('some_image.nii')
   data = img.get_data()
   data[:] = 0
   img2 = nibabel.load('some_image.nii')
   assert not np.all(img2.get_data() == img.get_data())

The image consists of the data, the affine and a header.  In order to
keep track of the header and affine, we could cache them when loading
the image::

   img = nibabel.load('some_image.nii')
   hdr = img.get_header()
   assert img._cache['header'] == img.get_header()
   hdr.set_data_dtype(np.complex64)
   assert img._cache['header'] != img.get_header()

When asking to return a filename, or similar, check the current header
and current affine (the header may be separate from the affine for an
SPM image) against their cached copies, if they are the same and the
'dirty' flag is not set, we know that the filename is OK.

This may be OK for small bits of memory like the affine and the header,
but would quickly become prohibitive for larger image metadata such as
large nifti header extensions.  We could just always assume that images
with large header extensions are *not* the same as for on disk.

The user can override the result of these checks directly::

   img = nibabel.load('some_image.nii')
   assert img.is_dirty == False
   hdr = img.get_header()
   hdr.set_data_dtype(np.complex64)
   assert img.is_dirty == True
   img.is_dirty == False

The checks are magic behind the scenes stuff that do some safe
optimization (in the sense that we are not resaving the data if that is
not necessary), but drops back to the default (resaving the data) if
there is any uncertainty, or the cost is too high to be able to check.


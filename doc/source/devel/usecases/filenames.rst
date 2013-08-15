========================
 Usecases for filenames
========================

Here we describe the uses that we might make of a filename attached to
an image.

The images need not ever be associated with files.  We can imagine other
storage for images, that are not in files, but - for example -
databases, or file-like objects.

An image can exist without a filename, when we create the image in memory:

   >>> import numpy as np
   >>> import nibabel as nib
   >>> data = np.arange(24).reshape((2,3,4))
   >>> img = nib.Nifti1Image(data, np.eye(4))

Some images will have filenames.  For example, if we load an image from
disk, there is a natural association with the filename.  To show this,
we first get an example filename from the nibabel code:

   >>> from nibabel.testing import data_path, pjoin
   >>> fname = pjoin(data_path, 'example4d.nii.gz')

Now we can load from the filename:

   >>> img = nib.load(fname)

We could get the filename with one of::

   >>> img.filename == fname
   True

or::

   >>> img.get_filename() == fname
   True

The argument for the second is that we will also want to set the
filename.  The setter will want to expand filenames that are
interpretable but not full.  For example, the following might all be OK
for an image ``somefile.img``:

   >>> img = nib.Spm99AnalyzeImage(np.zeros((2,3,4)), np.eye(4))
   >>> img.set_filename('somefile')
   >>> img.set_filename('somefile.hdr')
   >>> img.set_filename('somefile.img')

If that is true, imagine a property ``img.filename`` which encapsulates
the get and set calls.  It would have the following annoying behavior::

   img.filename = 'somefile'
   assert img.filename == 'somefile.img'
   img.filename = 'somefile.hdr'
   assert img.filename == 'somefile.img'

In this case I think getters and setters are more pleasant.

The filename should probably follow the image around even if the image
in memory has diverged very far from the image on disk:

   >>> img = nib.load(fname)
   >>> idata = img.get_data()
   >>> idata[:] = 0
   >>> hdr = img.get_header()
   >>> hdr['descrip'] = 'something strange'
   >>> img.get_filename() == fname
   True

This also allows saving into - and loading from - already open file-like
objects.  The main use for this is testing, but we could imagine
wrapping some other storage mechanism in file-like objects:

   >>> from StringIO import StringIO
   >>> img_klass = nib.Nifti1Image
   >>> file_map = img_klass.make_file_map({'image':StringIO()})
   >>> img = img_klass(data, np.eye(4), file_map=file_map)
   >>> img.to_file_map()
   
In this last case, there is obviously not a valid filename:

   >>> assert img.get_filename() is None
   



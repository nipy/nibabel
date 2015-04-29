.. -*- mode: rst -*-

##################################################
Relationship between images and io implementations
##################################################

********************
Summary and sign-off
********************

These were some meditations about splitting the image into two API parts.

The first part would be the lower level IO implementation.  This part is
rather like a fusion of the :class:`.Header` and :class:`.ArrayProxy` objects
in current nibabel.  It takes care of lower level details like i/o data dtype,
shape, offset, and it might help with slicing to get the data.  On top of that
would be a high level interface implementing ``load``, ``save``, ``filename``,
``data``.  The top-level image also had the novel idea of a ``mode`` parameter
which, if ``'r'``, would raise an error on attempting to ``save``.

******
Images
******

An image houses the association of the:

* data array
* affine
* output space
* metadata
* mode

These are straightforward attributes, and have no necessary relationship
to stuff on disk.

By ''disk'', we mean, file-like objects - not necessarily on disk.

The *io implementation* manages the relationship of images and stuff on
disk.

Specifically, it manages ``load`` of images from disk, and ``save`` of
images to disk.

The user does not see the io implementation unless they ask to.  In
standard use of images they will not need to do this.

******************
IO implementations
******************

By use case.

::

    Creating array image, saving

    >>> import tempfile
    >>> from nibabel.images import Image
    >>> from nibabel import load, save
    >>> fp, fname = tempfile.mkstemp('.nii')
    >>> data = np.arange(24).reshape((2,3,4))
    >>> img = Image(data)
    >>> img.filename is None
    True
    >>> img.save()
    Traceback (most recent call last):
       ...
    ImageError: no filespec to save to
    >>> save(img)
    Traceback (most recent call last):
       ...
    ImageError: no filespec to save to
    >>> img2 = save(img, 'some_image.nii') # type guessed from filename
    >>> img2.filename == fname
    True
    >>> img.filename is None # still
    True
    >>> img.filename = 'some_filename.nii' # read only property
    Traceback (most recent call last):
       ...
    AttributeError: can't set attribute

    Load, futz, save

    >>> img3 = load(fname, mode='r')
    >>> img3.filename == fname
    True
    >>> np.all(img3.data == data)
    True
    >>> img3.data[0,0] = 99
    >>> img3.save()
    Traceback (most recent call last):
       ...
    ImageError: trying to write to read only image
    >>> img3.mode = 'rw'
    >>> img3.save()
    >>> load(img4)
    >>> img4.mode # 'r' is the default
    'r'
    >>> mod_data = data.copy()
    >>> mod_data[0,0] = 99
    >>> np.all(img4.data = mod_data)
    True
    
    Prepare image for later writing

    >>> img5 = Image(np.zeros(2,3,4)) 
    >>> fp, fname2 = tempfile.mkstemp('.nii')
    >>> img5.set_filespec(fname2)
    >>> # then do some things to the image
    >>> img5.save()

    This is an example where you do need the io API

    >>> from nibabel.ioimps import guessed_imp
    >>> fp, fname3 = tempfile.mkstemp('.nii')
    >>> ioimp = guessed_imp(fname3)
    >>> ioimp.set_data_dtype(np.float)
    >>> ioimp.set_data_shape((2,3,4)) # set_data_shape method
    >>> slice_def = (slice(None), slice(None), 0)
    >>> ioimp.write_slice(data[slice_def], slice_def) # write_slice method
    >>> slice_def = (2, 3, 1)
    >>> ioimp.write_slice(data[slice_def], slice_def) # write_slice method
    Traceback (most recent call last):
       ...
    ImageIOError: data write is not contiguous
    
    
    

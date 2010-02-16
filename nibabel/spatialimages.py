''' Very simple spatial image class

The image class maintains the association between a 3D (or greater)
array, and an affine transform that maps voxel coordinates to some real
world space.  It also has a ``header`` - some standard set of meta-data
that is specific to the image format - and ``extra`` - a dictionary
container for any other metadata.

It has attributes:

   * extra
    
methods:

   * .get_data()
   * .get_affine()
   * .get_header()
   * .get_shape()
   * .set_shape(shape)
   * .to_filename(fname) - writes data to filename(s) derived from
     ``fname``, where the derivation may differ between formats.
   * to_files() - save image to files with which the image is already
     associated.

classmethods:

   * from_filename(fname) - make instance by loading from filename
   * instance_to_filename(img, fname) - save ``img`` instance to
     filename ``fname``.

There are several ways of writing data.
=======================================

There is the usual way, which is the default::

    img.to_filename(fname)

and that is, to take the data encapsulated by the image and cast it to
the datatype the header expects, setting any available header scaling
into the header to help the data match.

You can load the data into an image from file with::

   img.from_filename(fname)

The image stores its associated files in its ``files`` attribute.  In
order to just save an image, for which you know there is an associated
filename, or other storage, you can do::

   img.to_files()

You can get the data out again with of::

    img.get_data()

Less commonly, for some image types that support it, you might want to
fetch out the unscaled array via the header::

    unscaled_data = img.get_unscaled_data()

Analyze-type images (including nifti) support this, but others may not
(MINC, for example).

Sometimes you might to avoid any loss of precision by making the
data type the same as the input::

    hdr = img.get_header()
    hdr.set_data_dtype(data.dtype)
    img.to_filename(fname)

Files interface
===============

The image has an attribute ``files``.  This is a mapping, that has keys
corresponding to the file types that an image needs for storage.  For
example, the Analyze data format needs an ``image`` and a ``header``
file type for storage:

   >>> import nibabel as nib
   >>> data = np.arange(24).reshape((2,3,4))
   >>> img = nib.AnalyzeImage(data, np.eye(4))
   >>> sorted(img.files)
   ['header', 'image']

The values of ``files`` are not in fact files but objects with
attributes ``filename``, ``fileobj`` and ``pos``.

The reason for this interface, is that the contents of files has to
contain enough information so that an existing image instance can save
itself back to the files pointed to in ``files``.   When a file holder
holds active file-like objects, then these may be affected by the
initial file read; in this case, the contains file-like objects need to
carry the position at which a write (with ``to_files``) should place the
data.   The ``files`` contents should therefore be such, that this will
work:

   >>> # write an image to files
   >>> from StringIO import StringIO
   >>> files = nib.AnalyzeImage.make_files()
   >>> files['image'].fileobj = StringIO()
   >>> files['header'].fileobj = StringIO()
   >>> img = nib.AnalyzeImage(data, np.eye(4))
   >>> img.files = files
   >>> img.to_files()
   >>> # read it back again from the written files
   >>> img2 = nib.AnalyzeImage.from_files(files)
   >>> np.all(img2.get_data() == data)
   True
   >>> # write, read it again
   >>> img2.to_files()
   >>> img3 = nib.AnalyzeImage.from_files(files)
   >>> np.all(img3.get_data() == data)
   True

'''

import warnings

import numpy as np

from nibabel.filename_parser import types_filenames, TypesFilenamesError
from nibabel.fileholders import FileHolder

class ImageDataError(Exception):
    pass


class SpatialImage(object):
    _header_class = dict
    files_types = (('image', None),)
    _compressed_exts = ()
    
    ''' Template class for images '''
    def __init__(self, data, affine, header=None, extra=None, files=None):
        ''' Initialize image

        Parameters
        ----------
        data : array-like
           image data.  It should be some object that retuns an array
           from ``np.asanyarray``
        affine : (4,4) array
           homogenous affine giving relationship between voxel
           coordinates and world coordinates
        header : None or mapping or header instance, optional
           metadata for this image format
        extra : None or mapping, optional
           metadata to associate with image that cannot be stored in the
           metadata of this image type
        files : mapping, optional
           mapping giving file information for this image format
        '''
        self._data = data
        self._affine = affine
        if extra is None:
            extra = {}
        self.extra = extra
        self._set_header(header)
        if files is None:
            files = self.__class__.make_files()
        self.files = files
        
    def __str__(self):
        shape = self.get_shape()
        affine = self.get_affine()
        return '\n'.join((
                str(self.__class__),
                'data shape %s' % (shape,),
                'affine: ',
                '%s' % affine,
                'metadata:',
                '%s' % self._header))

    def get_data(self):
        if self._data is None:
            raise ImageDataError('No data in this image')
        return np.asanyarray(self._data)

    def get_shape(self):
        if self._data:
            return self._data.shape

    def get_data_dtype(self):
        return self._header.get_data_dtype()
    
    def set_data_dtype(self, dtype):
        self._header.set_data_dtype(dtype)

    def get_affine(self):
        return self._affine

    def get_header(self):
        return self._header

    def _set_header(self, header=None):
        raise NotImplementedError

    def get_filename(self):
        ''' Fetch the image filename

        Parameters
        ----------
        None

        Returns
        -------
        fname : None or str
           Returns None if there is no filename, or a filename string.
           If an image may have several filenames assoctiated with it
           (e.g Analyze ``.img, .hdr`` pair) then we return the more
           characteristic filename (the ``.img`` filename in the case of
           Analyze') 
        '''
        # which filename is returned depends on the ordering of the
        # 'files_types' class attribute - we return the name
        # corresponding to the first in that tuple
        characteristic_type = self.files_types[0][0]
        return self.files[characteristic_type].filename
        
    def set_filename(self, filename):
        ''' Sets the files in the object from a given filename

        The different image formats may check whether the filename has
        an extension characteristic of the format, and raise an error if
        not. 
        
        Parameters
        ----------
        filename : str
           If the image format only has one file associated with it,
           this will be the only filename set into the image ``.files``
           attribute. Otherwise, the image instance will try and guess
           the other filenames from this given filename.
        '''
        self.files = self.__class__.filespec_to_files(filename)

    @classmethod
    def from_filename(klass, filename):
        files = klass.filespec_to_files(filename)
        return klass.from_files(files)
    
    @classmethod
    def from_filespec(klass, img, filespec):
        warnings.warn('``from_filespec`` class method is deprecated\n'
                      'Please use the ``from_filename`` class method '
                      'instead',
                      DeprecationWarning)
        klass.from_filespec(filespec)

    def from_files(klass, files):
        raise NotImplementedError

    @classmethod
    def filespec_to_files(klass, filespec):
        try:
            filenames = types_filenames(filespec,
                                        klass.files_types,
                                        trailing_suffixes=klass._compressed_exts)
        except TypesFilenamesError:
            raise ValueError('Filespec "%s" does not look right for '
                             'class %s ' % (filespec, klass))
        files = {}
        for key, fname in filenames.items():
            files[key] = FileHolder(filename=fname)
        return files

    def to_filename(self, filename):
        ''' Write image to files implied by filename string

        Paraameters
        -----------
        filename : str
           filename to which to save image.  We will parse `filename`
           with ``filespec_to_files`` to work out names for image,
           header etc.

        Returns
        -------
        None
        '''
        self.files = self.filespec_to_files(filename)
        self.to_files()

    def to_filespec(self, filename):
        warnings.warn('``to_filespec`` is deprecated, please '
                      'use ``to_filename`` instead',
                      DeprecationWarning)
        self.to_filename(filename)

    def to_files(self):
        raise NotImplementedError

    @classmethod
    def make_files(klass, mapping=None):
        ''' Class method to make files holder for this image type

        Parameters
        ----------
        mapping : None or mapping, optional
           mapping with keys corresponding to image file types (such as
           'image', 'header' etc, depending on image class) and values
           that are filenames or file-like.  Default is None
           
        Returns
        -------
        files : dict
           dict with string keys given by first entry in tuples in
           sequence klass.files_types, and values of type FileHolder,
           where FileHolder objects have default values, other than
           those given by `mapping`
        '''
        if mapping is None:
            mapping = {}
        files = {}
        for key, ext in klass.files_types:
            files[key] = FileHolder()
            mapval = mapping.get(key, None)
            if isinstance(mapval, basestring):
                files[key].filename = mapval
            elif hasattr(mapval, 'tell'):
                files[key].fileobj = mapval
        return files

    @classmethod
    def load(klass, filename):
        return klass.from_filename(filename)

    @classmethod
    def save(klass, img, filename):
        warnings.warn('``save`` class method is deprecated\n'
                      'You probably want the ``to_filename`` instance '
                      'method, or the module-level ``save`` function',
                      DeprecationWarning)
        klass.instance_to_filename(img, filename)

    @classmethod
    def instance_to_filename(klass, img, filename):
        ''' Save `img` in our own format, to name implied by `filename`

        This is a class method

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``get_data``, ``get_affine``, ``get_header`` and
           ``extra``.
        filename : str
           Filename, implying name to which to save image.
        '''
        img = klass.from_image(img)
        img.to_filename(filename)
        
    @classmethod
    def from_image(klass, img):
        ''' Create new instance of own class from `img`

        This is a class method.  Note that, for this general method, we
        throw away the header from the image passed in, on the basis
        that we cannot predict whether it is convertible in general.
        Sub-classes can override this class method to try and use the
        information from the passed header. 
        
        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``get_data``, ``get_affine``,  and
           ``extra``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        '''
        return klass(img.get_data(),
                     img.get_affine(),
                     extra=img.extra)
    

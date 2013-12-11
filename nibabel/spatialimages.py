# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
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
   * .set_shape(shape)
   * .to_filename(fname) - writes data to filename(s) derived from
     ``fname``, where the derivation may differ between formats.
   * to_file_map() - save image to files with which the image is already
     associated.
   * .get_shape() (Deprecated)

properties:

   * shape

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

   img.to_file_map()

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

The image has an attribute ``file_map``.  This is a mapping, that has keys
corresponding to the file types that an image needs for storage.  For
example, the Analyze data format needs an ``image`` and a ``header``
file type for storage:

   >>> import nibabel as nib
   >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
   >>> img = nib.AnalyzeImage(data, np.eye(4))
   >>> sorted(img.file_map)
   ['header', 'image']

The values of ``file_map`` are not in fact files but objects with
attributes ``filename``, ``fileobj`` and ``pos``.

The reason for this interface, is that the contents of files has to
contain enough information so that an existing image instance can save
itself back to the files pointed to in ``file_map``.  When a file holder
holds active file-like objects, then these may be affected by the
initial file read; in this case, the contains file-like objects need to
carry the position at which a write (with ``to_files``) should place the
data.  The ``file_map`` contents should therefore be such, that this will
work:

   >>> # write an image to files
   >>> from io import BytesIO
   >>> file_map = nib.AnalyzeImage.make_file_map()
   >>> file_map['image'].fileobj = BytesIO()
   >>> file_map['header'].fileobj = BytesIO()
   >>> img = nib.AnalyzeImage(data, np.eye(4))
   >>> img.file_map = file_map
   >>> img.to_file_map()
   >>> # read it back again from the written files
   >>> img2 = nib.AnalyzeImage.from_file_map(file_map)
   >>> np.all(img2.get_data() == data)
   True
   >>> # write, read it again
   >>> img2.to_file_map()
   >>> img3 = nib.AnalyzeImage.from_file_map(file_map)
   >>> np.all(img3.get_data() == data)
   True

'''

try:
    basestring
except NameError:  # python 3
    basestring = str

import warnings

import numpy as np

from .filename_parser import types_filenames, TypesFilenamesError
from .fileholders import FileHolder
from .volumeutils import shape_zoom_affine


class HeaderDataError(Exception):
    ''' Class to indicate error in getting or setting header data '''
    pass


class HeaderTypeError(Exception):
    ''' Class to indicate error in parameters into header functions '''
    pass


class Header(object):
    ''' Template class to implement header protocol '''
    default_x_flip = True

    def __init__(self,
                 data_dtype=np.float32,
                 shape=(0,),
                 zooms=None):
        self.set_data_dtype(data_dtype)
        self._zooms = ()
        self.set_data_shape(shape)
        if not zooms is None:
            self.set_zooms(zooms)

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            return klass()
        # I can't do isinstance here because it is not necessarily true
        # that a subclass has exactly the same interface as it's parent
        # - for example Nifti1Images inherit from Analyze, but have
        # different field names
        if type(header) == klass:
            return header.copy()
        return klass(header.get_data_dtype(),
                     header.get_data_shape(),
                     header.get_zooms())

    @classmethod
    def from_fileobj(klass, fileobj):
        raise NotImplementedError

    def write_to(self, fileobj):
        raise NotImplementedError

    def __eq__(self, other):
        return ((self.get_data_dtype(),
                 self.get_data_shape(),
                 self.get_zooms()) ==
                (other.get_data_dtype(),
                 other.get_data_shape(),
                 other.get_zooms()))

    def __ne__(self, other):
        return not self == other

    def copy(self):
        ''' Copy object to independent representation

        The copy should not be affected by any changes to the original
        object.
        '''
        return self.__class__(self._dtype, self._shape, self._zooms)

    def get_data_dtype(self):
        return self._dtype

    def set_data_dtype(self, dtype):
        self._dtype = np.dtype(dtype)

    def get_data_shape(self):
        return self._shape

    def set_data_shape(self, shape):
        ndim = len(shape)
        if ndim == 0:
            self._shape = (0,)
            self._zooms = (1.0,)
            return
        self._shape = tuple([int(s) for s in shape])
        # set any unset zooms to 1.0
        nzs = min(len(self._zooms), ndim)
        self._zooms = self._zooms[:nzs] + (1.0,) * (ndim-nzs)

    def get_zooms(self):
        return self._zooms

    def set_zooms(self, zooms):
        zooms = tuple([float(z) for z in zooms])
        shape = self.get_data_shape()
        ndim = len(shape)
        if len(zooms) != ndim:
            raise HeaderDataError('Expecting %d zoom values for ndim %d'
                                  % (ndim, ndim))
        if len([z for z in zooms if z < 0]):
            raise HeaderDataError('zooms must be positive')
        self._zooms = zooms

    def get_base_affine(self):
        shape = self.get_data_shape()
        zooms = self.get_zooms()
        return shape_zoom_affine(shape, zooms,
                                 self.default_x_flip)

    get_best_affine = get_base_affine

    def data_to_fileobj(self, data, fileobj, rescale=True):
        ''' Write image data to file in Fortran memory layout

        Parameters
        ----------
        data : array-like
            data to write
        fileobj : file-like object
            file-like object implementing 'write'
        rescale : {True, False}, optional
            Whether to try and rescale data to match output dtype specified by
            header. For this minimal header, `rescale` has no effect
        '''
        data = np.asarray(data)
        dtype = self.get_data_dtype()
        fileobj.write(data.astype(dtype).tostring(order='F'))

    def data_from_fileobj(self, fileobj):
        ''' Read data in fortran order '''
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        data_size = int(np.prod(shape) * dtype.itemsize)
        data_bytes = fileobj.read(data_size)
        return np.ndarray(shape, dtype, data_bytes, order='F')


def supported_np_types(obj):
    """ Numpy data types that instance `obj` supports

    Parameters
    ----------
    obj : object
        Object implementing `get_data_dtype` and `set_data_dtype`.  The object
        should raise ``HeaderDataError`` for setting unsupported dtypes. The
        object will likely be a header or a :class:`SpatialImage`

    Returns
    -------
    np_types : set
        set of numpy types that `obj` supports
    """
    dt = obj.get_data_dtype()
    supported = []
    for name, np_types in np.sctypes.items():
        for np_type in np_types:
            try:
                obj.set_data_dtype(np_type)
            except HeaderDataError:
                continue
            # Did set work?
            if np.dtype(obj.get_data_dtype()) == np.dtype(np_type):
                supported.append(np_type)
    # Reset original header dtype
    obj.set_data_dtype(dt)
    return set(supported)


class ImageDataError(Exception):
    pass


class ImageFileError(Exception):
    pass


class SpatialImage(object):
    header_class = Header
    files_types = (('image', None),)
    _compressed_exts = ()

    ''' Template class for images '''
    def __init__(self, dataobj, affine, header=None,
                 extra=None, file_map=None):
        ''' Initialize image

        The image is a combination of (array, affine matrix, header), with
        optional metadata in `extra`, and filename / file-like objects contained
        in the `file_map` mapping.

        Parameters
        ----------
        dataobj : object
           Object containg image data.  It should be some object that retuns an
           array from ``np.asanyarray``.  It should have a ``shape`` attribute
           or property
        affine : None or (4,4) array-like
           homogenous affine giving relationship between voxel coordinates and
           world coordinates.  Affine can also be None.  In this case,
           ``obj.get_affine()`` also returns None, and the affine as written to
           disk will depend on the file format.
        header : None or mapping or header instance, optional
           metadata for this image format
        extra : None or mapping, optional
           metadata to associate with image that cannot be stored in the
           metadata of this image type
        file_map : mapping, optional
           mapping giving file information for this image format
        '''
        self._dataobj = dataobj
        if not affine is None:
            # Check that affine is array-like 4,4.  Maybe this is too strict at
            # this abstract level, but so far I think all image formats we know
            # do need 4,4.
            # Copy affine to isolate from environment.  Specify float type to
            # avoid surprising integer rounding when setting values into affine
            affine = np.array(affine, dtype=np.float64, copy=True)
            if not affine.shape == (4,4):
                raise ValueError('Affine should be shape 4,4')
        self._affine = affine
        if extra is None:
            extra = {}
        self.extra = extra
        self._header = self.header_class.from_header(header)
        # if header not specified, get data type from input array
        if header is None:
            if hasattr(dataobj, 'dtype'):
                self._header.set_data_dtype(dataobj.dtype)
        # make header correspond with image and affine
        self.update_header()
        if file_map is None:
            file_map = self.__class__.make_file_map()
        self.file_map = file_map
        self._load_cache = None
        self._data_cache = None

    @property
    def _data(self):
        warnings.warn('Please use ``dataobj`` instead of ``_data``; '
                      'We will remove this wrapper for ``_data`` soon',
                      FutureWarning,
                      stacklevel=2)
        return self._dataobj

    @property
    def dataobj(self):
        return self._dataobj

    @property
    def affine(self):
        return self._affine

    @property
    def header(self):
        return self._header

    def update_header(self):
        ''' Harmonize header with image data and affine

        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = SpatialImage(data, affine)
        >>> hdr = img.get_header()
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_header()
        >>> hdr.get_data_shape() == (2, 3, 4)
        True
        >>> hdr.get_zooms()
        (1.0, 2.0, 3.0)
        '''
        hdr = self._header
        shape = self._dataobj.shape
        # We need to update the header if the data shape has changed.  It's a
        # bit difficult to change the data shape using the standard API, but
        # maybe it happened
        if hdr.get_data_shape() != shape:
            hdr.set_data_shape(shape)
        # If the affine is not None, and it is different from the main affine in
        # the header, update the heaader
        if self._affine is None:
            return
        if np.allclose(self._affine, hdr.get_best_affine()):
            return
        self._affine2header()

    def _affine2header(self):
        """ Unconditionally set affine into the header """
        RZS = self._affine[:3, :3]
        vox = np.sqrt(np.sum(RZS * RZS, axis=0))
        hdr = self._header
        zooms = list(hdr.get_zooms())
        n_to_set = min(len(zooms), 3)
        zooms[:n_to_set] = vox[:n_to_set]
        hdr.set_zooms(zooms)

    def __str__(self):
        shape = self.shape
        affine = self.get_affine()
        return '\n'.join((
                str(self.__class__),
                'data shape %s' % (shape,),
                'affine: ',
                '%s' % affine,
                'metadata:',
                '%s' % self._header))

    def get_data(self):
        """ Return image data from image with any necessary scalng applied

        If the image data is a array proxy (data not yet read from disk) then
        read the data, and store in an internal cache.  Future calls to
        ``get_data`` will return the cached copy.

        Returns
        -------
        data : array
            array of image data
        """
        if self._data_cache is None:
            self._data_cache = np.asanyarray(self._dataobj)
        return self._data_cache

    def uncache(self):
        """ Delete any cached read of data from proxied data

        Remember there are two types of images:

        * *array images* where the data ``img.dataobj`` is an array
        * *proxy images* where the data ``img.dataobj`` is a proxy object

        If you call ``img.get_data()`` on a proxy image, the result of reading
        from the proxy gets cached inside the image object, and this cache is
        what gets returned from the next call to ``img.get_data()``.  If you
        modify the returned data, as in::

            data = img.get_data()
            data[:] = 42

        then the next call to ``img.get_data()`` returns the modified array,
        whether the image is an array image or a proxy image::

            assert np.all(img.get_data() == 42)

        When you uncache an array image, this has no effect on the return of
        ``img.get_data()``, but when you uncache a proxy image, the result of
        ``img.get_data()`` returns to its original value.
        """
        self._data_cache = None

    @property
    def shape(self):
        return self._dataobj.shape

    def get_shape(self):
        """ Return shape for image

        This function deprecated; please use the ``shape`` property instead
        """
        warnings.warn('Please use the shape property instead of get_shape',
                      DeprecationWarning,
                      stacklevel=2)
        return self.shape

    def get_data_dtype(self):
        return self._header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._header.set_data_dtype(dtype)

    def get_affine(self):
        """ Get affine from image

        Please use the `affine` property instead of `get_affine`; we will
        deprecate this method in future versions of nibabel.
        """
        return self.affine

    def get_header(self):
        """ Get header from image

        Please use the `header` property instead of `get_header`; we will
        deprecate this method in future versions of nibabel.
        """
        return self.header

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
        return self.file_map[characteristic_type].filename

    def set_filename(self, filename):
        ''' Sets the files in the object from a given filename

        The different image formats may check whether the filename has
        an extension characteristic of the format, and raise an error if
        not.

        Parameters
        ----------
        filename : str
           If the image format only has one file associated with it,
           this will be the only filename set into the image
           ``.file_map`` attribute. Otherwise, the image instance will
           try and guess the other filenames from this given filename.
        '''
        self.file_map = self.__class__.filespec_to_file_map(filename)

    @classmethod
    def from_filename(klass, filename):
        file_map = klass.filespec_to_file_map(filename)
        return klass.from_file_map(file_map)

    @classmethod
    def from_filespec(klass, filespec):
        warnings.warn('``from_filespec`` class method is deprecated\n'
                      'Please use the ``from_filename`` class method '
                      'instead',
                      DeprecationWarning, stacklevel=2)
        klass.from_filename(filespec)

    @classmethod
    def from_file_map(klass, file_map):
        raise NotImplementedError

    @classmethod
    def from_files(klass, file_map):
        warnings.warn('``from_files`` class method is deprecated\n'
                      'Please use the ``from_file_map`` class method '
                      'instead',
                      DeprecationWarning, stacklevel=2)
        return klass.from_file_map(file_map)

    @classmethod
    def filespec_to_file_map(klass, filespec):
        try:
            filenames = types_filenames(filespec,
                                        klass.files_types,
                                        trailing_suffixes=klass._compressed_exts)
        except TypesFilenamesError:
            raise ImageFileError('Filespec "%s" does not look right for '
                             'class %s ' % (filespec, klass))
        file_map = {}
        for key, fname in filenames.items():
            file_map[key] = FileHolder(filename=fname)
        return file_map

    @classmethod
    def filespec_to_files(klass, filespec):
        warnings.warn('``filespec_to_files`` class method is deprecated\n'
                      'Please use the ``filespec_to_file_map`` class method '
                      'instead',
                      DeprecationWarning, stacklevel=2)
        return klass.filespec_to_file_map(filespec)

    def to_filename(self, filename):
        ''' Write image to files implied by filename string

        Parameters
        ----------
        filename : str
           filename to which to save image.  We will parse `filename`
           with ``filespec_to_file_map`` to work out names for image,
           header etc.

        Returns
        -------
        None
        '''
        self.file_map = self.filespec_to_file_map(filename)
        self.to_file_map()

    def to_filespec(self, filename):
        warnings.warn('``to_filespec`` is deprecated, please '
                      'use ``to_filename`` instead',
                      DeprecationWarning, stacklevel=2)
        self.to_filename(filename)

    def to_file_map(self, file_map=None):
        raise NotImplementedError

    def to_files(self, file_map=None):
        warnings.warn('``to_files`` method is deprecated\n'
                      'Please use the ``to_file_map`` method '
                      'instead',
                      DeprecationWarning, stacklevel=2)
        self.to_file_map(file_map)

    @classmethod
    def make_file_map(klass, mapping=None):
        ''' Class method to make files holder for this image type

        Parameters
        ----------
        mapping : None or mapping, optional
           mapping with keys corresponding to image file types (such as
           'image', 'header' etc, depending on image class) and values
           that are filenames or file-like.  Default is None

        Returns
        -------
        file_map : dict
           dict with string keys given by first entry in tuples in
           sequence klass.files_types, and values of type FileHolder,
           where FileHolder objects have default values, other than
           those given by `mapping`
        '''
        if mapping is None:
            mapping = {}
        file_map = {}
        for key, ext in klass.files_types:
            file_map[key] = FileHolder()
            mapval = mapping.get(key, None)
            if isinstance(mapval, basestring):
                file_map[key].filename = mapval
            elif hasattr(mapval, 'tell'):
                file_map[key].fileobj = mapval
        return file_map

    @classmethod
    def load(klass, filename):
        return klass.from_filename(filename)

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
        ''' Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``get_data``, ``get_affine``, ``get_header`` and
           ``extra``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        '''
        return klass(img._dataobj,
                     img._affine,
                     klass.header_class.from_header(img._header),
                     extra=img.extra.copy())

# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' A simple spatial image class

The image class maintains the association between a 3D (or greater)
array, and an affine transform that maps voxel coordinates to some world space.
It also has a ``header`` - some standard set of meta-data that is specific to
the image format, and ``extra`` - a dictionary container for any other
metadata.

It has attributes:

   * extra

methods:

   * .get_data()
   * .get_affine() (deprecated, use affine property instead)
   * .get_header() (deprecated, use header property instead)
   * .to_filename(fname) - writes data to filename(s) derived from
     ``fname``, where the derivation may differ between formats.
   * to_file_map() - save image to files with which the image is already
     associated.
   * .get_shape() (deprecated)

properties:

   * shape
   * affine
   * header
   * dataobj

classmethods:

   * from_filename(fname) - make instance by loading from filename
   * from_file_map(fmap) - make instance from file map
   * instance_to_filename(img, fname) - save ``img`` instance to
     filename ``fname``.

You cannot slice an image, and trying to slice an image generates an
informative TypeError.

There are several ways of writing data.
=======================================

There is the usual way, which is the default::

    img.to_filename(fname)

and that is, to take the data encapsulated by the image and cast it to
the datatype the header expects, setting any available header scaling
into the header to help the data match.

You can load the data into an image from file with::

   img.from_filename(fname)

The image stores its associated files in its ``file_map`` attribute.  In order
to just save an image, for which you know there is an associated filename, or
other storage, you can do::

   img.to_file_map()

You can get the data out again with::

    img.get_data()

Less commonly, for some image types that support it, you might want to
fetch out the unscaled array via the object containing the data::

    unscaled_data = img.dataoobj.get_unscaled()

Analyze-type images (including nifti) support this, but others may not
(MINC, for example).

Sometimes you might to avoid any loss of precision by making the
data type the same as the input::

    hdr = img.header
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
    >>> import nibabel as nib
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

import numpy as np

from .filebasedimages import FileBasedHeader
from .dataobj_images import DataobjImage
from .filebasedimages import ImageFileError  # flake8: noqa; for back-compat
from .viewers import OrthoSlicer3D
from .volumeutils import shape_zoom_affine
from .deprecated import deprecate_with_version
from .orientations import apply_orientation


class HeaderDataError(Exception):
    ''' Class to indicate error in getting or setting header data '''


class HeaderTypeError(Exception):
    ''' Class to indicate error in parameters into header functions '''


class SpatialHeader(FileBasedHeader):
    ''' Template class to implement header protocol '''
    default_x_flip = True
    data_layout = 'F'

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
        # that a subclass has exactly the same interface as its parent
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
        self._zooms = self._zooms[:nzs] + (1.0,) * (ndim - nzs)

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
        ''' Write array data `data` as binary to `fileobj`

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
        fileobj.write(data.astype(dtype).tostring(order=self.data_layout))

    def data_from_fileobj(self, fileobj):
        ''' Read binary image data from `fileobj` '''
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        data_size = int(np.prod(shape) * dtype.itemsize)
        data_bytes = fileobj.read(data_size)
        return np.ndarray(shape, dtype, data_bytes, order=self.data_layout)


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


class Header(SpatialHeader):
    '''Alias for SpatialHeader; kept for backwards compatibility.'''

    @deprecate_with_version('Header class is deprecated.\n'
                            'Please use SpatialHeader instead.'
                            'instead.',
                            '2.1', '4.0')
    def __init__(self, *args, **kwargs):
        super(Header, self).__init__(*args, **kwargs)


class ImageDataError(Exception):
    pass


class SpatialImage(DataobjImage):
    ''' Template class for volumetric (3D/4D) images '''
    header_class = SpatialHeader

    def __init__(self, dataobj, affine, header=None,
                 extra=None, file_map=None):
        ''' Initialize image

        The image is a combination of (array-like, affine matrix, header), with
        optional metadata in `extra`, and filename / file-like objects
        contained in the `file_map` mapping.

        Parameters
        ----------
        dataobj : object
           Object containg image data.  It should be some object that retuns an
           array from ``np.asanyarray``.  It should have a ``shape`` attribute
           or property
        affine : None or (4,4) array-like
           homogenous affine giving relationship between voxel coordinates and
           world coordinates.  Affine can also be None.  In this case,
           ``obj.affine`` also returns None, and the affine as written to disk
           will depend on the file format.
        header : None or mapping or header instance, optional
           metadata for this image format
        extra : None or mapping, optional
           metadata to associate with image that cannot be stored in the
           metadata of this image type
        file_map : mapping, optional
           mapping giving file information for this image format
        '''
        super(SpatialImage, self).__init__(dataobj, header=header, extra=extra,
                                           file_map=file_map)
        if not affine is None:
            # Check that affine is array-like 4,4.  Maybe this is too strict at
            # this abstract level, but so far I think all image formats we know
            # do need 4,4.
            # Copy affine to isolate from environment.  Specify float type to
            # avoid surprising integer rounding when setting values into affine
            affine = np.array(affine, dtype=np.float64, copy=True)
            if not affine.shape == (4, 4):
                raise ValueError('Affine should be shape 4,4')
        self._affine = affine

        # if header not specified, get data type from input array
        if header is None:
            if hasattr(dataobj, 'dtype'):
                self._header.set_data_dtype(dataobj.dtype)
        # make header correspond with image and affine
        self.update_header()
        self._data_cache = None

    @property
    def affine(self):
        return self._affine

    def update_header(self):
        ''' Harmonize header with image data and affine

        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = SpatialImage(data, affine)
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_header()
        >>> img.header.get_data_shape() == (2, 3, 4)
        True
        >>> img.header.get_zooms()
        (1.0, 2.0, 3.0)
        '''
        hdr = self._header
        shape = self._dataobj.shape
        # We need to update the header if the data shape has changed.  It's a
        # bit difficult to change the data shape using the standard API, but
        # maybe it happened
        if hdr.get_data_shape() != shape:
            hdr.set_data_shape(shape)
        # If the affine is not None, and it is different from the main affine
        # in the header, update the header
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
        affine = self.affine
        return '\n'.join((str(self.__class__),
                          'data shape %s' % (shape,),
                          'affine: ',
                          '%s' % affine,
                          'metadata:',
                          '%s' % self._header))

    def get_data_dtype(self):
        return self._header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._header.set_data_dtype(dtype)

    @deprecate_with_version('get_affine method is deprecated.\n'
                            'Please use the ``img.affine`` property '
                            'instead.',
                            '2.1', '4.0')
    def get_affine(self):
        """ Get affine from image
        """
        return self.affine

    @classmethod
    def from_image(klass, img):
        ''' Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``dataobj``, ``affine``, ``header`` and ``extra``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        '''
        return klass(img.dataobj,
                     img.affine,
                     klass.header_class.from_header(img.header),
                     extra=img.extra.copy())

    def __getitem__(self, idx):
        ''' No slicing or dictionary interface for images
        '''
        raise TypeError("Cannot slice image objects; consider slicing image "
                        "array data with `img.dataobj[slice]` or "
                        "`img.get_data()[slice]`")

    def orthoview(self):
        """Plot the image using OrthoSlicer3D

        Returns
        -------
        viewer : instance of OrthoSlicer3D
            The viewer.

        Notes
        -----
        This requires matplotlib. If a non-interactive backend is used,
        consider using viewer.show() (equivalently plt.show()) to show
        the figure.
        """
        return OrthoSlicer3D(self.dataobj, self.affine,
                             title=self.get_filename())


    def transpose(self, ornt, new_aff):
        """Apply an orientation change and return a new image

        Parameters
        ----------
        ornt : (n,2) orientation array
           orientation transform. ``ornt[N,1]` is flip of axis N of the
           array implied by `shape`, where 1 means no flip and -1 means
           flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
           there's an array ``arr`` of shape `shape`, the flip would
           correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
           the transpose that needs to be done to the implied array, as in
           ``arr.transpose(ornt[:,0])``

        Notes
        -----
        Subclasses should override this if they have additional requirements
        when re-orienting an image.
        """

        t_arr = apply_orientation(self.get_data(), ornt)

        return self.__class__(t_arr, new_aff, self.header)

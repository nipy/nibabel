#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This module provides two classes for accessing NIfTI files.

* :class:`~nifti.niftiimage.NiftiImage` (traditional load-as-much-as-you-can
  approach)
* :class:`~nifti.niftiimage.MemMappedNiftiImage` (memory-mapped access to
  uncompressed NIfTI files)
"""

__docformat__ = 'restructuredtext'


# the swig wrapper if the NIfTI C library
import nifti.clib as ncl
from nifti.format import NiftiFormat
from nifti.utils import splitFilename, nifti2numpy_dtype_map
import numpy as N


class NiftiImage(NiftiFormat):
    """Wrapper class for convenient access to NIfTI data.

    The class can either load an image from file or convert a NumPy ndarray
    into a NIfTI file structure. Either way is automatically determined
    by the type of the 'source' argument. If `source` is a string, it is
    assumed to be a filename an ndarray is treated as such.

    One can optionally specify whether the image data should be loaded into
    memory when opening NIfTI data from files (`load`). When converting a NumPy
    array one can optionally specify a dictionary with NIfTI header data as
    available via the `header` attribute.
    """

    def __init__(self, source, header={}, load=False):
        """
        This method decides whether to load a nifti image from file or create
        one from ndarray data, depending on the datatype of `source`.

        :Parameters:
          source: str | ndarray
            If source is a string, it is assumed to be a filename and an
            attempt will be made to open the corresponding NIfTI file.
            In case of an ndarray the array data will be used for the to be
            created nifti image and a matching nifti header is generated.
            If an object of a different type is supplied as 'source' a
            ValueError exception will be thrown.
          header: dict
            Additional header data might be supplied. However,
            dimensionality and datatype are determined from the ndarray and
            not taken from a header dictionary.
          load: Boolean
            If set to True the image data will be loaded into memory. This
            is only useful if loading a NIfTI image from file.
        """
        # setup all nifti header related stuff
        NiftiFormat.__init__(self, source, header)

        # where the data will go to
        self._data = None

        # load data
        if type(source) == N.ndarray:
            # assign data from source array
            self._data = source[:]
        elif type(source) in (str, unicode):
            # only load image data from file if requested
            if load:
                self.load()
        else:
            raise ValueError, \
                  "Unsupported source type. Only NumPy arrays and filename " \
                  + "string are supported."


    def __del__(self):
        self.unload()

        # it is required to call base class destructors!
        NiftiFormat.__del__(self)



    def save(self, filename=None, filetype = 'NIFTI', update_minmax=True):
        """Save the image to a file.

        If the image was created using array data (i.e., not loaded from a file)
        a filename has to be specified.

        If not yet done already, the image data will be loaded into memory
        before saving the file.

        :Parameters:
          filename: str | None
            The name of the target file (typically including its extension).
            Usually setting the filename also determines the filetype
            (NIfTI/ANALYZE).  Please see
            :meth:`~nifti.niftiimage.NiftiImage.setFilename` for some more
            details. If None, an image loaded from a file will cause the
            original image to be overwritten.
          filetype: str
            Provide intented filetype. Please see the documentation of the
            `setFilename()` method for some more details.
          update_minmax: bool
            Whether the image header's min and max values should be updated
            according to the current image data.

        .. warning::

          There will be no exception if writing fails for any reason, as the
          underlying function nifti_write_hdr_img() from libniftiio does not
          provide any feedback. Suggestions for improvements are appreciated.
       """

        # If image data is not yet loaded, do it now.
        # It is important to do it already here, because nifti_image_load
        # depends on the correct filename set in the nifti_image struct
        # and this will be modified in this function!
        self.load()

        # set a default description if there is none
        if not self.description:
            self.description = 'Created with PyNIfTI'

        # update header information
        if update_minmax:
            self.updateCalMinMax()

        # saving for the first time?
        if not self.filename or filename:
            if not filename:
                raise ValueError, \
                      "When saving an image for the first time a filename " \
                      + "has to be specified."

            self.setFilename(filename, filetype)

        # if still no data is present data source has been an array
        # -> allocate memory in nifti struct and assign data to it
        if not self.raw_nimg.data:
            if not ncl.allocateImageMemory(self.raw_nimg):
                raise RuntimeError, "Could not allocate memory for image data."

        a = ncl.wrapImageDataWithArray(self.raw_nimg)
        a[:] = self._data[:]

        # now save it
        ncl.nifti_image_write_hdr_img(self.raw_nimg, 1, 'wb')
        # yoh comment: unfortunately return value of nifti_image_write_hdr_img
        # can't be used to track the successful completion of save
        # raise IOError, 'An error occured while attempting to save the image
        # file.'

        # take data pointer away from nifticlibs so we can let Python manage
        # the memory
        ncl.detachDataFromImage(self.raw_nimg)


    def __haveImageData(self):
        """Returns if the image data is accessible -- either loaded into
        memory or memory mapped.

        See: `load()`, `unload()`

        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        return (not self._data == None)


    def load(self):
        """Load the image data into memory, if it is not already accessible.

        It is save to call this method several times successively.
        """
        # do nothing if there already is data
        # which included memory mapped arrays not just data in memory
        if self.__haveImageData():
            return

        if ncl.nifti_image_load( self.raw_nimg ) < 0:
            raise RuntimeError, "Unable to load image data."

        self._data = ncl.wrapImageDataWithArray(self.raw_nimg)

        # take data pointer away from nifticlibs so we can let Python manage
        # the memory
        ncl.detachDataFromImage(self.raw_nimg)


    def unload(self):
        """Unload image data and free allocated memory.

        This methods does nothing in case of memory mapped files.
        """
        # simply assign none. The data array will free itself when the
        # reference count goes to zero.
        self._data = None


    def getDataArray(self):
        """Return the NIfTI image data wrapped into a NumPy array.

        The `data` property is an alternative way to access this function.
        """
        return self.asarray(False)


    def asarray(self, copy = True):
        """Convert the image data into a ndarray.

        :Parameters:
          copy: Boolean
            If set to False the array only wraps the image data, while True
            will return a copy of the data array.
        """
        # make sure data is accessible
        self.load()

        if copy:
            return self._data.copy()
        else:
            return self._data


    def getScaledData(self):
        """Returns a scaled copy of the data array.

        Scaling is done by multiplying with the slope and adding the intercept
        that is stored in the NIfTI header. In compliance with the NIfTI
        standard scaling is only performed in case of a non-zero slope value.
        The original data array is returned otherwise.

        :Returns:
          ndarray
        """
        data = self.asarray(copy = True)

        # NIfTI standard says: scaling only if non-zero slope
        if self.slope:
            data *= self.slope
            data += self.intercept

        return data


    def updateCalMinMax(self):
        """Update the image data maximum and minimum value in the nifti header.
        """
        self.raw_nimg.cal_max = float(self.data.max())
        self.raw_nimg.cal_min = float(self.data.min())


    def getBoundingBox(self):
        """Get the bounding box of the image.

        This functions returns a tuple of (min, max) tuples. It contains as
        many tuples as image dimensions. The order of dimensions is identical
        to that in the data array.

        The `bbox` property is an alternative way to access this function.
        """
        nz = self.data.squeeze().nonzero()

        bbox = []

        for dim in nz:
            bbox.append( ( dim.min(), dim.max() ) )

        return tuple(bbox)


    def setFilename(self, filename, filetype = 'NIFTI'):
        """Set the filename for the NIfTI image.

        Setting the filename also determines the filetype. If the filename
        ends with '.nii' the type will be set to NIfTI single file. A '.hdr'
        extension can be used for NIfTI file pairs. If the desired filetype
        is ANALYZE the extension should be '.img'. However, one can use the
        '.hdr' extension and force the filetype to ANALYZE by setting the
        filetype argument to ANALYZE. Setting filetype if the filename
        extension is '.nii' has no effect, the file will always be in NIFTI
        format.

        If the filename carries an additional '.gz' the resulting file(s) will
        be compressed.

        Uncompressed NIfTI single files are the default filetype that will be
        used if the filename has no valid extension. The '.nii' extension is
        appended automatically. The 'filetype' argument can be used to force a
        certain filetype when no extension can be used to determine it. 
        'filetype' can be one of the nifticlibs filtetypes or any of 'NIFTI',
        'NIFTI_GZ', 'NIFTI_PAIR', 'NIFTI_PAIR_GZ', 'ANALYZE', 'ANALYZE_GZ'.

        Setting the filename will cause the image data to be loaded into memory
        if not yet done already. This has to be done, because without the
        filename of the original image file there would be no access to the
        image data anymore. As a side-effect a simple operation like setting a
        filename may take a significant amount of time (e.g. for a large 4d
        dataset).

        By passing an empty string or none as filename one can reset the
        filename and detach the NiftiImage object from any file on disk.

        Examples:

          ================  ==================================
          Filename          Output of save()
          ----------------  ----------------------------------
          exmpl.nii         exmpl.nii (NIfTI)
          exmpl.hdr         exmpl.hdr, exmpl.img (NIfTI)
          exmpl.img         exmpl.hdr, exmpl.img (ANALYZE)
          exmpl             exmpl.nii (NIfTI)
          exmpl.hdr.gz      exmpl.hdr.gz, exmpl.img.gz (NIfTI)
          ----------------  ----------------------------------
          exmpl.gz          exmpl.gz.nii (uncompressed NIfTI)
          ================  ==================================

        Setting the filename is also possible by assigning to the 'filename'
        property.
        """
        # If image data is not yet loaded, do it now.
        # It is important to do it already here, because nifti_image_load
        # depends on the correct filename set in the nifti_image struct
        # and this will be modified in this function!
        self.load()

        # if no filename is given simply reset it to nothing
        if not filename:
            self.raw_nimg.fname = ''
            self.raw_nimg.iname = ''
            return

        # separate basename and extension
        base, ext = splitFilename(filename)

        # if no extension default to nifti single files
        if ext == '': 
            if filetype == 'NIFTI' \
               or filetype == ncl.NIFTI_FTYPE_NIFTI1_1:
                ext = 'nii'
            elif filetype == 'NIFTI_PAIR' \
                 or filetype == ncl.NIFTI_FTYPE_NIFTI1_2:
                ext = 'hdr'
            elif filetype == 'ANALYZE' \
                 or filetype == ncl.NIFTI_FTYPE_ANALYZE:
                ext = 'img'
            elif filetype == 'NIFTI_GZ':
                ext = 'nii.gz'
            elif filetype == 'NIFTI_PAIR_GZ':
                ext = 'hdr.gz'
            elif filetype == 'ANALYZE_GZ':
                ext = 'img.gz'
            else:
                raise RuntimeError, "Unhandled filetype."

        # Determine the filetype and set header and image filename
        # appropriately.

        # nifti single files are easy
        if ext == 'nii.gz' or ext == 'nii':
            self.raw_nimg.fname = base + '.' + ext
            self.raw_nimg.iname = base + '.' + ext
            self.raw_nimg.nifti_type = ncl.NIFTI_FTYPE_NIFTI1_1
        # uncompressed nifti file pairs
        elif ext in [ 'hdr', 'img' ]:
            self.raw_nimg.fname = base + '.hdr'
            self.raw_nimg.iname = base + '.img'
            if ext == 'hdr' and not filetype.startswith('ANALYZE'):
                self.raw_nimg.nifti_type = ncl.NIFTI_FTYPE_NIFTI1_2
            else:
                self.raw_nimg.nifti_type = ncl.NIFTI_FTYPE_ANALYZE
        # compressed file pairs
        elif ext in [ 'hdr.gz', 'img.gz' ]:
            self.raw_nimg.fname = base + '.hdr.gz'
            self.raw_nimg.iname = base + '.img.gz'
            if ext == 'hdr.gz' and not filetype.startswith('ANALYZE'):
                self.raw_nimg.nifti_type = ncl.NIFTI_FTYPE_NIFTI1_2
            else:
                self.raw_nimg.nifti_type = ncl.NIFTI_FTYPE_ANALYZE
        else:
            raise RuntimeError, "Unhandled filetype."


    def updateHeader(self, hdrdict):
        """Deprecated method only here for backward compatibility.

        Please refer to NiftiFormat.updateFromDict()
        """
        NiftiFormat.updateFromDict(self, hdrdict)


    # class properties
    # read only
    data =   property(fget=getDataArray)
    bbox =   property(fget=getBoundingBox)



class MemMappedNiftiImage(NiftiImage):
    """Memory mapped access to uncompressed NIfTI files.

    This access mode might be the prefered one whenever whenever only a small
    part of the image data has to be accessed or the memory is not sufficient
    to load the whole dataset.
    Please note, that memory-mapping is not required when exclusively header
    information shall be accessed. By default the `NiftiImage` class does not
    load any image data into memory.
    """

    def __init__(self, source):
        """Create a NiftiImage object.

        This method decides whether to load a nifti image from file or create
        one from ndarray data, depending on the datatype of `source`.

        :Parameters:
          source: str | ndarray
            If source is a string, it is assumed to be a filename and an
            attempt will be made to open the corresponding NIfTI file.
            In case of an ndarray the array data will be used for the to be
            created nifti image and a matching nifti header is generated.
            If an object of a different type is supplied as 'source' a
            ValueError exception will be thrown.
        """
        NiftiImage.__init__(self, source)

        # not working on compressed files
        if ncl.nifti_is_gzfile(self.raw_nimg.iname):
            raise RuntimeError, \
                  "Memory mapped access is only supported for " \
                  "uncompressed files."
 
        # determine byte-order
        if ncl.nifti_short_order() == 1:
            byteorder_flag = '<'
        else:
            byteorder_flag = '>'

        # create memmap array
        self._data = N.memmap(
            self.raw_nimg.iname,
            shape=self.extent[::-1],
            offset=self.raw_nimg.iname_offset,
            dtype=byteorder_flag + \
            nifti2numpy_dtype_map[self.raw_nimg.datatype],
            mode='r+')


    def __del__(self):
        """Do all necessary cleanups by calling __close().
        """
        self._data.flush()

        # it is required to call base class destructors!
        NiftiFormat.__del__(self)


    def save(self):
        """Save the image.

        This methods does nothing except for syncing the file on the disk.

        Please note that the NIfTI header might not be completely up-to-date.
        For example, the min and max values might be outdated, but this
        class does not automatically update them, because it would require to
        load and search through the whole array.
        """
        self._data.flush()


    def load(self):
        """Does nothing for memory mapped images.
        """
        return


    def unload(self):
        """Does nothing for memory mapped images.
        """
        return


    def setFilename(self, filename, filetype = 'NIFTI'):
        """Does not work for memory mapped images and therefore raises an
        exception.
        """
        raise RuntimeError, \
              "Filename modifications are not supported for memory mapped " \
              "images."

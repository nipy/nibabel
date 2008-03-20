#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Python class representation of a NIfTI image"""

__docformat__ = 'restructuredtext'


# the swig wrapper if the NIfTI C library
import nifti.nifticlib as nifticlib
from nifti.niftiformat import NiftiFormat
from nifti.utils import splitFilename
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

    Alternatively, uncompressed NIfTI images can also be memory-mapped. This
    is the preferred method whenever only a small part of the image data has
    to be accessed or the memory is not sufficient to load the whole dataset.
    Please note, that memory-mapping is not required when exclusively header
    information shall be accessed. By default no image data is loaded into
    memory.
    """



    def __init__(self, source, header={}, load=False):
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
            header: dict
                Additonal header data might be supplied. However,
                dimensionality and datatype are determined from the ndarray and
                not taken from a header dictionary.
            load: Boolean
                If set to True the image data will be loaded into memory. This
                is only useful if loading a NIfTI image from file.
        """
        # setup all nifti header related stuff
        NiftiFormat.__init__(self, source, header)

        # where the data will go to
        self.__data = None

        # load data
        if type(source) == N.ndarray:
            # assign data from source array
            self.__data = data[:]
        elif type(source) in (str, unicode):
            # only load image data from file if requested
            if load:
                self.load()
        else:
            raise ValueError, \
                  "Unsupported source type. Only NumPy arrays and filename " \
                  + "string are supported."


    def __del__(self):
        """Do all necessary cleanups by calling.
        Close the file and free all unnecessary memory.
        """
        self.unload()


    def save(self, filename=None, filetype = 'NIFTI'):
        """Save the image.

        If the image was created using array data (not loaded from a file) one
        has to specify a filename.

        Warning: There will be no exception if writing fails for any reason,
        as the underlying function nifti_write_hdr_img() from libniftiio does
        not provide any feedback. Suggestions for improvements are appreciated.

        If not yet done already, the image data will be loaded into memory
        before saving the file.

        :Parameters:
            filename: str | None
                Calling save() with `filename` equal None on a NiftiImage
                loaded from a file, it will overwrite the original file.

                Usually setting the filename also determines the filetype
                (NIfTI/ANALYZE). Please see the documentation of the
                `setFilename()` method for some more details.
            filetype: str
                Override filetype. Please see the documentation of the
                `setFilename()` method for some more details.
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
        self.updateCalMinMax()

        # saving for the first time?
        if not self.filename or filename:
            if not filename:
                raise ValueError, \
                      "When saving an image for the first time a filename " \
                      + "has to be specified."

            self.setFilename(filename, filetype)

        # now save it
        nifticlib.nifti_image_write_hdr_img(self.raw_nimg, 1, 'wb')
        # yoh comment: unfortunately return value of nifti_image_write_hdr_img
        # can't be used to track the successful completion of save
        # raise IOError, 'An error occured while attempting to save the image
        # file.'


    def __haveImageData(self):
        """Returns if the image data is accessible -- either loaded into
        memory or memory mapped.

        See: `load()`, `unload()`
        """
        return (not self.__data == None)


    def load(self):
        """Load the image data into memory, if it is not already accessible.

        It is save to call this method several times successively.
        """
        # do nothing if there already is data
        # which included memory mapped arrays not just data in memory
        if self.__haveImageData():
            return

        if nifticlib.nifti_image_load( self.raw_nimg ) < 0:
            raise RuntimeError, "Unable to load image data."

        self.__data = nifticlib.wrapImageDataWithArray(self.raw_nimg)


    def unload(self):
        """Unload image data and free allocated memory.

        This methods does nothing in case of memory mapped files.
        """
        # if no filename is se, the data will be lost and cannot be recovered
        if not self.filename:
            raise RuntimeError, \
                  "No filename is set, unloading the data would " \
                  "loose it completely without a chance of recovery."

        nifticlib.nifti_image_unload(self.raw_nimg)

        # reset array storage, as data pointer became invalid
        self.__data = None


    def getDataArray(self):
        """Return the NIfTI image data wrapped into a NumPy array.

        Attention: The array shares the data with the NiftiImage object. Any
        resize operation or datatype conversion will most likely result in a
        fatal error. If you need to perform such things, get a copy
        of the image data by using `asarray(copy=True)`.

        The `data` property is an alternative way to access this function.
        """
        return self.asarray(False)


    def asarray(self, copy = True):
        """Convert the image data into a ndarray.

        :Parameters:
            copy: Boolean
                If set to False the array only wraps the image data. Any
                modification done to the array is also done to the image data.
                In this case changing the shape, size or datatype of a wrapping
                array is not supported and will most likely result in a fatal
                error. If you want to do anything else to the data but reading
                or simple value assignment use a copy of the data by setting
                the copy flag to True. Later you can convert the modified data
                array into a NIfTi file again.
        """
        # make sure data is accessible
        self.load()

        if copy:
            return self.__data.copy()
        else:
            return self.__data


    def getScaledData(self):
        """Returns a scaled copy of the data array.

        Scaling is done by multiplying with the slope and adding the intercept
        that is stored in the NIfTI header.

        :Returns:
            ndarray
        """
        data = self.asarray(copy = True)

        return data * self.slope + self.intercept


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

          Filename          Output of save()
          ----------------------------------
          exmpl.nii         exmpl.nii (NIfTI)
          exmpl.hdr         exmpl.hdr, exmpl.img (NIfTI)
          exmpl.img         exmpl.hdr, exmpl.img (ANALYZE)
          exmpl             exmpl.nii (NIfTI)
          exmpl.hdr.gz      exmpl.hdr.gz, exmpl.img.gz (NIfTI)

        ! exmpl.gz          exmpl.gz.nii (uncompressed NIfTI)

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
               or filetype == nifticlib.NIFTI_FTYPE_NIFTI1_1:
                ext = 'nii'
            elif filetype == 'NIFTI_PAIR' \
                 or filetype == nifticlib.NIFTI_FTYPE_NIFTI1_2:
                ext = 'hdr'
            elif filetype == 'ANALYZE' \
                 or filetype == nifticlib.NIFTI_FTYPE_ANALYZE:
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
            self.raw_nimg.nifti_type = nifticlib.NIFTI_FTYPE_NIFTI1_1
        # uncompressed nifti file pairs
        elif ext in [ 'hdr', 'img' ]:
            self.raw_nimg.fname = base + '.hdr'
            self.raw_nimg.iname = base + '.img'
            if ext == 'hdr' and not filetype.startswith('ANALYZE'):
                self.raw_nimg.nifti_type = nifticlib.NIFTI_FTYPE_NIFTI1_2
            else:
                self.raw_nimg.nifti_type = nifticlib.NIFTI_FTYPE_ANALYZE
        # compressed file pairs
        elif ext in [ 'hdr.gz', 'img.gz' ]:
            self.raw_nimg.fname = base + '.hdr.gz'
            self.raw_nimg.iname = base + '.img.gz'
            if ext == 'hdr.gz' and not filetype.startswith('ANALYZE'):
                self.raw_nimg.nifti_type = nifticlib.NIFTI_FTYPE_NIFTI1_2
            else:
                self.raw_nimg.nifti_type = nifticlib.NIFTI_FTYPE_ANALYZE
        else:
            raise RuntimeError, "Unhandled filetype."

    # class properties
    # read only
    data =          property(fget=getDataArray)
    bbox =          property(fget=getBoundingBox)

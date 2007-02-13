import clibs
import os
import numpy

def numpydtype2niftidtype(array):
    
    # get the real datatype from numpy type dictionary
    dtype = numpy.typeDict[str(array.dtype)]

    if not DTnumpy2nifti.has_key(dtype):
        raise ValueError, "Unsupported datatype '%s'" % str(array.dtype)

    return DTnumpy2nifti[dtype]


nifti_unit_ids = [ 'm', 'mm', 'um' ]

DTnumpy2nifti = { numpy.uint8: clibs.NIFTI_TYPE_UINT8,
                  numpy.int8 : clibs.NIFTI_TYPE_INT8,
                  numpy.uint16: clibs.NIFTI_TYPE_UINT16,
                  numpy.int16 : clibs.NIFTI_TYPE_INT16,
                  numpy.uint32: clibs.NIFTI_TYPE_UINT32,
                  numpy.int32 : clibs.NIFTI_TYPE_INT32,
                  numpy.uint64: clibs.NIFTI_TYPE_UINT64,
                  numpy.int64 : clibs.NIFTI_TYPE_INT64,
                  numpy.float32: clibs.NIFTI_TYPE_FLOAT32,
                  numpy.float64: clibs.NIFTI_TYPE_FLOAT64,
                  numpy.complex128: clibs.NIFTI_TYPE_COMPLEX128
                }
    
class NiftiFile(object):
    """Wrapper class for convenient access to NIfTI data.
    
    The class can either load an image from file or convert a 3d/4d NumPy 
    array into a NIfTI file structure. Either way is automatically determined
    by the type of the 'source' argument (string == filename, array == Numpy).

    One can optionally specify whether the image data should be loaded into 
    memory when opening NIfTI data from files ('load'). When converting a NumPy
    array one can optionally specify the 'voxelsize' (x,y,z) and the repetition 
    time ('tr') as well as the measurment 'unit'. 

    Optional arguments of the respective other mode are ignored.
    """

    filetypes = [ 'ANALYZE', 'NIFTI', 'NIFTI_PAIR', 'ANALYZE_GZ', 'NIFTI_GZ',
                  'NIFTI_PAIR_GZ' ]

    # class properties
    filename = property(fget=lambda self: self.__nimg.fname)


    @staticmethod
    def splitFilename(filename):
        """ Split a NIfTI filename and returns a tuple of basename and 
        extension. If no valid NIfTI filename extension is found, the whole
        string is returned as basename and the extension string will be empty.
        """

        parts = filename.split('.')

        if parts[-1] == 'gz':
            if parts[-2] != 'nii' and parts[-2] != 'hdr':
                return filename, ''
            else:
                return '.'.join(parts[:-2]), '.'.join(parts[-2:])
        else:
            if parts[-1] != 'nii' and parts[-1] != 'hdr':
                return filename, ''
            else:
                return '.'.join(parts[:-1]), parts[-1]


    def __init__(self, source, load=False, voxelsize=(1,1,1), tr=1, unit='mm'):
        """
        """

        self.fslio = None
        self.__nimg = None

        if type( source ) == numpy.ndarray:
            self.__newFromArray( source, voxelsize, tr, unit )
        elif type ( source ) == str:
            self.__newFromFile( source, load )
        else:
            raise ValueError, "Unsupported source type. Only NumPy arrays and filename string are supported."

        
    def __del__(self):
        self.__close()


    def __close(self):
        """Close the file and free all unnecessary memory.
        """
        if self.fslio:
            clibs.FslClose(self.fslio)
            clibs.nifti_image_free(self.fslio.niftiptr)

        self.fslio = clibs.FslInit()
        self.__nimg = self.fslio.niftiptr


    def __newFromArray(self, data, voxelsize, tr, unit):
        
        if len(data.shape) > 4 or len(data.shape) < 3:
            raise ValueError, "Only 3d or 4d array are supported"

        if not unit in nifti_unit_ids:
            raise ValueError, "Unsupported unit '%s'. Supported units are '%s'" % (unit, ", ".join(nifti_unit_ids))

        # make clean table
        self.__close()

        dim = len(data.shape)

        if dim == 4:
            timesteps = data.shape[-4]
        else:
            timesteps = 1

        # init the data structure
        clibs.FslInitHeader( self.fslio,
                             numpydtype2niftidtype(data),
                             data.shape[-1], data.shape[-2], data.shape[-3],
                             timesteps,
                             voxelsize[0], voxelsize[1], voxelsize[2],
                             tr,
                             dim,
                             unit)

        self.__nimg = self.fslio.niftiptr

        # allocate memory for image data
        if not clibs.allocateImageMemory(self.__nimg):
            raise RuntimeError, "Could not allocate memory for image data."
    
        # assign data
        self.asarray()[:] = data[:]
        


    def __newFromFile(self, filename, load=False):
        """Open a NIfTI file.

        If there is already an open file it is closed first. If 'load' is True
        the image data is loaded into memory.
        """
        self.__close()
        self.fslio = clibs.FslOpen(filename, 'r+')

        if not self.fslio:
            raise RuntimeError, "Error while opening nifti header."
        
        self.__nimg = self.fslio.niftiptr

        if load:
            self.load()

    
    def save(self, filename=None, filetype='NIFTI'):
        """Save the image.

        If the image was created using array data (not loaded from a file) one
        has to specify a filename. 
        
        Calling save() without a specified filename on a NiftiFile loaded from 
        a file, will overwrite the original file.

        If a filename is specified, it will be made an attempt to guess the 
        corresponding filetype. A filename has to be the name of the 
        corresponding headerfile! In ambigous cases (.hdr might stand for 
        ANALYZE or uncompressed NIFTI file pairs) one can use the filetype 
        parameter to choose a certain type. If no filetype parameter is 
        specified NIfTI files will be written by default.

        If filename is only the basefilename (i.e. does not have a valid 
        extension of NIfTI/ANALYZE header files '.nii' is appended 
        automatically and a NIfTI single file will be written.

        If not yet done already, the image data will be loaded into memory 
        before saving the file.

        Warning: There will be no exception if writing fails for any reason, 
        as the underlying function nifti_write_hdr_img() from libniftiio does
        not provide any feedback. Suggestions for improvements are appreciated.
        """

        # If image data is not yet loaded, do it now.
        # It is important to do it already here, because nifti_image_load
        # depends on the correct filename set in the nifti_image struct
        # and this will be modified in this function!
        if not self.__haveImageData():
            self.load()

        # saving for the first time?
        if not self.filename or filename:
            if not filename:
                raise ValueError, "When saving an image for the first time a filename has to be specified."
            

            # check for valid filetype specifier
            if not filetype in self.filetypes:
                raise ValueError, \
                    "Unknown filetype '%s'. Known filetypes are: %s" % (filetype, ' '.join(nifti_filetype_ids))

            base, ext = NiftiFile.splitFilename(filename)

            # if no extension default to nifti single files
            if ext == '': ext = 'nii'

            # Determine the filetype and set header and image filename 
            # appropriately. If the filename extension is ambiguous the 
            # filetype setting is used to determine the intended format.

            # nifti single files are easy
            if ext == 'nii.gz' or ext == 'nii':
                self.__nimg.fname = base + '.' + ext
                self.__nimg.iname = base + '.' + ext
                self.__nimg.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_1
            # uncompressed file pairs
            elif ext == 'hdr':
                self.__nimg.fname = base + '.hdr'
                self.__nimg.iname = base + '.img'
                if filetype.startswith('NIFTI'):
                    self.__nimg.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_2
                else:
                    self.__nimg.nifti_type = clibs.NIFTI_FTYPE_ANALYZE
            # compressed file pairs
            elif ext == 'hdr.gz':
                self.__nimg.fname = base + '.hdr.gz'
                self.__nimg.iname = base + '.img.gz'
                if filetype.startswith('NIFTI'):
                    self.__nimg.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_2
                else:
                    self.__nimg.nifti_type = clibs.NIFTI_FTYPE_ANALYZE
            else:
                raise RuntimeError, "Unhandled filetype."
        
        # now save it
        clibs.nifti_image_write_hdr_img(self.__nimg, 1, 'wb')


    def __haveImageData(self):
        """Returns true if the image data was loaded into memory.
        or False if not.

        See: load(), unload()
        """
        self.__checkForNiftiImage()

        if self.__nimg.data:
            return True
        else:
            return False


    def load(self):
        """Load the image data into memory.

        It is save to call this method several times.
        """
        self.__checkForNiftiImage()

        if clibs.nifti_image_load( self.__nimg ) < 0:
            raise RuntimeError, "Unable to load image data." 
    

    def unload(self):
        """Unload image data and free allocated memory.
        """
        clibs.nifti_image_unload(self.__nimg)
    

    def asarray(self, copy = False):
        """Convert the image data into a multidimensional array.

        Attention: If copy == False (the default) the array only wraps 
        the image data. Any modification done to the array is also done 
        to the image data. If copy is true the array contains a copy
        of the image data.

        Changing the shape or size of the wrapping array is not supported
        and will most likely result in a fatal error.
        """
        if not self.__haveImageData():
            self.load()

        a = clibs.wrapImageDataWithArray(self.__nimg)

        if copy:
            return a.copy()
        else:
            return a


    def __checkForNiftiImage(self):
        """Check whether a NIfTI image is present.

        Returns True if there is a nifti image file structure or False otherwise.
        One can create a file structure by calling open().
        """
        if not self.__nimg:
            raise RuntimeError, "There is no NIfTI image file structure."


    def setDescription(self, description):
        if len(description) > 79:
            raise ValueError, "The NIfTI format only support descriptions shorter than 80 chars."

        self.__nimg.descrip = description


    def datatype(self):
        return clibs.nifti_datatype_string(self.__nimg.datatype)


    def voxDims(self):
        """Returns the dimensions of a single voxel as a tuple (x,y,z).
        """
        return ( self.__nimg.dx,
                 self.__nimg.dy,
                 self.__nimg.dz
               )


    def tr(self):
        return self.__nimg.dt


    def slope(self):
        return self.__nimg.scl_slope


    def intercept(self):
        return self.__nimg.scl_inter


    def q2xyz(self):
        return clibs.mat44ToArray(self.__nimg.qto_xyz)


    def q2ijk(self):
        return clibs.mat44ToArray(self.__nimg.qto_ijk)


    def s2xyz(self):
        return clibs.mat44ToArray(self.__nimg.sto_xyz)


    def s2ijk(self):
        return clibs.mat44ToArray(self.__nimg.sto_ijk)





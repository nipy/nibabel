#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Python class representation of a NIfTI image header"""

__docformat__ = 'restructuredtext'


# the swig wrapper if the NIfTI C library
import nifti.nifticlib as nifticlib
from nifti.utils import nhdr2dict, updateNiftiHeaderFromDict, \
                        Ndtype2niftidtype, nifti_xform_map
import numpy as N


class NiftiFormat(object):
    """NIfTI header representation.

    NIfTI header can be created by loading information from an existing NIfTI
    file or by creating a matching NIfTI header for a ndarray.

    In addition, a number of methods to manipulate the header information are
    provided. However, this class is not able to write a NIfTI header back to
    disk. Please refer to the NIfTIImage class for this functionality.
    """
    def __init__(self, source, header=None):
        """Create a NiftiImage object.

        This method decides whether to load a nifti image header from file or
        create one from ndarray data, depending on the datatype of `source`.

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
        """
        self.__nimg = None

        if header == None:
            header = {}

        if type(source) == N.ndarray:
            self.__newFromArray(source, header)
        elif type(source) in (str, unicode):
            self.__newFromFile(source)
        else:
            raise ValueError, \
                  "Unsupported source type. Only NumPy arrays and filename " \
                  + "string are supported."


    def __del__(self):
        """Do all necessary cleanups.
        """
        if self.__nimg:
            nifticlib.nifti_image_free(self.__nimg)


    def __newFromArray(self, data, hdr = {}):
        """Create a `nifti_image` struct from a ndarray.

        :Parameters:
          data: ndarray
            Source ndarray.
          hdr: dict
            Optional dictionary with NIfTI header data.
        """

        # check array
        if len(data.shape) > 7:
            raise ValueError, \
                  "NIfTI does not support data with more than 7 dimensions."

        # create template nifti header struct
        niptr = nifticlib.nifti_simple_init_nim()
        nhdr = nifticlib.nifti_convert_nim2nhdr(niptr)

        # intermediate cleanup
        nifticlib.nifti_image_free(niptr)

        # convert virgin nifti header to dict to merge properties
        # with supplied information and array properties
        hdic = nhdr2dict(nhdr)

        # copy data from supplied header dict
        for k, v in hdr.iteritems():
            hdic[k] = v

        # finally set header data that is determined by the data array
        # convert NumPy to nifti datatype
        hdic['datatype'] = Ndtype2niftidtype(data)

        # make sure there are no zeros in the dim vector
        # especially not in #4 as FSLView doesn't like that
        hdic['dim'] = [ 1 for i in hdic['dim'] ]

        # set number of dims
        hdic['dim'][0] = len(data.shape)

        # set size of each dim (and reverse the order to match nifti format
        # requirements)
        for i, s in enumerate(data.shape):
            hdic['dim'][len(data.shape)-i] = s

        # set magic field to mark as nifti file
        hdic['magic'] = 'n+1'

        # update nifti header with information from dict
        updateNiftiHeaderFromDict(nhdr, hdic)

        # convert nifti header to nifti image struct
        self.__nimg = nifticlib.nifti_convert_nhdr2nim(nhdr, 'pynifti_none')

        if not self.__nimg:
            raise RuntimeError, "Could not create nifti image structure."

        # kill filename for nifti images from arrays
        self.__nimg.fname = ''
        self.__nimg.iname = ''


    def __newFromFile(self, filename):
        """Open a NIfTI file.

        :Parameters:
          filename: str
            Filename of the to be opened image file.
        """
        # do not load image data!
        self.__nimg = nifticlib.nifti_image_read(filename, 0)

        if not self.__nimg:
            raise RuntimeError, "Error while opening nifti header."


    def getVoxDims(self):
        """Returns a 3-tuple a voxel dimensions/size in (x,y,z).

        The `voxdim` property is an alternative way to access this function.
        """
        return (self.__nimg.dx, self.__nimg.dy, self.__nimg.dz)


    def setVoxDims(self, value):
        """Set voxel dimensions/size.

        The qform matrix and its inverse will be recalculated automatically.

        :Parameter:
          value: 3-tuple of floats

        Besides reading it is also possible to set the voxel dimensions by
        assigning to the `voxdim` property.
        """
        if len(value) != 3:
            raise ValueError, 'Requires 3-tuple.'

        self.__nimg.dx = float(value[0])
        self.__nimg.dy = float(value[1])
        self.__nimg.dz = float(value[2])

        self.updateQFormFromQuaternion()


    def setPixDims(self, value):
        """Set the pixel dimensions.

        :Parameter:
          value: sequence
            Up to 7 values (max. number of dimensions supported by the
            NIfTI format) are allowed in the sequence.

            The supplied sequence can be shorter than seven elements. In
            this case only present values are assigned starting with the
            first dimension (spatial: x).

        Calling `setPixDims()` with a length-3 sequence equals calling
        `setVoxDims()`.
        """
        if len(value) > 7:
            raise ValueError, \
                  'The Nifti format does not support more than 7 dimensions.'

        pixdim = nifticlib.floatArray_frompointer( self.__nimg.pixdim )

        for i, val in enumerate(value):
            pixdim[i+1] = float(val)


    def getPixDims(self):
        """Returns the pixel dimensions on all 7 dimensions.

        The function is similar to `getVoxDims()`, but instead of the 3d
        spatial dimensions of a voxel it returns the dimensions of an image
        pixel on all 7 dimensions supported by the NIfTI dataformat.
        """
        return \
            tuple([ nifticlib.floatArray_frompointer(self.__nimg.pixdim)[i]
                    for i in range(1,8) ] )


    def getExtent(self):
        """Returns the shape of the dataimage.

        :Returns:
          Tuple with the size in voxel/timepoints.

          The order of dimensions is (x,y,z,t,u,v,w). If the image has less
          dimensions than 7 the return tuple will be shortened accordingly.

        Please note that the order of dimensions is different from the tuple
        returned by calling `NiftiImage.data.shape`!

        See also `getVolumeExtent()` and `getTimepoints()`.

        The `extent` property is an alternative way to access this function.
        """
        # wrap dim array in nifti image struct
        dims_array = nifticlib.intArray_frompointer(self.__nimg.dim)
        dims = [ dims_array[i] for i in range(8) ]

        return tuple( dims[1:dims[0]+1] )


    def getVolumeExtent(self):
        """Returns the size/shape of the volume(s) in the image as a tuple.

        :Returns:
          Either a 3-tuple or 2-tuple or 1-tuple depending on the available
          dimensions in the image.

          The order of dimensions in the tuple is (x [, y [, z ] ] ).

        The `volextent` property is an alternative way to access this function.
        """

        # it is save to do this even if self.extent is shorter than 4 items
        return self.extent[:3]


    def getTimepoints(self):
        """Returns the number of timepoints in the image.

        In case of a 3d (or less dimension) image this method returns 1.

        The `timepoints` property is an alternative way to access this
        function.
        """

        if len(self.extent) < 4:
            return 1
        else:
            return self.extent[3]


    def getRepetitionTime(self):
        """Returns the temporal distance between the volumes in a timeseries.

        The `rtime` property is an alternative way to access this function.
        """
        return self.__nimg.dt


    def setRepetitionTime(self, value):
        """Set the repetition time of a nifti image (dt).
        """
        self.__nimg.dt = float(value)


    def asDict(self):
        """Returns the header data of the `NiftiImage` in a dictionary.

        Note, that modifications done to this dictionary do not cause any
        modifications in the NIfTI image. Please use the `updateFromDict()`
        method to apply changes to the image.

        The `header` property is an alternative way to access this function. 
        But please note that the `header` property cannot be used like this::

            nimg.header['something'] = 'new value'

        Instead one has to get the header dictionary, modify and later reassign
        it::

            h = nimg.header
            h['something'] = 'new value'
            nimg.header = h
        """
        # Convert nifti_image struct into nifti1 header struct.
        # This get us all data that will actually make it into a
        # NIfTI file.
        nhdr = nifticlib.nifti_convert_nim2nhdr(self.__nimg)

        return nhdr2dict(nhdr)


    def updateFromDict(self, hdrdict):
        """Update NIfTI header information.

        Updated header data is read from the supplied dictionary. One cannot
        modify dimensionality and datatype of the image data. If such
        information is present in the header dictionary it is removed before
        the update. If resizing or datatype casting are required one has to 
        convert the image data into a separate array (`NiftiImage.assarray()`)
        and perform resize and data manipulations on this array. When finished,
        the array can be converted into a nifti file by calling the NiftiImage
        constructor with the modified array as 'source' and the nifti header
        of the original NiftiImage object as 'header'.
        """
        # rebuild nifti header from current image struct
        nhdr = nifticlib.nifti_convert_nim2nhdr(self.__nimg)

        # remove settings from the hdrdict that are determined by
        # the data set and must not be modified to preserve data integrity
        if hdrdict.has_key('datatype'):
            del hdrdict['datatype']
        if hdrdict.has_key('dim'):
            del hdrdict['dim']

        # update the nifti header
        updateNiftiHeaderFromDict(nhdr, hdrdict)

        # if no filename was set already (e.g. image from array) set a temp
        # name now, as otherwise nifti_convert_nhdr2nim will fail
        have_temp_filename = False
        if not self.filename:
            self.__nimg.fname = 'pynifti_updateheader_temp_name'
            self.__nimg.iname = 'pynifti_updateheader_temp_name'
            have_temp_filename = True

        # recreate nifti image struct
        new_nimg = nifticlib.nifti_convert_nhdr2nim(nhdr, self.filename)
        if not new_nimg:
            raise RuntimeError, \
                  "Could not recreate NIfTI image struct from updated header."

        # replace old image struct by new one
        # be careful with memory leak (still not checked whether successful)

        # assign the new image struct
        self.__nimg = new_nimg

        # reset filename if temp name was set
        if have_temp_filename:
            self.__nimg.fname = ''
            self.__nimg.iname = ''


    def setSlope(self, value):
        """Set the slope attribute in the NIfTI header.

        Besides reading it is also possible to set the slope by assigning
        to the `slope` property.
        """
        self.__nimg.scl_slope = float(value)


    def setIntercept(self, value):
        """Set the intercept attribute in the NIfTI header.

        Besides reading it is also possible to set the intercept by assigning
        to the `intercept` property.
        """
        self.__nimg.scl_inter = float(value)


    def setDescription(self, value):
        """Set the description element in the NIfTI header.

        :Parameter:
          value: str
            Description -- must not be longer than 79 characters.

        Besides reading it is also possible to set the description by assigning
        to the `description` property.
        """
        if len(value) > 79:
            raise ValueError, \
                  "The NIfTI format only supports descriptions shorter than " \
                  + "80 chars."

        self.__nimg.descrip = value


    def setXFormCode(self, xform, code):
        """Set the type of space described by the NIfTI transformations.

        The NIfTI format defines five coordinate system types which are used
        to describe the target space of a transformation (qform or sform).
        Please note, that the last four transformation types are only available
        in the NIfTI format and not when saving into ANALYZE.

          'unkown', `NIFTI_XFORM_UNKNOWN`, 0:
             Transformation is arbitrary. This is the ANALYZE compatibility
             mode. In this case no *sform* matrix will be written, even when
             stored in NIfTI and not in ANALYZE format. Additionally, only the
             pixdim parts of the *qform* matrix will be saved (upper-left 3x3).
          'scanner', `NIFTI_XFORM_SCANNER_ANAT`, 1:
             Scanner-based anatomical coordinates.
          'aligned', `NIFTI_XFORM_ALIGNED_ANAT`, 2:
             Coordinates are aligned to another file's coordinate system.
          'talairach', `NIFTI_XFORM_TALAIRACH`, 3:
             Coordinate system is shifted to have its origin (0,0,0) at the
             anterior commissure, as in the Talairach-Tournoux Atlas.
          'mni152', `NIFTI_XFORM_MNI_152`, 4:
             Coordinates are in MNI152 space.

        :Parameters:
          xform: str('qform' | 'sform')
            Which of the two NIfTI transformations to set.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The Transformation code can be specified either by a string, the
            `NIFTI_XFORM_CODE` defined in the nifti1.h header file (accessible
            via the `nifticlib` module, or the corresponding integer value
            (see above list for all possibilities).
        """
        if isinstance(code, str):
            code = nifti_xform_map[code]

        if xform == 'qform':
            self.raw_nimg.qform_code = code
        elif xform == 'sform':
            self.raw_nimg.sform_code = code
        else:
            raise ValueError, "Unkown transformation '%s'" % xform


    def getSForm(self):
        """Returns the sform matrix.

        Please note, that the returned SForm matrix is not bound to the
        NiftiImage object. Therefore it cannot be successfully modified
        in-place. Modifications to the SForm matrix can only be done by setting
        a new SForm matrix either by calling `setSForm()` or by assigning it to
        the `sform` attribute.

        The `sform` property is an alternative way to access this function.
        """
        return nifticlib.mat442array(self.__nimg.sto_xyz)


    def setSForm(self, m, code='mni152'):
        """Sets the sform matrix.

        The supplied value has to be a 4x4 matrix. The matrix elements will be
        converted to floats. By definition the last row of the sform matrix has
        to be (0,0,0,1). However, different values can be assigned, but will
        not be stored when the niftifile is saved.

        The inverse sform matrix will be automatically recalculated.

        Besides reading it is also possible to set the sform matrix by
        assigning to the `sform` property.

        :Parameters:
          m: 4x4 ndarray
            The sform matrix.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the sform matrix is describing.
            By default this coordinate system is assumed to be the MNI152 space.
            Please refer to the `setXFormCode()` method for a full list of
            possible codes and their meaning.
        """
        if m.shape != (4, 4):
            raise ValueError, "SForm matrix has to be of size 4x4."

        # make sure it is float
        m = m.astype('float')

        nifticlib.set_mat44( self.__nimg.sto_xyz,
                         m[0,0], m[0,1], m[0,2], m[0,3],
                         m[1,0], m[1,1], m[1,2], m[1,3],
                         m[2,0], m[2,1], m[2,2], m[2,3],
                         m[3,0], m[3,1], m[3,2], m[3,3] )

        # recalculate inverse
        self.__nimg.sto_ijk = \
            nifticlib.nifti_mat44_inverse( self.__nimg.sto_xyz )

        # set sform code, which decides how the sform matrix is interpreted
        self.setXFormCode('sform', code)


    def getInverseSForm(self):
        """Returns the inverse sform matrix.

        Please note, that the inverse SForm matrix cannot be modified in-place.
        One needs to set a new SForm matrix instead. The corresponding inverse
        matrix is then re-calculated automatically.

        The `sform_inv` property is an alternative way to access this function.
        """
        return nifticlib.mat442array(self.__nimg.sto_ijk)


    def getQForm(self):
        """Returns the qform matrix.

        Please note, that the returned QForm matrix is not bound to the
        NiftiImage object. Therefore it cannot be successfully modified
        in-place. Modifications to the QForm matrix can only be done by setting
        a new QForm matrix either by calling `setQForm()` or by assigning it to
        the `qform` property.
        """
        return nifticlib.mat442array(self.__nimg.qto_xyz)


    def getInverseQForm(self):
        """Returns the inverse qform matrix.

        The `qform_inv` property is an alternative way to access this function.

        Please note, that the inverse QForm matrix cannot be modified in-place.
        One needs to set a new QForm matrix instead. The corresponding inverse
        matrix is then re-calculated automatically.
        """
        return nifticlib.mat442array(self.__nimg.qto_ijk)


    def setQForm(self, m, code='scanner'):
        """Sets the qform matrix.

        The supplied value has to be a 4x4 matrix. The matrix will be converted
        to float.

        The inverse qform matrix and the quaternion representation will be
        automatically recalculated.

        Besides reading it is also possible to set the qform matrix by
        assigning to the `qform` property.

        :Parameters:
          m: 4x4 ndarray
            The qform matrix.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the qform matrix is describing.
            By default this coordinate system is assumed to be the scanner
            anatomical space. Please refer to the `setXFormCode()` method for
            a full list of possible codes and their meaning.
        """
        if m.shape != (4, 4):
            raise ValueError, "QForm matrix has to be of size 4x4."

        # make sure it is float
        m = m.astype('float')

        nifticlib.set_mat44( self.__nimg.qto_xyz,
                         m[0,0], m[0,1], m[0,2], m[0,3],
                         m[1,0], m[1,1], m[1,2], m[1,3],
                         m[2,0], m[2,1], m[2,2], m[2,3],
                         m[3,0], m[3,1], m[3,2], m[3,3] )

        # recalculate inverse
        self.__nimg.qto_ijk = \
            nifticlib.nifti_mat44_inverse( self.__nimg.qto_xyz )

        # update quaternions
        ( self.__nimg.quatern_b, self.__nimg.quatern_c, self.__nimg.quatern_d,
          self.__nimg.qoffset_x, self.__nimg.qoffset_y, self.__nimg.qoffset_z,
          self.__nimg.dx, self.__nimg.dy, self.__nimg.dz,
          self.__nimg.qfac ) = \
            nifticlib.nifti_mat44_to_quatern( self.__nimg.qto_xyz )

        # set qform code, which decides how the qform matrix is interpreted
        self.setXFormCode('qform', code)


    def updateQFormFromQuaternion(self):
        """Recalculates the qform matrix (and the inverse) from the quaternion
        representation.
        """
        # recalculate qform
        self.__nimg.qto_xyz = nifticlib.nifti_quatern_to_mat44 (
          self.__nimg.quatern_b, self.__nimg.quatern_c, self.__nimg.quatern_d,
          self.__nimg.qoffset_x, self.__nimg.qoffset_y, self.__nimg.qoffset_z,
          self.__nimg.dx, self.__nimg.dy, self.__nimg.dz,
          self.__nimg.qfac )


        # recalculate inverse
        self.__nimg.qto_ijk = \
            nifticlib.nifti_mat44_inverse( self.__nimg.qto_xyz )


    def setQuaternion(self, value, code='scanner'):
        """Set Quaternion from 3-tuple (qb, qc, qd).

        The qform matrix and it's inverse are re-computed automatically.

        Besides reading it is also possible to set the quaternion by assigning
        to the `quatern` property.

        :Parameters:
          value: length-3 sequence
            qb, qc and qd quaternions.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the corresponding qform matrix
            is describing. By default this coordinate system is assumed to be
            the scanner anatomical space. Please refer to the `setXFormCode()`
            method for a full list of possible codes and their meaning.
        """
        if len(value) != 3:
            raise ValueError, 'Requires 3-tuple.'

        self.__nimg.quatern_b = float(value[0])
        self.__nimg.quatern_c = float(value[1])
        self.__nimg.quatern_d = float(value[2])

        self.updateQFormFromQuaternion()
        self.setXFormCode('qform', code)


    def getQuaternion(self):
        """Returns a 3-tuple containing (qb, qc, qd).

        The `quatern` property is an alternative way to access this function.
        """
        return( ( self.__nimg.quatern_b, 
                  self.__nimg.quatern_c, 
                  self.__nimg.quatern_d ) )


    def setQOffset(self, value, code='scanner'):
        """Set QOffset from 3-tuple (qx, qy, qz).

        The qform matrix and its inverse are re-computed automatically.

        Besides reading it is also possible to set the qoffset by assigning
        to the `qoffset` property.

        :Parameters:
          value: length-3 sequence
            qx, qy and qz offsets.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the corresponding qform matrix
            is describing. By default this coordinate system is assumed to be
            the scanner anatomical space. Please refer to the `setXFormCode()`
            method for a full list of possible codes and their meaning.
        """
        if len(value) != 3:
            raise ValueError, 'Requires 3-tuple.'

        self.__nimg.qoffset_x = float(value[0])
        self.__nimg.qoffset_y = float(value[1])
        self.__nimg.qoffset_z = float(value[2])

        self.updateQFormFromQuaternion()
        self.setXFormCode('qform', code)


    def getQOffset(self):
        """Returns a 3-tuple containing (qx, qy, qz).

        The `qoffset` property is an alternative way to access this function.
        """
        return( ( self.__nimg.qoffset_x,
                  self.__nimg.qoffset_y,
                  self.__nimg.qoffset_z ) )


    def setQFac(self, value, code='scanner'):
        """Set qfac (scaling factor of qform matrix).

        The qform matrix and its inverse are re-computed automatically.

        Besides reading it is also possible to set the qfac by assigning
        to the `qfac` property.

        :Parameters:
          value: float
            Scaling factor.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the corresponding qform matrix
            is describing. By default this coordinate system is assumed to be
            the scanner anatomical space. Please refer to the `setXFormCode()`
            method for a full list of possible codes and their meaning.
        """
        self.__nimg.qfac = float(value)
        self.updateQFormFromQuaternion()
        self.setXFormCode('qform', code)


    def getQOrientation(self, as_string = False):
        """Returns to orientation of the i, j and k axis as stored in the
        qform matrix.

        By default NIfTI orientation codes are returned, but if `as_string` is
        set to true a string representation ala 'Left-to-right' is returned
        instead.
        """
        codes = nifticlib.nifti_mat44_to_orientation(self.__nimg.qto_xyz)
        if as_string:
            return [ nifticlib.nifti_orientation_string(i) for i in codes ]
        else:
            return codes


    def getSOrientation(self, as_string = False):
        """Returns to orientation of the i, j and k axis as stored in the
        sform matrix.

        By default NIfTI orientation codes are returned, but if `as_string` is
        set to true a string representation ala 'Left-to-right' is returned
        instead.
        """
        codes = nifticlib.nifti_mat44_to_orientation(self.__nimg.sto_xyz)
        if as_string:
            return [ nifticlib.nifti_orientation_string(i) for i in codes ]
        else:
            return codes


    def getFilename(self):
        """Returns the filename.

        To distinguish ANALYZE from 2-file NIfTI images the image filename is
        returned for ANALYZE images while the header filename is returned for
        NIfTI files.

        The `filename` property is an alternative way to access this function.
        """
        if self.__nimg.nifti_type == nifticlib.NIFTI_FTYPE_ANALYZE:
            return self.__nimg.iname
        else:
            return self.__nimg.fname

    # class properties
    # read only
    nvox =          property(fget=lambda self: self.__nimg.nvox)
    max =           property(fget=lambda self: self.__nimg.cal_max)
    min =           property(fget=lambda self: self.__nimg.cal_min)
    sform_inv =     property(fget=getInverseSForm)
    qform_inv =     property(fget=getInverseQForm)
    extent =        property(fget=getExtent)
    volextent =     property(fget=getVolumeExtent)
    timepoints =    property(fget=getTimepoints)
    raw_nimg =      property(fget=lambda self: self.__nimg)

    # read and write
    filename =      property(fget=getFilename)
    slope =         property(fget=lambda self: self.__nimg.scl_slope,
                             fset=setSlope)
    intercept =     property(fget=lambda self: self.__nimg.scl_inter,
                             fset=setIntercept)
    voxdim =        property(fget=getVoxDims, fset=setVoxDims)
    pixdim =        property(fget=getPixDims, fset=setPixDims)
    description =   property(fget=lambda self: self.__nimg.descrip,
                             fset=setDescription)
    header =        property(fget=asDict, fset=updateFromDict)
    sform =         property(fget=getSForm, fset=setSForm)
    qform =         property(fget=getQForm, fset=setQForm)
    quatern =       property(fget=getQuaternion, fset=setQuaternion)
    qoffset =       property(fget=getQOffset, fset=setQOffset)
    qfac =          property(fget=lambda self: self.__nimg.qfac, fset=setQFac)
    rtime =         property(fget=getRepetitionTime, fset=setRepetitionTime)


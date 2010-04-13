#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This modules provides a class representation of a NIfTI image header. The
interface provides pythonic access to NIfTI properties using Python datatypes.
"""

__docformat__ = 'restructuredtext'


import cPickle
from warnings import warn

import numpy as N

# the NIfTI pieces
import nifti.clib as ncl
from nifti.extensions import NiftiExtensions
from nifti.utils import nhdr2dict, updateNiftiHeaderFromDict, \
    Ndtype2niftidtype, nifti_xform_map, nifti_xform_inv_map, nifti_units_map, \
    _checkUnit, valid_xyz_unit_codes, valid_time_unit_codes, \
    nifti2numpy_dtype_map


class NiftiFormat(object):
    """NIfTI header representation.

    NIfTI header can be created by loading information from an existing NIfTI
    file or by creating a matching NIfTI header for a ndarray.

    In addition, a number of methods to manipulate the header information are
    provided. However, this class is not able to write a NIfTI header back to
    disk. Please refer to the NIfTIImage class for this functionality.

    .. note::

      Handling of NIfTI header extensions is provided by the
      :class:`~nifti.extensions.NiftiExtensions` class (see its documentation
      for more information). Access to an instance of this class is available
      through the `NiftiFormat.extensions` attribute.

    """
    #
    # object constructors, destructors and generic Python interface
    #
    def __init__(self, source, header=None, loadmeta=False):
        """
        The constructor decides whether to load a nifti image header from file
        or create one from ndarray data, depending on the datatype of `source`.

        :Parameters:
          source: str | ndarray
            If source is a string, it is assumed to be a filename and an
            attempt will be made to open the corresponding NIfTI file.
            Filenames might be provided as unicode strings. However, as the
            underlying library does not support unicode, they must be
            ascii-encodable, i.e. must not contain pure unicode characters.
            In case of an ndarray the array data will be used for the to be
            created nifti image and a matching nifti header is generated.
            If an object of a different type is supplied as 'source' a
            ValueError exception will be thrown.
          header: dict
            Additional header data might be supplied if image data is not loaded
            from a file. However, dimensionality and datatype are determined
            from the ndarray and not taken from a header dictionary.
        """
        self.__nimg = None
        # prepare empty meta dict
        self.meta = {}
        # placeholder for extensions interface
        self.extensions = None

        if type(source) == N.ndarray:
            self.__newFromArray(source, header)
        elif type(source) in (str, unicode):
            self.__newFromFile(source, loadmeta)
        else:
            raise ValueError, \
                  "Unsupported source type. Only NumPy arrays and filename " \
                  + "string are supported."


    def __newFromArray(self, data, hdr=None):
        """Create a `nifti_image` struct from a ndarray.

        :Parameters:
          data: ndarray
            Source ndarray.
          hdr: dict
            Optional dictionary with NIfTI header data.

        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        if hdr == None:
            hdr = {}

        # check array
        if len(data.shape) > 7:
            raise ValueError, \
                  "NIfTI does not support data with more than 7 dimensions."

        # create template nifti header struct
        niptr = ncl.nifti_simple_init_nim()
        nhdr = ncl.nifti_convert_nim2nhdr(niptr)

        # intermediate cleanup
        ncl.nifti_image_free(niptr)

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

        self._rebuildNimgFromHdrAndDict(nhdr, hdic)


    def __newFromFile(self, filename, loadmeta):
        """Open a NIfTI file.

        :Parameters:
          filename: str
            Filename of the to be opened image file.

        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        # make sure filename is not unicode
        try:
            filename = str(filename)
        except UnicodeEncodeError:
            raise UnicodeError, \
                  "The filename must not contain unicode characters, since " \
                  "the NIfTI library cannot handle them."

        # do not load image data!
        self.__nimg = ncl.nifti_image_read(filename, 0)

        if not self.__nimg:
            raise RuntimeError, "Error while opening NIfTI file '%s'." % \
                                    filename

        # simply create extension interface since nifticlib took care of
        # loading all extensions already
        self.extensions = NiftiExtensions(self.raw_nimg)

        # unpickle meta data if present
        if loadmeta and 'pypickle' in self.extensions:
            if self.extensions.count('pypickle') > 1:
                warn("Handling more than one 'pypickle' extension is not "
                     "supported. Will continue using the first detected "
                     "extension.")

            # unpickle meta data
            self.meta = cPickle.loads(self.extensions['pypickle'])
            # and remove the pickle extension to not confuse data integrity when
            # users would add something to it manually, i.e. via
            # self.extensions['pypickle']
            self._removePickleExtension()


    def __del__(self):
        # enforce del on extensions wrapper so Python GC doesn't try
        # to free it up later on causing writes to freed memory.
        del self.extensions

        if self.__nimg:
            ncl.nifti_image_free(self.__nimg)


    def __str__(self):
        lines = []

        lines.append('extent' + str(self.extent))

        lines.append('dtype(' \
                     + nifti2numpy_dtype_map[self.raw_nimg.datatype] \
                     + ')')

        s = 'voxels('
        s += 'x'.join(["%f" % d for d in self.voxdim])
        if self.xyz_unit:
            s += ' ' + self.getXYZUnit(as_string=True)
        lines.append(s + ')')

        if self.timepoints > 1:
            s = "timepoints(%i, dt=%f" % (self.timepoints, self.rtime)
            if self.time_unit:
                s += ' ' + self.getTimeUnit(as_string=True)
            s += ')'
            lines.append(s)

        if self.slope:
            lines.append("scaling(slope=%f, intercept=%f)" \
                    % (self.slope, self.intercept))

        if self.qform_code:
            lines.append("qform(%s)" % self.getQFormCode(as_string=True))
            lines.append("qform_orientation(%s)" \
                         % ', '.join(self.getQOrientation(as_string=True)))

        if self.sform_code:
            lines.append("sform(%s)" % self.getSFormCode(as_string=True))
            lines.append("sform_orientation(%s)" \
                         % ', '.join(self.getSOrientation(as_string=True)))

        if self.description:
            lines.append("descr('%s')" % self.description)

        if len(self.meta.keys()):
            lines.append("meta(%s)" % str(self.meta.keys()))

        return '<NIfTI:\n  ' + ';\n  '.join(lines) + ';\n>'


    #
    # converters
    #
    def asDict(self):
        """Returns the header data of the `NiftiImage` in a dictionary.

        :Returns:
          dict
            The dictionary contains all NIfTI header information. Additionally,
            it might also contain a special 'meta' item that contains the
            meta data currently assigned to this instance.

        .. note::

          Modifications done to the returned dictionary do not cause any
          modifications in the NIfTI image itself. Please use
          :meth:`~nifti.format.NiftiFormat.updateFromDict` to apply
          changes to the image.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.updateFromDict`,
          :attr:`~nifti.format.NiftiFormat.header`
        """
        # Convert nifti_image struct into nifti1 header struct.
        # This get us all data that will actually make it into a
        # NIfTI file.
        nhdr = ncl.nifti_convert_nim2nhdr(self.raw_nimg)

        # pass extensions as well
        ret = nhdr2dict(nhdr, extensions=self.extensions)

        if len(self.meta.keys()):
            ret['meta'] = self.meta

        return ret


    def updateFromDict(self, hdrdict):
        """Update NIfTI header information.

        Updated header data is read from the supplied dictionary. One cannot
        modify dimensionality and datatype of the image data. If such
        information is present in the header dictionary it is removed before
        the update. If resizing or datatype casting are required one has to
        convert the image data into a separate array and perform resize and
        data manipulations on this array. When finished, the array can be
        converted into a nifti file by calling the NiftiImage constructor with
        the modified array as 'source' and the nifti header of the original
        NiftiImage object as 'header'.

        .. note::

          If the provided dictionary contains a 'meta' item its content is
          used to overwrite any potentially existing meta data.
          dictionary.

          The same behavior will be used for 'extensions'. If extensions
          are defined in the provided dictionary all currently existing
          extensions will be overwritten.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.asDict`,
          :attr:`~nifti.format.NiftiFormat.header`
        """
        # rebuild nifti header from current image struct
        nhdr = ncl.nifti_convert_nim2nhdr(self.__nimg)

        # remove settings from the hdrdict that are determined by
        # the data set and must not be modified to preserve data integrity
        if hdrdict.has_key('datatype'):
            del hdrdict['datatype']
        if hdrdict.has_key('dim'):
            del hdrdict['dim']

        self._rebuildNimgFromHdrAndDict(nhdr, hdrdict)


    def vx2q(self, coord):
        """Transform a voxel's index into coordinates (qform-defined).

        :Parameter:
          coord: 3-tuple
            A voxel's index in the volume fiven as three positive integers
            (i, j, k).

        :Returns:
          vector

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setQForm`,
          :meth:`~nifti.format.NiftiFormat.getQForm`
          :attr:`~nifti.format.NiftiFormat.qform`
        """
        # add dummy one to row vector
        coord_ = N.r_[coord, [1.0]]
        # apply affine transformation
        result = N.dot(self.qform, coord_)
        # return 3D coordinates
        return result[0:-1]


    def vx2s(self, coord):
        """Transform a voxel's index into coordinates (sform-defined).

        :Parameter:
          coord: 3-tuple
            A voxel's index in the volume fiven as three positive integers
            (i, j, k).

        :Returns:
          vector

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setSForm`,
          :meth:`~nifti.format.NiftiFormat.getSForm`
          :attr:`~nifti.format.NiftiFormat.sform`
        """
        # add dummy one to row vector
        coord_ = N.r_[coord, [1.0]]
        # apply affine transformation
        result = N.dot(self.qform, coord_)
        # return 3D coordinates
        return result[0:-1]


    #
    # private helpers
    #
    def _updateNimgFromArray(self, val):
        """Update all relevant items in the nimg struct to match a given
        array's properties.

        We can only savely modify the respective nimg items since the data
        array is disconnected from the struct all the time.

        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        # convert dtype and store in struct
        self.raw_nimg.datatype = Ndtype2niftidtype(val)

        # wrap dims
        dim = ncl.intArray_frompointer(self.raw_nimg.dim)

        # make sure there are no zeros in the dim vector
        # especially not in #4 as FSLView doesn't like that
        target_dim = N.ones(7, dtype='int')

        # reverse the array shape
        target_dim[:len(val.shape)] = val.shape[::-1]

        # set number of dims
        dim[0] = len(val.shape)

        # assign remaining dim vector
        for i in range(7):
            dim[i+1] = target_dim[i]

        # expand dim vector
        self.raw_nimg.ndim = dim[0]
        self.raw_nimg.nx = dim[1]
        self.raw_nimg.ny = dim[2]
        self.raw_nimg.nz = dim[3]
        self.raw_nimg.nt = dim[4]
        self.raw_nimg.nu = dim[5]
        self.raw_nimg.nv = dim[6]
        self.raw_nimg.nw = dim[7]


    def _rebuildNimgFromHdrAndDict(self, nhdr, hdic):
        """
        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        # first updated the header struct from the provided dictionary
        # data
        updateNiftiHeaderFromDict(nhdr, hdic)

        # if no filename was set already (e.g. image from array) set a temp
        # name now, as otherwise nifti_convert_nhdr2nim will fail
        have_temp_filename = False
        if not self.__nimg:
            # we are creating from scratch
            new_nimg = ncl.nifti_convert_nhdr2nim(nhdr, 'pynifti_none')
            have_temp_filename = True
        elif not self.filename:
            # rebuild but no filename yet
            self.__nimg.fname = 'pynifti_updateheader_temp_name'
            self.__nimg.iname = 'pynifti_updateheader_temp_name'
            new_nimg = ncl.nifti_convert_nhdr2nim(nhdr, 'pynifti_none')
            have_temp_filename = True
        else:
            # recreate nifti image struct using current filename
            new_nimg = ncl.nifti_convert_nhdr2nim(nhdr, self.filename)

        if not new_nimg:
            raise RuntimeError, \
                  "Could not create NIfTI image struct from header."

        # rescue all extensions
        if self.extensions:
            # need to create a new extensions wrapper around the new
            # nifti image struct
            new_ext = NiftiExtensions(
                        new_nimg,
                        [ext for ext in self.extensions.iteritems()])
        else:
            # just create an empty wrapper
            new_ext = NiftiExtensions(new_nimg)

        # replace old image struct by new one
        # be careful with memory leak (still not checked whether successful)

        # assign the new image struct
        self.__nimg = new_nimg
        # and the extensions
        self.extensions = new_ext

        # reset filename if temp name was set
        if have_temp_filename:
            self.__nimg.fname = ''
            self.__nimg.iname = ''

        #
        # merge new extensions
        #
        if hdic.has_key('extensions'):
            # wipe current set of extensions
            self.extensions.clear()
            for e in hdic['extensions']:
                self.extensions.append(e)

        #
        # assign new meta data
        if hdic.has_key('meta'):
            self.meta = hdic['meta']


    def _removePickleExtension(self):
        """Remove the 'pypickle' extension from the raw NIfTI image struct.

        Its content is expanded into the `meta` attribute in a NiftiImage
        instance.

        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        if 'pypickle' in self.extensions:
            del self.extensions['pypickle']


    def updateQFormFromQuaternion(self):
        """Only here for backward compatibility."""
        from warnings import warn
        warn("The method has been renamed to " \
             "NiftiFormat.__updateQFormFromQuaternion and should not be used " \
             "in user code. This redirect will be removed with PyNIfTI 1.0.", \
             DeprecationWarning)

        self.__updateQFormFromQuaternion()


    def __updateQFormFromQuaternion(self):
        """Recalculates the qform matrix (and the inverse) from the quaternion
        representation.

        .. warning::
          This is an internal method. Neither its availability nor its API is
          guarenteed.
        """
        # recalculate qform
        self.__nimg.qto_xyz = ncl.nifti_quatern_to_mat44 (
          self.__nimg.quatern_b, self.__nimg.quatern_c, self.__nimg.quatern_d,
          self.__nimg.qoffset_x, self.__nimg.qoffset_y, self.__nimg.qoffset_z,
          self.__nimg.dx, self.__nimg.dy, self.__nimg.dz,
          self.__nimg.qfac )


        # recalculate inverse
        self.__nimg.qto_ijk = \
            ncl.nifti_mat44_inverse( self.__nimg.qto_xyz )


    #
    # getters and setters
    #
    def getVoxDims(self):
        """Returns a 3-tuple a voxel dimensions/size in (x,y,z).

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setVoxDims`,
          :attr:`~nifti.format.NiftiFormat.voxdim`
        """
        return (self.__nimg.dx, self.__nimg.dy, self.__nimg.dz)


    def setVoxDims(self, value):
        """Set voxel dimensions/size.

        The qform matrix and its inverse will be recalculated automatically.

        :Parameter:
          value: 3-tuple of floats
            Have to be given in (x,y,z) order.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getVoxDims`,
          :attr:`~nifti.format.NiftiFormat.voxdim`
        """
        if len(value) != 3:
            raise ValueError, 'Requires 3-tuple.'

        self.__nimg.dx = float(value[0])
        self.__nimg.dy = float(value[1])
        self.__nimg.dz = float(value[2])

        self.__updateQFormFromQuaternion()


    def setPixDims(self, value):
        """Set the pixel dimensions.

        :Parameter:
          value: sequence
            Up to 7 values (max. number of dimensions supported by the
            NIfTI format) are allowed in the sequence.

            The supplied sequence can be shorter than seven elements. In
            this case only present values are assigned starting with the
            first dimension (spatial: x).

        .. note::
          Calling `setPixDims()` with a length-3 sequence equals calling
          `setVoxDims()`.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setVoxDims`,
          :meth:`~nifti.format.NiftiFormat.getPixDims`,
          :attr:`~nifti.format.NiftiFormat.pixdim`
        """
        if len(value) > 7:
            raise ValueError, \
                  'The Nifti format does not support more than 7 dimensions.'

        pixdim = ncl.floatArray_frompointer( self.__nimg.pixdim )

        for i, val in enumerate(value):
            pixdim[i+1] = float(val)

        # The nifticlib uses dimension deltas (dx, dy, dz, dt...) to store
        # the pixdim values (in addition to the pixdim array).  When
        # saving the image to a file, the deltas are used, not the pixdims.
        # The nifti_update_dims_from_array sync's the deltas with the pixdims.
        # (It also syncs the dim array with it's duplicate scalar variables.)
        ncl.nifti_update_dims_from_array(self.__nimg)


    def getPixDims(self):
        """Returns the pixel dimensions on all 7 dimensions.

        The function is similar to `getVoxDims()`, but instead of the 3d
        spatial dimensions of a voxel it returns the dimensions of an image
        pixel on all 7 dimensions supported by the NIfTI dataformat.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getVoxDims`,
          :meth:`~nifti.format.NiftiFormat.setPixDims`,
          :attr:`~nifti.format.NiftiFormat.pixdim`
        """
        return \
            tuple([ ncl.floatArray_frompointer(self.__nimg.pixdim)[i]
                    for i in range(1,8) ] )


    def getExtent(self):
        """Returns the shape of the dataimage.

        :Returns:
          tuple: Tuple with the size in voxel/timepoints.
            The order of dimensions is (x,y,z,t,u,v,w). If the image has less
            dimensions than 7 the return tuple will be shortened accordingly.

            Please note that the order of dimensions is different from the tuple
            returned by calling `NiftiImage.data.shape`!

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getVolumeExtent`,
          :meth:`~nifti.format.NiftiFormat.getTimepoints`,
          :attr:`~nifti.format.NiftiFormat.extent`
        """
        # wrap dim array in nifti image struct
        dims_array = ncl.intArray_frompointer(self.__nimg.dim)
        dims = [ dims_array[i] for i in range(8) ]

        return tuple( dims[1:dims[0]+1] )


    def getVolumeExtent(self):
        """Returns the size/shape of the volume(s) in the image as a tuple.

        :Returns:
          tuple:
            Either a 3-tuple or 2-tuple or 1-tuple depending on the available
            dimensions in the image.

            The order of dimensions in the tuple is (x [, y [, z ] ] ).

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getExtent`,
          :attr:`~nifti.format.NiftiFormat.volextent`
        """

        # it is save to do this even if self.extent is shorter than 4 items
        return self.extent[:3]


    def getTimepoints(self):
        """Returns the number of timepoints in the image.

        In case of a 3d (or less dimension) image this method returns 1.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.timepoints`
        """

        if len(self.extent) < 4:
            return 1
        else:
            return self.extent[3]


    def getRepetitionTime(self):
        """Returns the temporal distance between the volumes in a timeseries.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setRepetitionTime`,
          :attr:`~nifti.format.NiftiFormat.rtime`
        """
        return self.__nimg.dt


    def setRepetitionTime(self, value):
        """Set the repetition time of a NIfTI image (dt).

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getRepetitionTime`,
          :attr:`~nifti.format.NiftiFormat.rtime`
        """
        self.__nimg.dt = float(value)


    def setSlope(self, value):
        """Set the slope attribute in the NIfTI header.

        Setting the slope to zero, will disable scaling.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.slope`,
          :attr:`~nifti.format.NiftiFormat.intercept`
        """
        self.__nimg.scl_slope = float(value)


    def setIntercept(self, value):
        """Set the intercept attribute in the NIfTI header.

        The intercept is only considered for scaling in case of a non-zero
        slope value.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.slope`,
          :attr:`~nifti.format.NiftiFormat.intercept`
        """
        self.__nimg.scl_inter = float(value)


    def setDescription(self, value):
        """Set the description element in the NIfTI header.

        :Parameter:
          value: str
            Description -- must not be longer than 79 characters.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.description`
        """
        if len(value) > 79:
            raise ValueError, \
                  "The NIfTI format only supports descriptions shorter than " \
                  "80 chars. (got length %i)" % len(value)

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
          xform: 'qform' | 'q' | 'sform' | 's'
            Which of the two NIfTI transformations to set.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The Transformation code can be specified either by a string, the
            `NIFTI_XFORM_CODE` defined in the nifti1.h header file (accessible
            via the `nifti.clib` module, or the corresponding integer value.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQFormCode`,
          :meth:`~nifti.format.NiftiFormat.setSFormCode`,
          :meth:`~nifti.format.NiftiFormat.getSFormCode`,
          :attr:`~nifti.format.NiftiFormat.qform_code`,
          :attr:`~nifti.format.NiftiFormat.sform_code`
        """
        if isinstance(code, str):
            if not code in nifti_xform_map.keys():
                raise ValueError, \
                      "Unknown xform code '%s'. Must be one of '%s'" \
                      % (code, str(nifti_xform_map.keys()))
            code = nifti_xform_map[code]
        else:
            if not code in nifti_xform_map.values():
                raise ValueError, \
                      "Unknown xform code '%s'. Must be one of '%s'" \
                      % (str(code), str(nifti_xform_map.values()))

        if xform == 'qform' or xform == 'q':
            self.raw_nimg.qform_code = code
        elif xform == 'sform' or xform == 's':
            self.raw_nimg.sform_code = code
        else:
            raise ValueError, "Unkown transformation '%s'" % xform


    def setQFormCode(self, code):
        """Set the qform code.

        .. note::
          This is a convenience frontend for
          :meth:`~nifti.format.NiftiFormat.setXFormCode`. Please see its
          documentation for more information.
        """
        self.setXFormCode('qform', code)


    def getQFormCode(self, as_string = False):
        """Return the qform code.

        By default NIfTI xform codes are returned, but if `as_string` is set to
        true a string representation ala 'talairach' is returned instead.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getQFormCode`,
          :attr:`~nifti.format.NiftiFormat.qform_code`
        """
        code = self.raw_nimg.qform_code
        if as_string:
            code = nifti_xform_inv_map[code]

        return code


    def getSFormCode(self, as_string = False):
        """Return the sform code.

        By default NIfTI xform codes are returned, but if `as_string` is set to
        true a string representation ala 'talairach' is returned instead.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getSFormCode`,
          :attr:`~nifti.format.NiftiFormat.sform_code`
        """
        code = self.raw_nimg.sform_code
        if as_string:
            code = nifti_xform_inv_map[code]

        return code


    def setSFormCode(self, code):
        """Set the sform code.

        .. note::
          This is a convenience frontend for
          :meth:`~nifti.format.NiftiFormat.setXFormCode`. Please see its
          documentation for more information.
        """
        self.setXFormCode('sform', code)


    def getSForm(self):
        """Returns the sform matrix.

        .. note::

          The returned sform matrix is not bound to the object. Therefore it
          cannot be successfully modified in-place. Modifications to the sform
          matrix can only be done by setting a new sform matrix

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setSForm`,
          :meth:`~nifti.format.NiftiFormat.setSFormCode`,
          :meth:`~nifti.format.NiftiFormat.getSFormCode`,
          :attr:`~nifti.format.NiftiFormat.sform`,
          :attr:`~nifti.format.NiftiFormat.sform_inv`,
          :attr:`~nifti.format.NiftiFormat.sform_code`
        """
        return ncl.mat442array(self.__nimg.sto_xyz)


    def setSForm(self, m, code='mni152'):
        """Sets the sform matrix.

        The supplied value has to be a 4x4 matrix. The matrix elements will be
        converted to floats. By definition the last row of the sform matrix has
        to be (0,0,0,1). However, different values can be assigned, but will
        not be stored when the NIfTI image is saved to a file.

        The inverse sform matrix will be automatically recalculated.

        :Parameters:
          m: 4x4 ndarray
            The sform matrix.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the sform matrix is describing.
            By default this coordinate system is assumed to be the MNI152
            space.  Please refer to the
            :meth:`~nifti.format.NiftiFormat.setXFormCode` method for a
            full list of possible codes and their meaning.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getSForm`,
          :meth:`~nifti.format.NiftiFormat.setSFormCode`,
          :meth:`~nifti.format.NiftiFormat.getSFormCode`,
          :attr:`~nifti.format.NiftiFormat.sform`,
          :attr:`~nifti.format.NiftiFormat.sform_code`
        """
        if m.shape != (4, 4):
            raise ValueError, "SForm matrix has to be of size 4x4."

        # make sure it is float
        m = m.astype('float')

        ncl.set_mat44( self.__nimg.sto_xyz,
                         m[0,0], m[0,1], m[0,2], m[0,3],
                         m[1,0], m[1,1], m[1,2], m[1,3],
                         m[2,0], m[2,1], m[2,2], m[2,3],
                         m[3,0], m[3,1], m[3,2], m[3,3] )

        # recalculate inverse
        self.__nimg.sto_ijk = \
            ncl.nifti_mat44_inverse( self.__nimg.sto_xyz )

        # set sform code, which decides how the sform matrix is interpreted
        self.setXFormCode('sform', code)


    def getInverseSForm(self):
        """Returns the inverse sform matrix.

        .. note::

          The inverse sform matrix cannot be modified in-place.  One needs to
          set a new sform matrix instead. The corresponding inverse matrix is
          then re-calculated automatically.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getSForm`,
          :attr:`~nifti.format.NiftiFormat.sform`,
          :attr:`~nifti.format.NiftiFormat.sform_inv`,
        """
        return ncl.mat442array(self.__nimg.sto_ijk)


    def getQForm(self):
        """Returns the qform matrix.

        .. note::

          The returned qform matrix is not bound to the object. Therefore it
          cannot be successfully modified in-place. Modifications to the qform
          matrix can only be done by setting a new qform matrix

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setQForm`,
          :meth:`~nifti.format.NiftiFormat.setQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQuaternion`,
          :meth:`~nifti.format.NiftiFormat.getQOffset`,
          :meth:`~nifti.format.NiftiFormat.setQuaternion`,
          :meth:`~nifti.format.NiftiFormat.setQOffset`,
          :meth:`~nifti.format.NiftiFormat.setQFac`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qform_inv`,
          :attr:`~nifti.format.NiftiFormat.qform_code`,
          :attr:`~nifti.format.NiftiFormat.quatern`,
          :attr:`~nifti.format.NiftiFormat.qoffset`,
          :attr:`~nifti.format.NiftiFormat.qfac`
        """
        return ncl.mat442array(self.__nimg.qto_xyz)


    def getInverseQForm(self):
        """Returns the inverse qform matrix.

        .. note::

          The inverse qform matrix cannot be modified in-place.  One needs to
          set a new qform matrix instead. The corresponding inverse matrix is
          then re-calculated automatically.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getQForm`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qform_inv`,
        """
        return ncl.mat442array(self.__nimg.qto_ijk)


    def setQForm(self, m, code='scanner'):
        """Sets the qform matrix.

        The supplied value has to be a 4x4 matrix. The matrix will be converted
        to float.

        The inverse qform matrix and the quaternion representation will be
        automatically recalculated.

        :Parameters:
          m: 4x4 ndarray
            The qform matrix.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the qform matrix is describing.
            By default this coordinate system is assumed to be the scanner
            space.  Please refer to the
            :meth:`~nifti.format.NiftiFormat.setXFormCode` method for a
            full list of possible codes and their meaning.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getQForm`,
          :meth:`~nifti.format.NiftiFormat.setQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQuaternion`,
          :meth:`~nifti.format.NiftiFormat.getQOffset`,
          :meth:`~nifti.format.NiftiFormat.setQuaternion`,
          :meth:`~nifti.format.NiftiFormat.setQOffset`,
          :meth:`~nifti.format.NiftiFormat.setQFac`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qform_inv`,
          :attr:`~nifti.format.NiftiFormat.qform_code`,
          :attr:`~nifti.format.NiftiFormat.quatern`,
          :attr:`~nifti.format.NiftiFormat.qoffset`,
          :attr:`~nifti.format.NiftiFormat.qfac`
        """
        if m.shape != (4, 4):
            raise ValueError, "QForm matrix has to be of size 4x4."

        # make sure it is float
        m = m.astype('float')

        ncl.set_mat44( self.__nimg.qto_xyz,
                         m[0,0], m[0,1], m[0,2], m[0,3],
                         m[1,0], m[1,1], m[1,2], m[1,3],
                         m[2,0], m[2,1], m[2,2], m[2,3],
                         m[3,0], m[3,1], m[3,2], m[3,3] )

        # recalculate inverse
        self.__nimg.qto_ijk = \
            ncl.nifti_mat44_inverse( self.__nimg.qto_xyz )

        # update quaternions
        ( self.__nimg.quatern_b, self.__nimg.quatern_c, self.__nimg.quatern_d,
          self.__nimg.qoffset_x, self.__nimg.qoffset_y, self.__nimg.qoffset_z,
          self.__nimg.dx, self.__nimg.dy, self.__nimg.dz,
          self.__nimg.qfac ) = \
            ncl.nifti_mat44_to_quatern( self.__nimg.qto_xyz )

        # set qform code, which decides how the qform matrix is interpreted
        self.setXFormCode('qform', code)


    def setQuaternion(self, value, code='scanner'):
        """Set Quaternion from 3-tuple (qb, qc, qd).

        The qform matrix and it's inverse are re-computed automatically.

        :Parameters:
          value: length-3 sequence
            qb, qc and qd quaternions.
          code: str | `NIFTI_XFORM_CODE` | int (0..4)
            The type of the coordinate system the corresponding qform matrix is
            describing.  By default this coordinate system is assumed to be the
            scanner space.  Please refer to the
            :meth:`~nifti.format.NiftiFormat.setXFormCode` method for a
            full list of possible codes and their meaning.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getQForm`,
          :meth:`~nifti.format.NiftiFormat.setQForm`,
          :meth:`~nifti.format.NiftiFormat.setQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQFormCode`,
          :meth:`~nifti.format.NiftiFormat.getQuaternion`,
          :meth:`~nifti.format.NiftiFormat.getQOffset`,
          :meth:`~nifti.format.NiftiFormat.setQOffset`,
          :meth:`~nifti.format.NiftiFormat.setQFac`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qform_inv`,
          :attr:`~nifti.format.NiftiFormat.qform_code`,
          :attr:`~nifti.format.NiftiFormat.quatern`,
          :attr:`~nifti.format.NiftiFormat.qoffset`,
          :attr:`~nifti.format.NiftiFormat.qfac`
        """
        if len(value) != 3:
            raise ValueError, 'Requires 3-tuple.'

        self.__nimg.quatern_b = float(value[0])
        self.__nimg.quatern_c = float(value[1])
        self.__nimg.quatern_d = float(value[2])

        self.__updateQFormFromQuaternion()
        self.setXFormCode('qform', code)


    def getQuaternion(self):
        """Returns a 3-tuple containing (qb, qc, qd).

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setQuaternion`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.quatern`
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
            The type of the coordinate system the corresponding qform matrix is
            describing.  By default this coordinate system is assumed to be the
            scanner space.  Please refer to the
            :meth:`~nifti.format.NiftiFormat.setXFormCode` method for a
            full list of possible codes and their meaning.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getQOffset`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qoffset`
        """
        if len(value) != 3:
            raise ValueError, 'Requires 3-tuple.'

        self.__nimg.qoffset_x = float(value[0])
        self.__nimg.qoffset_y = float(value[1])
        self.__nimg.qoffset_z = float(value[2])

        self.__updateQFormFromQuaternion()
        self.setXFormCode('qform', code)


    def getQOffset(self):
        """Returns a 3-tuple containing (qx, qy, qz).

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setQOffset`,
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qoffset`
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
            The type of the coordinate system the corresponding qform matrix is
            describing.  By default this coordinate system is assumed to be the
            scanner space.  Please refer to the
            :meth:`~nifti.format.NiftiFormat.setXFormCode` method for a
            full list of possible codes and their meaning.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.qform`,
          :attr:`~nifti.format.NiftiFormat.qfac`
        """
        self.__nimg.qfac = float(value)
        self.__updateQFormFromQuaternion()
        self.setXFormCode('qform', code)


    def getQOrientation(self, as_string = False):
        """Returns to orientation of the i, j and k axis as stored in the
        qform matrix.

        By default NIfTI orientation codes are returned, but if `as_string` is
        set to true a string representation ala 'Left-to-right' is returned
        instead.

        :Returns:
          list
            orientations fo the x, y and z axis respectively.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.qform`
        """
        codes = ncl.nifti_mat44_to_orientation(self.__nimg.qto_xyz)
        if as_string:
            return [ ncl.nifti_orientation_string(i) for i in codes ]
        else:
            return codes


    def getSOrientation(self, as_string = False):
        """Returns to orientation of the i, j and k axis as stored in the
        sform matrix.

        By default NIfTI orientation codes are returned, but if `as_string` is
        set to true a string representation ala 'Left-to-right' is returned
        instead.

        :Returns:
          list
            orientations fo the x, y and z axis respectively.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.sform`
        """
        codes = ncl.nifti_mat44_to_orientation(self.__nimg.sto_xyz)
        if as_string:
            return [ ncl.nifti_orientation_string(i) for i in codes ]
        else:
            return codes


    def getXYZUnit(self, as_string = False):
        """Return 3D-space unit.

        By default NIfTI unit codes are returned, but if `as_string` is set to
        true a string representation ala 'mm' is returned instead.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setXYZUnit`,
          :attr:`~nifti.format.NiftiFormat.xyz_unit`
        """
        code = self.__nimg.xyz_units
        if as_string:
            code = ncl.nifti_units_string(code)

        return code


    def setXYZUnit(self, value):
        """Set the unit of the spatial axes.

        :Parameter:
          value: int | str
            The unit can either be given as a NIfTI unit code or as any of the
            plain text abbrevations returned by
            :meth:'~nifti.format.NiftiFormat.getXYZUnit`

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getXYZUnit`,
          :attr:`~nifti.format.NiftiFormat.xyz_unit`
        """
        # check for valid codes according to NIfTI1 standard
        code = _checkUnit(value, valid_xyz_unit_codes)
        self.raw_nimg.xyz_units = code


    def getTimeUnit(self, as_string = False):
        """Return unit of temporal (4th) axis.

        By default NIfTI unit codes are returned, but if `as_string` is set to
        true a string representation ala 's' is returned instead.

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.setTimeUnit`,
          :attr:`~nifti.format.NiftiFormat.time_unit`
        """
        code = self.__nimg.time_units
        if as_string:
            code = ncl.nifti_units_string(code)

        return code


    def setTimeUnit(self, value):
        """Set the unit of the temporal axis (4th).

        :Parameter:
          value: int | str
            The unit can either be given as a NIfTI unit code or as any of the
            plain text abbrevations returned by
            :meth:'~nifti.format.NiftiFormat.getTimeUnit`

        .. seealso::
          :meth:`~nifti.format.NiftiFormat.getTimeUnit`,
          :attr:`~nifti.format.NiftiFormat.time_unit`
        """
        # check for valid codes according to NIfTI1 standard
        code = _checkUnit(value, valid_time_unit_codes)
        self.raw_nimg.time_units = code


    def getFilename(self):
        """Returns the filename.

        To distinguish ANALYZE from 2-file NIfTI images the image filename is
        returned for ANALYZE images while the header filename is returned for
        NIfTI files.

        .. seealso::
          :attr:`~nifti.format.NiftiFormat.filename`
        """
        if self.__nimg.nifti_type == ncl.NIFTI_FTYPE_ANALYZE:
            return self.__nimg.iname
        else:
            return self.__nimg.fname


    #
    # class properties
    #

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
    filename =      property(fget=getFilename)

    # read and write
    slope =         property(fget=lambda self: self.__nimg.scl_slope,
                             fset=setSlope)
    intercept =     property(fget=lambda self: self.__nimg.scl_inter,
                             fset=setIntercept)
    voxdim =        property(fget=getVoxDims, fset=setVoxDims)
    pixdim =        property(fget=getPixDims, fset=setPixDims)
    description =   property(fget=lambda self: self.__nimg.descrip,
                             fset=setDescription)
    header = property(
        fget=asDict, fset=updateFromDict,
        doc="""Access to a dictionary version of the NIfTI header data.

            .. note::

              This property cannot be used like this::

                nimg.header['something'] = 'new value'

              Instead one has to get the header dictionary, modify
              and later reassign it::

                h = nimg.header
                h['something'] = 'new value'
                nimg.header = h

            .. seealso::
              :meth:`~nifti.format.NiftiFormat.asDict`,
              :meth:`~nifti.format.NiftiFormat.updateFromDict`
            """)
    sform =         property(fget=getSForm, fset=setSForm)
    sform_code =    property(fget=getSFormCode, fset=setSFormCode)
    qform =         property(fget=getQForm, fset=setQForm)
    qform_code =    property(fget=getQFormCode, fset=setQFormCode)
    quatern =       property(fget=getQuaternion, fset=setQuaternion)
    qoffset =       property(fget=getQOffset, fset=setQOffset)
    qfac =          property(fget=lambda self: self.__nimg.qfac, fset=setQFac)
    rtime =         property(fget=getRepetitionTime, fset=setRepetitionTime)
    xyz_unit =      property(fget=getXYZUnit, fset=setXYZUnit)
    time_unit =     property(fget=getTimeUnit, fset=setTimeUnit)

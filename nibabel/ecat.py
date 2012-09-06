# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import warnings

import numpy as np

from .volumeutils import (native_code, swapped_code, make_dt_codes,
                           array_from_file)
from .spatialimages import SpatialImage, ImageDataError
from .arraywriters import make_array_writer


MAINHDRSZ = 502
main_header_dtd = [
    ('magic_number', '14S'),
    ('original_filename', '32S'),
    ('sw_version', np.uint16),
    ('system_type', np.uint16),
    ('file_type', np.uint16),
    ('serial_number', '10S'),
    ('scan_start_time',np.uint32),
    ('isotope_name', '8S'),
    ('isotope_halflife', np.float32),
    ('radiopharmaceutical','32S'),
    ('gantry_tilt', np.float32),
    ('gantry_rotation',np.float32),
    ('bed_elevation',np.float32),
    ('intrinsic_tilt', np.float32),
    ('wobble_speed',np.uint16),
    ('transm_source_type',np.uint16),
    ('distance_scanned',np.float32),
    ('transaxial_fov',np.float32),
    ('angular_compression', np.uint16),
    ('coin_samp_mode',np.uint16),
    ('axial_samp_mode',np.uint16),
    ('ecat_calibration_factor',np.float32),
    ('calibration_unitS', np.uint16),
    ('calibration_units_type',np.uint16),
    ('compression_code',np.uint16),
    ('study_type','12S'),
    ('patient_id','16S'),
    ('patient_name','32S'),
    ('patient_sex','1S'),
    ('patient_dexterity','1S'),
    ('patient_age',np.float32),
    ('patient_height',np.float32),
    ('patient_weight',np.float32),
    ('patient_birth_date',np.uint32),
    ('physician_name','32S'),
    ('operator_name','32S'),
    ('study_description','32S'),
    ('acquisition_type',np.uint16),
    ('patient_orientation',np.uint16),
    ('facility_name', '20S'),
    ('num_planes',np.uint16),
    ('num_frames',np.uint16),
    ('num_gates',np.uint16),
    ('num_bed_pos',np.uint16),
    ('init_bed_position',np.float32),
    ('bed_position','15f'),
    ('plane_separation',np.float32),
    ('lwr_sctr_thres',np.uint16),
    ('lwr_true_thres',np.uint16),
    ('upr_true_thres',np.uint16),
    ('user_process_code','10S'),
    ('acquisition_mode',np.uint16),
    ('bin_size',np.float32),
    ('branching_fraction',np.float32),
    ('dose_start_time',np.uint32),
    ('dosage',np.float32),
    ('well_counter_corr_factor', np.float32),
    ('data_units', '32S'),
    ('septa_state',np.uint16),
    ('fill', '12S')
    ]
hdr_dtype = np.dtype(main_header_dtd)


subheader_dtd = [
    ('data_type', np.uint16),
    ('num_dimensions', np.uint16),
    ('x_dimension', np.uint16),
    ('y_dimension', np.uint16),
    ('z_dimension', np.uint16),
    ('x_offset', np.float32),
    ('y_offset', np.float32),
    ('z_offset', np.float32),
    ('recon_zoom', np.float32),
    ('scale_factor', np.float32),
    ('image_min', np.int16),
    ('image_max', np.int16),
    ('x_pixel_size', np.float32),
    ('y_pixel_size', np.float32),
    ('z_pixel_size', np.float32),
    ('frame_duration', np.uint32),
    ('frame_start_time', np.uint32),
    ('filter_code', np.uint16),
    ('x_resolution', np.float32),
    ('y_resolution', np.float32),
    ('z_resolution', np.float32),
    ('num_r_elements', np.float32),
    ('num_angles', np.float32),
    ('z_rotation_angle', np.float32),
    ('decay_corr_fctr', np.float32),
    ('corrections_applied', np.uint32),
    ('gate_duration', np.uint32),
    ('r_wave_offset', np.uint32),
    ('num_accepted_beats', np.uint32),
    ('filter_cutoff_frequency', np.float32),
    ('filter_resolution', np.float32),
    ('filter_ramp_slope', np.float32),
    ('filter_order', np.uint16),
    ('filter_scatter_fraction', np.float32),
    ('filter_scatter_slope', np.float32),
    ('annotation', '40S'),
    ('mt_1_1', np.float32),
    ('mt_1_2', np.float32),
    ('mt_1_3', np.float32),
    ('mt_2_1', np.float32),
    ('mt_2_2', np.float32),
    ('mt_2_3', np.float32),
    ('mt_3_1', np.float32),
    ('mt_3_2', np.float32),
    ('mt_3_3', np.float32),
    ('rfilter_cutoff', np.float32),
    ('rfilter_resolution', np.float32),
    ('rfilter_code', np.uint16),
    ('rfilter_order', np.uint16),
    ('zfilter_cutoff', np.float32),
    ('zfilter_resolution',np.float32),
    ('zfilter_code', np.uint16),
    ('zfilter_order', np.uint16),
    ('mt_4_1', np.float32),
    ('mt_4_2', np.float32),
    ('mt_4_3', np.float32),
    ('scatter_type', np.uint16),
    ('recon_type', np.uint16),
    ('recon_views', np.uint16),
    ('fill', '174S'),
    ('fill2', '96S')]
subhdr_dtype = np.dtype(subheader_dtd)

# Ecat Data Types
_dtdefs = ( # code, name, equivalent dtype
    (1, 'ECAT7_BYTE', np.uint8),
    (2, 'ECAT7_VAXI2', np.int16),
    (3, 'ECAT7_VAXI4', np.float32),
    (4, 'ECAT7_VAXR4', np.float32),
    (5, 'ECAT7_IEEER4', np.float32),
    (6, 'ECAT7_SUNI2', np.uint16),
    (7, 'ECAT7_SUNI4', np.int32))
data_type_codes = make_dt_codes(_dtdefs)


# Matrix File Types
ft_defs = ( # code, name
    (0, 'ECAT7_UNKNOWN'),
    (1, 'ECAT7_2DSCAN'),
    (2, 'ECAT7_IMAGE16'),
    (3, 'ECAT7_ATTEN'),
    (4, 'ECAT7_2DNORM'),
    (5, 'ECAT7_POLARMAP'),
    (6, 'ECAT7_VOLUME8'),
    (7, 'ECAT7_VOLUME16'),
    (8, 'ECAT7_PROJ'),
    (9, 'ECAT7_PROJ16'),
    (10, 'ECAT7_IMAGE8'),
    (11, 'ECAT7_3DSCAN'),
    (12, 'ECAT7_3DSCAN8'),
    (13, 'ECAT7_3DNORM'),
    (14, 'ECAT7_3DSCANFIT'))

patient_orient_defs = ( #code, description
    (0, 'ECAT7_Feet_First_Prone'),
    (1, 'ECAT7_Head_First_Prone'),
    (2, 'ECAT7_Feet_First_Supine'),
    (3, 'ECAT7_Head_First_Supine'),
    (4, 'ECAT7_Feet_First_Decubitus_Right'),
    (5, 'ECAT7_Head_First_Decubitus_Right'),
    (6, 'ECAT7_Feet_First_Decubitus_Left'),
    (7, 'ECAT7_Head_First_Decubitus_Left'),
    (8, 'ECAT7_Unknown_Orientation'))

#Indexes from the patient_orient_defs structure defined above for the
#neurological and radiological viewing conventions
patient_orient_radiological = [0, 2, 4, 6]
patient_orient_neurological = [1, 3, 5, 7]

class EcatHeader(object):
    """Class for basic Ecat PET header
    Sub-parts of standard Ecat File
       main header
       matrix list
           which lists the information for each
           frame collected (can have 1 to many frames)
       subheaders specific to each frame
           with possibly-variable sized data blocks

    This just reads the main Ecat Header,
    it does not load the data
    or read the mlist or any sub headers

    """

    _dtype = hdr_dtype
    _ft_defs = ft_defs
    _patient_orient_defs = patient_orient_defs

    def __init__(self,
                 fileobj=None,
                 endianness=None):
        """Initialize Ecat header from file object

        Parameters
        ----------
        fileobj : {None, string} optional
            binary block to set into header, By default, None
            in which case we insert default empty header block
        endianness : {None, '<', '>', other endian code}, optional
            endian code of binary block, If None, guess endianness
            from the data
        """
        if fileobj is None:
            self._header_data = self._empty_headerdata(endianness)
            return

        hdr = np.ndarray(shape=(),
                         dtype=self._dtype,
                         buffer=fileobj)
        if endianness is None:
            endianness = self._guess_endian(hdr)

        if endianness != native_code:
            dt = self._dtype.newbyteorder(endianness)
            hdr = np.ndarray(shape=(),
                             dtype=dt,
                             buffer=fileobj)
        self._header_data = hdr.copy()

        return

    def get_header(self):
        """returns header """
        return self

    @property
    def binaryblock(self):
        return self._header_data.tostring()

    @property
    def endianness(self):
        if self._header_data.dtype.isnative:
            return native_code
        return swapped_code


    def _guess_endian(self, hdr):
        """Guess endian from MAGIC NUMBER value of header data
        """
        if not hdr['sw_version'] == 74:
            return swapped_code
        else:
            return native_code

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None):
        """Return /read header with given or guessed endian code

        Parameters
        ----------
        fileobj : file-like object
            Needs to implement ``read`` method
        endianness : None or endian code, optional
            Code specifying endianness of data to be read

        Returns
        -------
        hdr : EcatHeader object
            EcatHeader object initialized from data in file object

        Examples
        --------


        """
        raw_str = fileobj.read(klass._dtype.itemsize)
        return klass(raw_str, endianness)

    def write_to(self, fileobj):
        fileobj.write(self.binaryblock)

    def _empty_headerdata(self,endianness=None):
        """Return header data for empty header with given endianness"""
        #hdr_data = super(EcatHeader, self)._empty_headerdata(endianness)
        dt = self._dtype
        if not endianness is None:
            dt = dt.newbyteorder(endianness)
        hdr_data = np.zeros((), dtype=dt)
        hdr_data['magic_number'] = 'MATRIX72'
        hdr_data['sw_version'] = 74
        hdr_data['num_frames']= 0
        hdr_data['file_type'] = 0 # Unknown
        hdr_data['ecat_calibration_factor'] = 1.0 # scale factor
        return hdr_data


    def get_data_dtype(self):
        """ Get numpy dtype for data from header"""
        raise NotImplementedError("dtype is only valid from subheaders")


    def copy(self):
        return self.__class__(
            self.binaryblock,
            self.endianness)


    def __eq__(self, other):
        """ checks for equality between two headers"""
        self_end = self.endianness
        self_bb = self.binaryblock
        if self_end == other.endianness:
            return self_bb == other.binaryblock
        other_bb = other._header_data.byteswap().tostring()
        return self_bb == other_bb

    def __ne__(self, other):
        ''' equality between two headers defined by ``header_data``

        For examples, see ``__eq__`` method docstring
        '''
        return not self == other

    def __getitem__(self, item):
        ''' Return values from header data

        Examples
        --------
        >>> hdr = EcatHeader()
        >>> hdr['magic_number'] #23dt next : bytes
        'MATRIX72'
        '''
        return self._header_data[item].item()

    def __setitem__(self, item, value):
        ''' Set values in header data

        Examples
        --------
        >>> hdr = EcatHeader()
        >>> hdr['num_frames'] = 2
        >>> hdr['num_frames']
        2
        '''
        self._header_data[item] = value

    def get_patient_orient(self):
        """ gets orientation of patient based on code stored
        in header, not always reliable"""
        orient_code = dict(self._patient_orient_defs)
        code = self._header_data['patient_orientation'].item()
        if not orient_code.has_key(code):
            raise KeyError('Ecat Orientation CODE %d not recognized'%code)
        return orient_code[code]

    def get_filetype(self):
        """ gets type of ECAT Matrix File from
        code stored in header"""
        ft_codes = dict(self._ft_defs)
        code = self._header_data['file_type'].item()
        if not ft_codes.has_key(code):
            raise KeyError('Ecat Filetype CODE %d not recognized'%code)
        return ft_codes[code]

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        ''' Return keys from header data'''
        return list(self._dtype.names)

    def values(self):
        ''' Return values from header data'''
        data = self._header_data
        return [data[key] for key in self._dtype.names]

    def items(self):
        ''' Return items from header data'''
        return zip(self.keys(), self.values())

class EcatMlist(object):

    def __init__(self,fileobj, hdr):
        """ gets list of frames and subheaders in pet file

        Parameters
        -----------
        fileobj : ECAT file <filename>.v  fileholder or file object
                  with read, seek methods

        Returns
        -------
        mlist : numpy recarray  nframes X 4 columns
        1 - Matrix identifier.
        2 - subheader record number
        3 - Last record number of matrix data block.
        4 - Matrix status:
            1 - exists - rw
            2 - exists - ro
            3 - matrix deleted
        """
        self.hdr = hdr
        self._mlist = self.get_mlist(fileobj)

    def get_mlist(self, fileobj):
        fileobj.seek(512)
        dat=fileobj.read(128*32)

        dt = np.dtype([('matlist',np.int32)])
        if not self.hdr.endianness is native_code:
            dt = dt.newbyteorder(self.hdr.endianness)
        nframes = self.hdr['num_frames']
        mlist = np.zeros((nframes,4), dtype='uint32')
        record_count = 0
        done = False

        while not done: #mats['matlist'][0,1] == 2:

            mats = np.recarray(shape=(32,4), dtype=dt,  buf=dat)
            if not (mats['matlist'][0,0] +  mats['matlist'][0,3]) == 31:
                mlist = []
                return mlist

            nrecords = mats['matlist'][0,3]
            mlist[record_count:nrecords+record_count,:] = mats['matlist'][1:nrecords+1,:]
            record_count+= nrecords
            if mats['matlist'][0,1] == 2 or mats['matlist'][0,1] == 0:
                done = True
            else:
                # Find next subheader
                tmp = int(mats['matlist'][0,1]-1)#cast to int
                fileobj.seek(0)
                fileobj.seek(tmp*512)
                dat = fileobj.read(128*32)

        return mlist

    def get_frame_order(self):
        """Returns the order of the frames stored in the file
        Sometimes Frames are not stored in the file in
        chronological order, this can be used to extract frames
        in correct order

        Returns
        -------
        id_dict: dict mapping frame number -> [mlist_row, mlist_id]

        (where mlist id is value in the first column of the mlist matrix )

        Examples
        --------
        >>> import os
        >>> import nibabel as nib
        >>> nibabel_dir = os.path.dirname(nib.__file__)
        >>> from nibabel import ecat
        >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
        >>> img = ecat.load(ecat_file)
        >>> mlist = img.get_mlist()
        >>> mlist.get_frame_order()
        {0: [0, 16842758]}
        """
        mlist  = self._mlist
        ids = mlist[:, 0].copy()
        n_valid = np.sum(ids > 0)
        ids[ids <=0] = ids.max() + 1 # put invalid frames at end after sort
        valid_order = np.argsort(ids)
        if not all(valid_order == sorted(valid_order)):
            #raise UserWarning if Frames stored out of order
            warnings.warn_explicit('Frames stored out of order;'\
                          'true order = %s\n'\
                          'frames will be accessed in order '\
                          'STORED, NOT true order'%(valid_order),
                          UserWarning,'ecat', 0)
        id_dict = {}
        for i in range(n_valid):
            id_dict[i] = [valid_order[i], ids[valid_order[i]]]

        return id_dict

    def get_series_framenumbers(self):
        """ Returns framenumber of data as it was collected,
        as part of a series; not just the order of how it was
        stored in this or across other files

        For example, if the data is split between multiple files
        this should give you the true location of this frame as
        collected in the series
        (Frames are numbered starting at ONE (1) not Zero)

        Returns
        -------
        frame_dict: dict mapping order_stored -> frame in series
               where frame in series counts from 1; [1,2,3,4...]

        Examples
        --------
        >>> import os
        >>> import nibabel as nib
        >>> nibabel_dir = os.path.dirname(nib.__file__)
        >>> from nibabel import ecat
        >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
        >>> img = ecat.load(ecat_file)
        >>> mlist = img.get_mlist()
        >>> mlist.get_series_framenumbers()
        {0: 1}



        """
        frames_order = self.get_frame_order()
        nframes = self.hdr['num_frames']
        mlist_nframes = len(frames_order)
        trueframenumbers = np.arange(nframes - mlist_nframes, nframes)
        frame_dict = {}
        try:
            for frame_stored, (true_order, _) in frames_order.items():
                #frame as stored in file -> true number in series
                frame_dict[frame_stored] = trueframenumbers[true_order]+1
            return frame_dict
        except:
            raise IOError('Error in header or mlist order unknown')

class EcatSubHeader(object):

    _subhdrdtype = subhdr_dtype
    _data_type_codes = data_type_codes

    def __init__(self, hdr, mlist, fileobj):
        """parses the subheaders in the ecat (.v) file
        there is one subheader for each frame in the ecat file

        Parameters
        -----------
        hdr : EcatHeader

        mlist : EcatMlist

        fileobj : ECAT file <filename>.v  fileholder or file object
                  with read, seek methods


        """
        self._header = hdr
        self.endianness = hdr.endianness
        self._mlist = mlist
        self.fileobj = fileobj
        self.subheaders = self._get_subheaders()

    def _get_subheaders(self):
        """retreive all subheaders and return list of subheader recarrays
        """
        subheaders = []
        header = self._header
        endianness = self.endianness
        dt = self._subhdrdtype
        if not self.endianness is native_code:
            dt = self._subhdrdtype.newbyteorder(self.endianness)
        if self._header['num_frames'] > 1:
            for item in self._mlist._mlist:
                if item[1] == 0:
                    break
                self.fileobj.seek(0)
                offset = (int(item[1])-1)*512
                self.fileobj.seek(offset)
                tmpdat = self.fileobj.read(512)
                sh = (np.recarray(shape=(), dtype=dt,
                                  buf=tmpdat))
                subheaders.append(sh.copy())
        else:
            self.fileobj.seek(0)
            offset = (int(self._mlist._mlist[0][1])-1)*512
            self.fileobj.seek(offset)
            tmpdat = self.fileobj.read(512)
            sh = (np.recarray(shape=(), dtype=dt,
                              buf=tmpdat))
            subheaders.append(sh)
        return subheaders

    def get_shape(self, frame=0):
        """ returns shape of given frame"""
        subhdr = self.subheaders[frame]
        x = subhdr['x_dimension'].item()
        y = subhdr['y_dimension'].item()
        z = subhdr['z_dimension'].item()
        return (x,y,z)

    def get_nframes(self):
        """returns number of frames"""
        mlist = self._mlist
        framed = mlist.get_frame_order()
        return len(framed)


    def _check_affines(self):
        """checks if all affines are equal across frames"""
        nframes = self.get_nframes()
        if nframes == 1:
            return True
        affs = [self.get_frame_affine(i) for i in range(nframes)]
        if affs:
            i = iter(affs)
            first = i.next()
            for item in i:
                if not np.all(first == item):
                    return False
        return True

    def get_frame_affine(self,frame=0):
        """returns best affine for given frame of data"""
        subhdr = self.subheaders[frame]
        x_off = subhdr['x_offset']
        y_off = subhdr['y_offset']
        z_off = subhdr['z_offset']

        zooms = self.get_zooms(frame=frame)

        dims = self.get_shape(frame)
        # get translations from center of image
        origin_offset = (np.array(dims)-1) / 2.0
        aff = np.diag(zooms)
        aff[:3,-1] = -origin_offset * zooms[:-1] + np.array([x_off,y_off,z_off])
        return aff

    def get_zooms(self,frame=0):
        """returns zooms  ...pixdims"""
        subhdr = self.subheaders[frame]
        x_zoom = subhdr['x_pixel_size'] * 10
        y_zoom = subhdr['y_pixel_size'] * 10
        z_zoom = subhdr['z_pixel_size'] * 10
        return (x_zoom, y_zoom, z_zoom, 1)


    def _get_data_dtype(self, frame):
        dtcode = self.subheaders[frame]['data_type'].item()
        return self._data_type_codes.dtype[dtcode]

    def _get_frame_offset(self, frame=0):
        mlist = self._mlist._mlist
        offset = (mlist[frame][1]) * 512
        return int(offset)

    def _get_oriented_data(self, raw_data, orientation=None):
        '''
        Get data oriented following ``patient_orientation`` header field. If the
        ``orientation`` parameter is given, return data according to this
        orientation.

        :param raw_data: Numpy array containing the raw data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing the oriented data
        '''
        if orientation is None:
            orientation = self._header['patient_orientation']
        elif orientation == 'neurological':
            orientation = patient_orient_neurological[0]
        elif orientation == 'radiological':
            orientation = patient_orient_radiological[0]
        else:
            raise ValueError('orientation should be None,\
                neurological or radiological')

        if orientation in patient_orient_neurological:
            raw_data = raw_data[::-1, ::-1, ::-1]
        elif orientation in patient_orient_radiological:
            raw_data = raw_data[::, ::-1, ::-1]

        return raw_data

    def raw_data_from_fileobj(self, frame=0, orientation=None):
        '''
        Get raw data from file object.

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data

        .. seealso:: data_from_fileobj
        '''
        dtype = self._get_data_dtype(frame)
        if not self._header.endianness is native_code:
            dtype=dtype.newbyteorder(self._header.endianness)
        shape = self.get_shape(frame)
        offset = self._get_frame_offset(frame)
        fid_obj = self.fileobj
        raw_data = array_from_file(shape, dtype, fid_obj, offset=offset)
        raw_data = self._get_oriented_data(raw_data, orientation)
        return raw_data

    def data_from_fileobj(self, frame=0, orientation=None):
        '''
        Read scaled data from file for a given frame

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data

        .. seealso:: raw_data_from_fileobj
        '''
        header = self._header
        subhdr = self.subheaders[frame]
        raw_data = self.raw_data_from_fileobj(frame, orientation)
        data = raw_data * header['ecat_calibration_factor']
        data = data * subhdr['scale_factor']
        return data




class EcatImage(SpatialImage):
    """This class returns a list of Ecat images,
    with one image(hdr/data) per frame
    """
    _header = EcatHeader
    header_class = _header
    _subheader = EcatSubHeader
    _mlist = EcatMlist
    files_types = (('image', '.v'), ('header', '.v'))


    class ImageArrayProxy(object):
        ''' Ecat implemention of array proxy protocol

        The array proxy allows us to freeze the passed fileobj and
        header such that it returns the expected data array.
        '''
        def __init__(self, subheader):
            self._subheader = subheader
            self._data = None
            x, y, z = subheader.get_shape()
            nframes = subheader.get_nframes()
            self.shape = (x, y, z, nframes)

        def __array__(self):
            ''' Cached read of data from file
            This reads ALL FRAMES into one array, can be memory expensive
            use subheader.data_from_fileobj(frame) for less memory intensive
            reads
            '''
            if self._data is None:
                self._data = np.empty(self.shape)
                frame_mapping = self._subheader._mlist.get_frame_order()
                for i in sorted(frame_mapping):
                    self._data[:,:,:,i] = self._subheader.data_from_fileobj(frame_mapping[i][0])
            return self._data

    def __init__(self, data, affine, header,
                 subheader, mlist ,
                 extra = None, file_map = None):
        """ Initialize Image

        The image is a combination of
        (array, affine matrix, header, subheader, mlist)
        with optional meta data in `extra`, and filename / file-like objects
        contained in the `file_map`.

        Parameters
        ----------
        data : None or array-like
            image data
        affine : None or (4,4) array-like
            homogeneous affine giving relationship between voxel coords and
            world coords.
        header : None or header instance
            meta data for this image format
        subheader : None or subheader instance
            meta data for each sub-image for frame in the image
        mlist : None or mlist instance
            meta data with array giving offset and order of data in file
        extra : None or mapping, optional
            metadata associated with this image that cannot be
            stored in header or subheader
        file_map : mapping, optional
            mapping giving file information for this image format

        Examples
        --------
        >>> import os
        >>> import nibabel as nib
        >>> nibabel_dir = os.path.dirname(nib.__file__)
        >>> from nibabel import ecat
        >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
        >>> img = ecat.load(ecat_file)
        >>> frame0 = img.get_frame(0)
        >>> frame0.shape == (10, 10, 3)
        True
        >>> data4d = img.get_data()
        >>> data4d.shape == (10, 10, 3, 1)
        True
        """
        self._subheader = subheader
        self._mlist = mlist
        self._data = data
        if not affine is None:
            # Check that affine is array-like 4,4.  Maybe this is too strict at
            # this abstract level, but so far I think all image formats we know
            # do need 4,4.
            affine = np.asarray(affine)
            if not affine.shape == (4,4):
                raise ValueError('Affine should be shape 4,4')
        self._affine = affine
        if extra is None:
            extra = {}
        self.extra = extra
        self._header = header
        if file_map is None:
            file_map = self.__class__.make_file_map()
        self.file_map = file_map

    def _set_header(self, header):
        self._header = header

    def get_data(self):
        """returns scaled data for all frames in a numpy array
        returns as a 4D array """
        if self._data is None:
            raise ImageDataError('No data in this image')
        return np.asanyarray(self._data)

    def get_affine(self):
        if not self._subheader._check_affines():
            warnings.warn('Affines different across frames, loading affine from FIRST frame',
                          UserWarning )
        return self._affine

    def get_frame_affine(self, frame):
        """returns 4X4 affine"""
        return self._subheader.get_frame_affine(frame=frame)

    def get_frame(self,frame, orientation=None):
        '''
        Get full volume for a time frame

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data
        '''
        return self._subheader.data_from_fileobj(frame, orientation)

    def get_data_dtype(self,frame):
        subhdr = self._subheader
        dt = subhdr._get_data_dtype(frame)
        return dt

    @property
    def shape(self):
        x,y,z = self._subheader.get_shape()
        nframes = self._subheader.get_nframes()
        return(x, y, z, nframes)

    def get_mlist(self):
        """ get access to the mlist """
        return self._mlist

    def get_subheaders(self):
        """get access to subheaders"""
        return self._subheader

    @classmethod
    def from_filespec(klass, filespec):
        return klass.from_filename(filespec)


    @staticmethod
    def _get_fileholders(file_map):
        """ returns files specific to header and image of the image
        for ecat .v this is the same image file

        Returns
        -------
        header : file holding header data
        image : file holding image data
        """
        return file_map['header'], file_map['image']

    @classmethod
    def from_file_map(klass, file_map):
        """class method to create image from mapping
        specified in file_map"""
        hdr_file, img_file = klass._get_fileholders(file_map)
        #note header and image are in same file
        hdr_fid = hdr_file.get_prepare_fileobj(mode = 'rb')
        header = klass._header.from_fileobj(hdr_fid)
        hdr_copy = header.copy()
        ### LOAD MLIST
        mlist = klass._mlist(hdr_fid, hdr_copy)
        ### LOAD SUBHEADERS
        subheaders = klass._subheader(hdr_copy,
                                      mlist,
                                      hdr_fid)
        ### LOAD DATA
        ##  Class level ImageArrayProxy
        data = klass.ImageArrayProxy(subheaders)

        ## Get affine
        if not subheaders._check_affines():
            warnings.warn('Affines different across frames, loading affine from FIRST frame',
                          UserWarning )
        aff = subheaders.get_frame_affine()
        img = klass(data, aff, header, subheaders, mlist, extra=None, file_map = file_map)
        return img

    def _get_empty_dir(self):
        '''
        Get empty directory entry of the form
        [numAvail, nextDir, previousDir, numUsed]
        '''
        return np.array([31, 2, 0, 0], dtype=np.uint32)

    def _write_data(self, data, stream, pos, dtype=None, endianness=None):
        '''
        Write data to ``stream`` using an array_writer

        :param data: Numpy array containing the dat
        :param stream: The file-like object to write the data to
        :param pos: The position in the stream to write the data to
        :param endianness: Endianness code of the data to write
        '''
        if dtype is None:
            dtype = data.dtype

        if endianness is None:
            endianness = native_code

        stream.seek(pos)
        writer = make_array_writer(
            data.newbyteorder(endianness),
            dtype).to_fileobj(stream)

    def to_file_map(self, file_map=None):
        ''' Write ECAT7 image to `file_map` or contained ``self.file_map``

        The format consist of:

        - A main header (512L) with dictionary entries in the form
            [numAvail, nextDir, previousDir, numUsed]
        - For every frame (3D volume in 4D data)
          - A subheader (size = frame_offset)
          - Frame data (3D volume)
        '''
        if file_map is None:
            file_map = self.file_map

        data = self.get_data()
        hdr = self.get_header()
        mlist = self.get_mlist()._mlist
        subheaders = self.get_subheaders()
        dir_pos = 512L
        entry_pos = dir_pos + 16L #528L
        current_dir = self._get_empty_dir()

        hdr_fh, img_fh = self._get_fileholders(file_map)
        hdrf = hdr_fh.get_prepare_fileobj(mode='wb')
        imgf = hdrf

        #Write main header
        hdr.write_to(hdrf)

        #Write every frames
        for index in xrange(0, self.get_header()['num_frames']):
            #Move to subheader offset
            frame_offset = subheaders._get_frame_offset(index) - 512
            imgf.seek(frame_offset)

            #Write subheader
            subhdr = subheaders.subheaders[index]
            imgf.write(subhdr.tostring())

            #Seek to the next image block
            pos = imgf.tell()
            imgf.seek(pos + 2)

            #Get frame and its data type
            image = self._subheader.raw_data_from_fileobj(index)
            dtype = image.dtype

            #Write frame images
            self._write_data(image, imgf, pos+2, endianness='>')

            #Move to dictionnary offset and write dictionnary entry
            self._write_data(mlist[index], imgf, entry_pos,
                np.uint32, endianness='>')

            entry_pos = entry_pos + 16L

            current_dir[0] = current_dir[0] - 1
            current_dir[3] = current_dir[3] + 1

            #Create a new directory is previous one is full
            if current_dir[0] == 0:
                #self._write_dir(current_dir, imgf, dir_pos)
                self._write_data(current_dir, imgf, dir_pos)
                current_dir = self._get_empty_dir()
                current_dir[3] = dir_pos / 512L
                dir_pos = mlist[index][2] + 1
                entry_pos = dir_pos + 16L

        tmp_avail = current_dir[0]
        tmp_used = current_dir[3]

        #Fill directory with empty data until directory is full
        while current_dir[0] > 0:
            entry_pos = dir_pos + 16L + (16L * current_dir[3])
            self._write_data(np.array([0,0,0,0]), imgf, entry_pos, np.uint32)
            current_dir[0] = current_dir[0] - 1
            current_dir[3] = current_dir[3] + 1

        current_dir[0] = tmp_avail
        current_dir[3] = tmp_used

        #Write directory index
        self._write_data(current_dir, imgf, dir_pos, endianness='>')


    @classmethod
    def from_image(klass, img):
        raise NotImplementedError("Ecat images can only be generated "\
                                  "from file objects")

    @classmethod
    def load(klass, filespec):
        return klass.from_filename(filespec)


load = EcatImage.load

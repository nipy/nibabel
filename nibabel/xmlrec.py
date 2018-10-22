from copy import copy, deepcopy
import os
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

from .affines import from_matvec, dot_reduce, apply_affine
from .deprecated import deprecate_with_version
from .nifti1 import unit_codes
from .eulerangles import euler2mat
from .parrec import (one_line, PARRECImage, PARRECHeader, ACQ_TO_PSL,
                     PSL_TO_RAS, vol_is_full, vol_numbers, DEG2RAD)


class XMLRECError(Exception):
    """Exception for XML/REC format related problems.

    To be raised whenever XML/REC is not happy, or we are not happy with
    XML/REC.
    """

slice_orientation_rename = dict(Transversal='transverse',
                                Sagittal='sagittal',
                                Coronal='coronal')


# Dict for converting XML/REC types to appropriate python types
# could convert the Enumeration strings to int, but enums_dict is incomplete

def _xml_str_to_bool(s):
    if isinstance(s, bytes):
        s = s.decode()
    if s == 'Y':
        return True
    else:
        return False

# and the strings are more easily interpretable
xml_type_dict = {'Float': np.float32,
                 'Double': np.float64,
                 'Int16': np.int16,
                 'Int32': np.int32,
                 'UInt16': np.uint16,
                 'Enumeration': str,
                 'Boolean': bool,
                 'String': str,
                 'Date': str,
                 'Time': str}

# choose appropriate string length for use within the structured array dtype
str_length_dict = {'Enumeration': '|S32',
                   'String': '|S128',
                   'Date': '|S16',
                   'Time': '|S16'}

supported_xml_versions = ['PRIDE_V5']


def _process_gen_dict_xml(xml_root):
    """Read the general_info from an XML file.

    This is the equivalent of _process_gen_dict() for .PAR files
    """
    info = xml_root.find('Series_Info')
    if info is None:
        raise RuntimeError("No 'Series_Info' found in the XML file")
    general_info = {}
    for e in info:
        a = e.attrib
        if 'Name' in a:
            entry_type = xml_type_dict[a['Type']]
            if entry_type == bool:
                # convert string such as 'N' or 'Y' to boolean
                cast = _xml_str_to_bool
            else:
                cast = entry_type
            if 'ArraySize' in a:
                val = [cast(i) for i in e.text.strip().split()]
            else:
                val = cast(e.text)
            general_info[a['Name']] = val
    return general_info


def _get_image_def_attributes(xml_root):
    """Get names and dtypes for all attributes defined for each image.

    called by _process_image_lines_xml

    Paramters
    ---------
    xml_root :
        The root of the XML tree.

    Returns
    -------
    key_attributes : list of tuples
        The attributes that are used when sorting data from the REC file.
    other_attributes : list of tuples
        Additional attributes that are not considered when sorting.
    """
    image_defs_array = xml_root.find('Image_Array')
    if image_defs_array is None:
        raise XMLRECError("No 'Image_Array' found in the XML file")

    # can get all of the fields from the first entry
    first_def = image_defs_array[0]

    # Key element contains attributes corresponding to image keys
    # e.g. (Slice, Echo, Dynamic, ...) that can be used for sorting the images.
    img_keys = first_def.find('Key')
    if img_keys is None:
        raise XMLRECError(
            "Expected to find a Key attribute for each element in Image_Array")
    if not np.all(['Name' in k.attrib for k in img_keys]):
        raise XMLRECError("Expected each Key attribute to have a Name")

    def _get_type(name, type_dict, str_length_dict=str_length_dict):
        t = type_dict[name]
        if t is str:  # convert from str to a specific length such as |S32
            t = str_length_dict[name]
        return t

    # xml_type_dict keeps enums as their string representation
    key_attributes = [
        (k.get('Name'), _get_type(k.get('Type'), xml_type_dict))
        for k in img_keys]

    # Process other attributes that are not considered image keys
    other_attributes = []
    for element in first_def:
        a = element.attrib
        if 'Name' in a:
            # if a['Type'] == 'Enumeration':
            #     enum_type = a['EnumType']
            #     print("enum_type = {}".format(enum_type))
            name = a['Name']
            entry_type = _get_type(a['Type'], xml_type_dict)
            if 'ArraySize' in a:
                # handle vector entries (e.g. 'Pixel Size' is length 2)
                entry = (name, entry_type, int(a['ArraySize']))
            else:
                entry = (name, entry_type)
            other_attributes.append(entry)

    return key_attributes, other_attributes


def _process_image_lines_xml(xml_root):
    """Build image_defs by parsing the XML file.

    Parameters
    ----------
    xml_root :
        The root of the XML tree.

    Returns
    -------
    image_defs : np.ndarray
        A structured array with the labels for each 2D image frame.
    """
    image_defs_array = xml_root.find('Image_Array')
    if image_defs_array is None:
        raise RuntimeError("No 'Image_Array' found in the XML file")

    key_attributes, other_attributes = _get_image_def_attributes(xml_root)

    image_def_dtd = key_attributes + other_attributes
    # dtype dict based on the XML attribute names
    dtype_dict = {a[0]: a[1] for a in image_def_dtd}

    def _get_val(entry_dtype, text):
        if entry_dtype == '|S16':
            val = text[:16]
        elif entry_dtype == '|S32':
            val = text[:32]
        elif entry_dtype == '|S128':
            val = text[:128]
        elif entry_dtype == bool:
            # convert string 'Y' or 'N' to boolean
            val = _xml_str_to_bool(text)
        else:
            val = entry_dtype(text)
        return val

    image_defs = np.zeros(len(image_defs_array), dtype=image_def_dtd)
    for i, image_def in enumerate(image_defs_array):

        if image_def.find('Key') != image_def[0]:
            raise RuntimeError("Expected first element of image_def to be Key")

        key_def = image_def[0]
        for key in key_def:
            name = key.get('Name')
            val = key.text
            image_defs[name][i] = _get_val(dtype_dict[name], val)

        # for all image properties we know about
        for element in image_def[1:]:
            a = element.attrib
            text = element.text
            if 'Name' in a:
                name = a['Name']
                entry_dtype = dtype_dict[name]
                if 'ArraySize' in a:
                    val = [entry_dtype(i) for i in text.strip().split()]
                else:
                    val = _get_val(entry_dtype, text)
                image_defs[name][i] = val
    return image_defs


def parse_XML_header(fobj):
    """Parse a XML header and aggregate all information into useful containers.

    Parameters
    ----------
    fobj : file-object or str
        The XML header file object or file name.

    Returns
    -------
    general_info : dict
        Contains all "General Information" from the header file
    image_info : ndarray
        Structured array with fields giving all "Image information" in the
        header
    """
    tree = ET.parse(fobj)
    root = tree.getroot()

    version = root.tag  # e.g. PRIDE_V5
    if version not in supported_xml_versions:
        warnings.warn(one_line(
            """XML version '{0}' is currently not supported.  Only PRIDE_V5 XML
            files have been tested. --making an attempt to read nevertheless.
            Please email the NiBabel mailing list, if you are interested in
            adding support for this version.
            """.format(version)))
    try:
        general_info = _process_gen_dict_xml(root)
        image_defs = _process_image_lines_xml(root)
    except ET.ParseError:
            raise XMLRECError(
                "A ParseError occured in the ElementTree library while "
                "reading the XML file. This may be due to a truncated XML "
                "file.")

    return general_info, image_defs


def _truncation_checks(general_info, image_defs, permit_truncated):
    """ Check for presence of truncation in XML file parameters

    Raise error if truncation present and `permit_truncated` is False.
    """
    def _err_or_warn(msg):
        if not permit_truncated:
            raise XMLRECError(msg)
        warnings.warn(msg)

    def _chk_trunc(idef_name, gdef_max_name):
        if gdef_max_name not in general_info:
            return
        id_values = image_defs[idef_name]
        n_have = len(set(id_values))
        n_expected = general_info[gdef_max_name]
        if n_have != n_expected:
            _err_or_warn(
                "Header inconsistency: Found {0} {1} values, "
                "but expected {2}".format(n_have, idef_name, n_expected))

    _chk_trunc('Slice', 'Max No Slices')
    _chk_trunc('Echo', 'Max No Echoes')
    _chk_trunc('Dynamic', 'Max No Dynamics')
    _chk_trunc('BValue', 'Max No B Values')
    _chk_trunc('Grad Orient', 'Max No Gradient Orients')

    # Final check for partial volumes
    if not np.all(vol_is_full(image_defs['Slice'],
                              general_info['Max No Slices'])):
        _err_or_warn("Found one or more partial volume(s)")


class XMLRECHeader(PARRECHeader):
    """XML/REC header"""

    def __init__(self, info, image_defs, permit_truncated=False,
                 strict_sort=False):
        """
        Parameters
        ----------
        info : dict
            "General information" from the XML file (as returned by
            `parse_XML_header()`).
        image_defs : array
            Structured array with image definitions from the XML file (as
            returned by `parse_XML_header()`).
        permit_truncated : bool, optional
            If True, a warning is emitted instead of an error when a truncated
            recording is detected.
        strict_sort : bool, optional, keyword-only
            If True, a larger number of header fields are used while sorting
            the REC data array.  This may produce a different sort order than
            `strict_sort=False`, where volumes are sorted by the order in which
            the slices appear in the .xml file.
        """
        self.general_info = info.copy()
        self.image_defs = image_defs.copy()
        self.permit_truncated = permit_truncated
        self.strict_sort = strict_sort
        _truncation_checks(info, image_defs, permit_truncated)
        # charge with basic properties to be able to use base class
        # functionality
        # dtype
        bitpix = self._get_unique_image_prop('Pixel Size')
        if bitpix not in (8, 16):
            raise XMLRECError('Only 8- and 16-bit data supported (not %s)'
                              'please report this to the nibabel developers'
                              % bitpix)
        # REC data always little endian
        dt = np.dtype('uint' + str(bitpix)).newbyteorder('<')
        super(PARRECHeader, self).__init__(data_dtype=dt,
                                           shape=self._calc_data_shape(),
                                           zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise XMLRECError('Cannot create XMLRECHeader from air.')
        if type(header) == klass:
            return header.copy()
        raise XMLRECError('Cannot create XMLREC header from '
                          'non-XMLREC header.')

    @classmethod
    def from_fileobj(klass, fileobj, permit_truncated=False,
                     strict_sort=False):
        info, image_defs = parse_XML_header(fileobj)
        return klass(info, image_defs, permit_truncated, strict_sort)

    def copy(self):
        return XMLRECHeader(deepcopy(self.general_info),
                            self.image_defs.copy(),
                            self.permit_truncated,
                            self.strict_sort)

    def as_analyze_map(self):
        """Convert XML parameters to NIFTI1 format"""
        # Entries in the dict correspond to the parameters found in
        # the NIfTI1 header, specifically in nifti1.py `header_dtd` defs.
        # Here we set the parameters we can to simplify XML/REC
        # to NIfTI conversion.
        exam_date = '{}/{}'.format(
            self.general_info['Examination Date'],
            self.general_info['Examination Time'])
        descr = ("%s;%s;%s;%s"
                 % (self.general_info['Examination Name'],
                    self.general_info['Patient Name'],
                    exam_date,
                    self.general_info['Protocol Name']))[:80]  # max len
        is_fmri = (self.general_info['Max No Dynamics'] > 1)
        t = 'msec' if is_fmri else 'unknown'
        xyzt_units = unit_codes['mm'] + unit_codes[t]
        return dict(descr=descr, xyzt_units=xyzt_units)  # , pixdim=pixdim)

    def get_water_fat_shift(self):
        """Water fat shift, in pixels"""
        return self.general_info['Water Fat Shift']

    def get_echo_train_length(self):
        """Echo train length of the recording"""
        return self.general_info['EPI factor']

    def get_bvals_bvecs(self):
        """Get bvals and bvecs from data

        Returns
        -------
        b_vals : None or array
            Array of b values, shape (n_directions,), or None if not a
            diffusion acquisition.
        b_vectors : None or array
            Array of b vectors, shape (n_directions, 3), or None if not a
            diffusion acquisition.
        """
        if not self.general_info['Diffusion']:
            return None, None
        reorder = self.get_sorted_slice_indices()
        if len(self.get_data_shape()) == 3:
            n_slices, n_vols = self.get_data_shape()[-1], 1
        else:
            n_slices, n_vols = self.get_data_shape()[-2:]
        bvals = self.image_defs['Diffusion B Factor'][reorder].reshape(
            (n_slices, n_vols), order='F')
        # All bvals within volume should be the same
        assert not np.any(np.diff(bvals, axis=0))
        bvals = bvals[0]
        if 'Diffusion' not in self.image_defs.dtype.names:
            return bvals, None
        bvecs = self.image_defs['Diffusion'][reorder].reshape(
            (n_slices, n_vols, 3), order='F')
        # All 3 values of bvecs should be same within volume
        assert not np.any(np.diff(bvecs, axis=0))
        bvecs = bvecs[0]
        # rotate bvecs to match stored image orientation
        permute_to_psl = ACQ_TO_PSL[self.get_slice_orientation()]
        bvecs = apply_affine(np.linalg.inv(permute_to_psl), bvecs)
        return bvals, bvecs

    def _get_unique_resolution(self):
        """Return the 2D image plane shape.

        An error is raised if the shape is not unique.
        """
        resx = self.image_defs['Resolution X']
        resy = self.image_defs['Resolution Y']
        if len(set(zip(resx, resy))) > 1:
            raise XMLRECError('Varying resolution in image sequence. This is '
                              'not suppported.')
        return (resx[0], resy[0])

    @deprecate_with_version('get_voxel_size deprecated. '
                            'Please use "get_zooms" instead.',
                            '2.0', '4.0')
    def get_voxel_size(self):
        """Returns the spatial extent of a voxel.

        Does not include the slice gap in the slice extent.

        If you need the slice thickness not including the slice gap, use
        ``self.image_defs['slice thickness']``.

        Returns
        -------
        vox_size: shape (3,) ndarray
        """
        # slice orientation for the whole image series
        slice_thickness = self._get_unique_image_prop('Slice Thickness')
        voxsize_inplane = self._get_unique_image_prop('Pixel Spacing')
        voxsize = np.array((voxsize_inplane[0],
                            voxsize_inplane[1],
                            slice_thickness))
        return voxsize

    def _calc_zooms(self):
        """Compute image zooms from header data.

        Spatial axis are first three.

        Returns
        -------
        zooms : array
            Length 3 array for 3D image, length 4 array for 4D image.

        Notes
        -----
        This routine gets called in ``__init__``, so may not be able to use
        some attributes available in the fully initialized object.
        """
        # slice orientation for the whole image series
        slice_gap = self._get_unique_image_prop('Slice Gap')
        # scaling per image axis
        n_dim = 4 if self._get_n_vols() > 1 else 3
        zooms = np.ones(n_dim)
        # spatial sizes are inplane X mm, inplane Y mm + inter slice gap
        zooms[:2] = self._get_unique_image_prop('Pixel Spacing')
        slice_thickness = self._get_unique_image_prop('Slice Thickness')
        zooms[2] = slice_thickness + slice_gap
        # If 4D dynamic scan, convert time from milliseconds to seconds
        if len(zooms) > 3 and self.general_info['Dynamic Scan']:
            if len(self.general_info['Repetition Times']) > 1:
                warnings.warn("multiple TRs found in header file")
            zooms[3] = self.general_info['Repetition Times'][0] / 1000.
        return zooms

    def get_affine(self, origin='scanner'):
        """Compute affine transformation into scanner space.

        The method only considers global rotation and offset settings in the
        header and ignores potentially deviating information in the image
        definitions.

        Parameters
        ----------
        origin : {'scanner', 'fov'}
            Transformation origin. By default the transformation is computed
            relative to the scanner's iso center. If 'fov' is requested the
            transformation origin will be the center of the field of view
            instead.

        Returns
        -------
        aff : (4, 4) array
            4x4 array, with output axis order corresponding to RAS or (x,y,z)
            or (lr, pa, fh).

        Notes
        -----
        Transformations appear to be specified in (ap, fh, rl) axes.  The
        orientation of data is recorded in the "Slice Orientation" field of the
        XML header "General Information".

        We need to:

        * translate to coordinates in terms of the center of the FOV
        * apply voxel size scaling
        * reorder / flip the data to Philips' PSL axes
        * apply the rotations
        * apply any isocenter scaling offset if `origin` == "scanner"
        * reorder and flip to RAS axes
        """
        # shape, zooms in original data ordering (ijk ordering)
        ijk_shape = np.array(self.get_data_shape()[:3])
        to_center = from_matvec(np.eye(3), -(ijk_shape - 1) / 2.)
        zoomer = np.diag(list(self.get_zooms()[:3]) + [1])
        slice_orientation = self.get_slice_orientation()
        permute_to_psl = ACQ_TO_PSL.get(slice_orientation)
        if permute_to_psl is None:
            raise XMLRECError(
                "Unknown slice orientation ({0}).".format(slice_orientation))
        # hdr has deg, we need radians
        # Order is [ap, fh, rl]
        ap_rot = self.general_info['Angulation AP'] * DEG2RAD
        fh_rot = self.general_info['Angulation FH'] * DEG2RAD
        rl_rot = self.general_info['Angulation RL'] * DEG2RAD
        Mx = euler2mat(x=ap_rot)
        My = euler2mat(y=fh_rot)
        Mz = euler2mat(z=rl_rot)
        # By trial and error, this unexpected order of rotations seem to give
        # the closest to the observed (converted NIfTI) affine.
        rot = from_matvec(dot_reduce(Mz, Mx, My))
        # compose the PSL affine
        psl_aff = dot_reduce(rot, permute_to_psl, zoomer, to_center)
        if origin == 'scanner':
            # offset to scanner's isocenter (in ap, fh, rl)
            iso_offset = np.asarray([self.general_info['Off Center AP'],
                                     self.general_info['Off Center FH'],
                                     self.general_info['Off Center RL']])
            psl_aff[:3, 3] += iso_offset
        # Currently in PSL; apply PSL -> RAS
        return np.dot(PSL_TO_RAS, psl_aff)

    def _get_n_slices(self):
        """ Get number of slices for output data """
        return len(set(self.image_defs['Slice']))

    def _get_n_vols(self):
        """ Get number of volumes for output data """
        slice_nos = self.image_defs['Slice']
        vol_nos = vol_numbers(slice_nos)
        is_full = vol_is_full(slice_nos, self.general_info['Max No Slices'])
        return len(set(np.array(vol_nos)[is_full]))

    def _calc_data_shape(self):
        """ Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        n_inplaneX : int
            number of voxels in X direction.
        n_inplaneY : int
            number of voxels in Y direction.
        n_slices : int
            number of slices.
        n_vols : int
            number of volumes or absent for 3D image.

        Notes
        -----
        This routine gets called in ``__init__``, so may not be able to use
        some attributes available in the fully initialized object.
        """
        inplane_shape = self._get_unique_resolution()
        shape = inplane_shape + (self._get_n_slices(),)
        n_vols = self._get_n_vols()
        return shape + (n_vols,) if n_vols > 1 else shape

    def get_data_scaling(self, method="dv"):
        """Returns scaling slope and intercept.

        Parameters
        ----------
        method : {'fp', 'dv'}
          Scaling settings to be reported -- see notes below.

        Returns
        -------
        slope : array
            scaling slope
        intercept : array
            scaling intercept

        Notes
        -----
        The XML header contains two different scaling settings: 'dv' (value on
        console) and 'fp' (floating point value). Here is how they are defined:

        DV = PV * RS + RI
        FP = DV / (RS * SS)

        where:

        PV: value in REC
        RS: rescale slope
        RI: rescale intercept
        SS: scale slope
        """
        # These will be 3D or 4D
        scale_slope = self.image_defs['Scale Slope']
        rescale_slope = self.image_defs['Rescale Slope']
        rescale_intercept = self.image_defs['Rescale Intercept']
        if method == 'dv':
            slope, intercept = rescale_slope, rescale_intercept
        elif method == 'fp':
            slope = 1.0 / scale_slope
            intercept = rescale_intercept / (rescale_slope * scale_slope)
        else:
            raise ValueError("Unknown scaling method '%s'." % method)
        reorder = self.get_sorted_slice_indices()
        slope = slope[reorder]
        intercept = intercept[reorder]
        shape = (1, 1) + self.get_data_shape()[2:]
        slope = slope.reshape(shape, order='F')
        intercept = intercept.reshape(shape, order='F')
        return slope, intercept

    def get_slice_orientation(self):
        """Returns the slice orientation label.

        Returns
        -------
        orientation : {'transverse', 'sagittal', 'coronal'}
        """
        orientations = list(self.image_defs['Slice Orientation'])
        if len(set(orientations)) > 1:
            raise XMLRECError(
                'Varying slice orientation found in the image sequence. This '
                ' is not suppported.')
        return slice_orientation_rename[orientations[0].decode()]

    def get_rec_shape(self):
        inplane_shape = self._get_unique_resolution()
        return inplane_shape + (len(self.image_defs),)

    def _strict_sort_order(self):
        """ Determine the sort order based on several image definition fields.

        The fields taken into consideration, if present, are (in order from
        slowest to fastest variation after sorting):

            - image_defs['Type']              # Re, Im, Mag, Phase
            - image_defs['Dynamicr']          # repetition
            - image_defs['Label Type']        # ASL tag/control
            - image_defs['BValue']            # diffusion b value
            - image_defs['Grad Orient']  # diffusion directoin
            - image_defs['Phase']             # cardiac phase
            - image_defs['Echo']              # echo
            - image_defs['Slice']             # slice
            - image_defs['Index']             # index in the REC file

        Data sorting is done in two stages:

            1. an initial sort using the keys described above
            2. a resort after generating two additional sort keys:

                * a key to assign unique volume numbers to any volumes that
                  didn't have a unique sort based on the keys above
                  (see :func:`vol_numbers`).
                * a sort key based on `vol_is_full` to identify truncated
                  volumes

        """
        # sort keys present in all supported .xml versions
        idefs = self.image_defs
        index_nos = idefs['Index']
        slice_nos = idefs['Slice']
        dynamics = idefs['Dynamic']
        phases = idefs['Phase']
        echos = idefs['Echo']
        image_type = idefs['Type']
        asl_keys = (idefs['Label Type'], )
        diffusion_keys = (idefs['BValue'], idefs['Grad Orient'])

        # initial sort (last key is highest precedence)
        keys = (index_nos, slice_nos, echos, phases) + \
            diffusion_keys + asl_keys + (dynamics, image_type)
        initial_sort_order = np.lexsort(keys)

        # sequentially number the volumes based on the initial sort
        vol_nos = vol_numbers(slice_nos[initial_sort_order])
        # identify truncated volumes
        is_full = vol_is_full(slice_nos[initial_sort_order],
                              self.general_info['Max No Slices'])

        # second stage of sorting
        return initial_sort_order[np.lexsort((vol_nos, is_full))]

    def _lax_sort_order(self):
        """
        Sorts by (fast to slow): slice number, volume number.

        We calculate volume number by looking for repeating slice numbers (see
        :func:`vol_numbers`).
        """
        slice_nos = self.image_defs['Slice']
        is_full = vol_is_full(slice_nos, self.general_info['Max No Slices'])
        keys = (slice_nos, vol_numbers(slice_nos), np.logical_not(is_full))
        return np.lexsort(keys)

    def get_volume_labels(self):
        """ Dynamic labels corresponding to the final data dimension(s).

        This is useful for custom data sorting.  A subset of the info in
        ``self.image_defs`` is returned in an order that matches the final
        data dimension(s).  Only labels that have more than one unique value
        across the dataset will be returned.

        Returns
        -------
        sort_info : dict
            Each key corresponds to volume labels for a dynamically varying
            sequence dimension.  The ordering of the labels matches the volume
            ordering determined via ``self.get_sorted_slice_indices``.
        """
        sorted_indices = self.get_sorted_slice_indices()
        image_defs = self.image_defs

        # define which keys which might vary across image volumes
        dynamic_keys = ['Phase',
                        'Echo',
                        'Label Type',
                        'Type',
                        'Dynamic',
                        'Sequence',
                        'Grad Orient',
                        'BValue']

        non_unique_keys = []
        for key in dynamic_keys:
            ndim = image_defs[key].ndim
            if ndim == 1:
                num_unique = len(np.unique(image_defs[key]))
            else:
                raise ValueError("unexpected image_defs shape > 1D")
            if num_unique > 1:
                non_unique_keys.append(key)

        # each key in dynamic keys will be identical across slices, so use
        # the value at slice 1.
        sl1_indices = image_defs['Slice'][sorted_indices] == 1

        sort_info = OrderedDict()
        for key in non_unique_keys:
            sort_info[key] = image_defs[key][sorted_indices][sl1_indices]
        return sort_info


class XMLRECImage(PARRECImage):
    """XML/REC image"""
    header_class = XMLRECHeader
    valid_exts = ('.rec', '.xml')
    files_types = (('image', '.rec'), ('header', '.xml'))

    @classmethod
    def filespec_to_file_map(klass, filespec):
        file_map = super(PARRECImage, klass).filespec_to_file_map(filespec)
        # fix case of .REC (.xml/.REC tends to have mixed case file extensions)
        fname = file_map['image'].filename
        fname_upper = fname.replace('.rec', '.REC')
        if (not os.path.exists(fname) and
                os.path.exists(fname_upper)):
            file_map['image'].filename = fname_upper
        return file_map

load = XMLRECImage.load

from copy import copy, deepcopy
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from nibabel.parrec import one_line
from nibabel.parrec import PARRECImage, PARRECHeader


class XMLRECError(Exception):
    """Exception for XML/REC format related problems.

    To be raised whenever XML/REC is not happy, or we are not happy with
    XML/REC.
    """

# Dictionary of conversions from enumerated types to integer value for use in
# converting from XML enum names to PAR-style integers.
# The keys in enums_dict are the names of the enum keys used in XML files.
# The present conversions for strings to enumerated values was determined
# empirically via comparison of simultaneously exported .PAR and .xml files
# from a range of different scan types. Any enum labels that are not recognized
# will result in a warning encouraging the user to report the unkown case to
# the nibabel developers.
enums_dict = {
    'Label Type': {'CONTROL': 1, 'LABEL': 2, '-': 1},
    'Type': {'M': 0, 'R': 1, 'I': 2, 'P': 3, 'T1': 6, 'T2': 7, 'ADC': 11,
             'EADC': 17, 'B0': 18, 'PERFUSION': 30, 'F': 31, 'IP': 32,
             'FF': 34, 'R2': -1, 'R2_STAR': -1, 'T2_STAR': -1, 'W': -1,
             'STIFF': -1, 'WAVE': -1, 'SW_M': -1, 'SW_P': -1},
    'Sequence': {'IR': 0, 'SE': 1, 'FFE': 2, 'PCA': 4, 'UNSPECIFIED': 5,
                 'DERIVED': 7, 'B1': 9, 'MRE': 10},
    'Image Type Ed Es': {'U': 2},
    'Display Orientation': {'-': 0, 'NONE': 0},
    'Slice Orientation': {'Transversal': 1, 'Sagittal': 2, 'Coronal': 3},
    'Contrast Type': {'DIFFUSION': 0, 'FLOW_ENCODED': 1, 'PERFUSION': 3,
                      'PROTON_DENSITY': 4, 'TAGGING': 6, 'T1': 7, 'T2': 8,
                      'UNKNOWN': 11},
    'Diffusion Anisotropy Type': {'-': 0}}

# Dict for converting XML/REC types to appropriate python types
# could convert the Enumeration strings to int, but enums_dict is incomplete
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

# for use with parrec.py, the Enumeration strings must be converted to ints
par_type_dict = copy(xml_type_dict)
par_type_dict['Enumeration'] = int

supported_xml_versions = ['PRIDE_V5']

# convert XML names to expected property names as used in nibabel.parrec
# Note: additional special cases are handled in general_info_xml_to_par()
general_info_XML_to_nibabel = {
    'Patient Name': 'patient_name',
    'Examination Name': 'exam_name',
    'Protocol Name': 'protocol_name',
    'Aquisition Number': 'acq_nr',
    'Reconstruction Number': 'recon_nr',
    'Scan Duration': 'scan_duration',
    'Max No Phases': 'max_cardiac_phases',
    'Max No Echoes': 'max_echoes',
    'Max No Slices': 'max_slices',
    'Max No Dynamics': 'max_dynamics',
    'Max No Mixes': 'max_mixes',
    'Patient Position': 'patient_position',
    'Preparation Direction': 'prep_direction',
    'Technique': 'tech',
    'Scan Mode': 'scan_mode',
    'Repetition Times': 'repetition_time',
    'Water Fat Shift': 'water_fat_shift',
    'Flow Compensation': 'flow_compensation',
    'Presaturation': 'presaturation',
    'Phase Encoding Velocity': 'phase_enc_velocity',
    'MTC': 'mtc',
    'SPIR': 'spir',
    'EPI factor': 'epi_factor',
    'Dynamic Scan': 'dyn_scan',
    'Diffusion': 'diffusion',
    'Diffusion Echo Time': 'diffusion_echo_time',
    'Max No B Values': 'max_diffusion_values',
    'Max No Gradient Orients': 'max_gradient_orient',
    'No Label Types': 'nr_label_types',
    'Series Data Type': 'series_type'}


def general_info_xml_to_par(xml_info):
    """Convert general_info from XML-style names to PAR-style names."""
    xml_info_init = xml_info
    xml_info = deepcopy(xml_info)
    general_info = {}
    for k in xml_info_init.keys():
        # convert all keys with a simple 1-1 name conversion
        if k in general_info_XML_to_nibabel:
            general_info[general_info_XML_to_nibabel[k]] = xml_info.pop(k)
    try:
        general_info['exam_date'] = '{} / {}'.format(
            xml_info.pop('Examination Date'),
            xml_info.pop('Examination Time'))
    except KeyError:
        pass

    try:
        general_info['angulation'] = np.asarray(
            [xml_info.pop('Angulation AP'),
             xml_info.pop('Angulation FH'),
             xml_info.pop('Angulation RL')])
    except KeyError:
        pass

    try:
        general_info['off_center'] = np.asarray(
            [xml_info.pop('Off Center AP'),
             xml_info.pop('Off Center FH'),
             xml_info.pop('Off Center RL')])
    except KeyError:
        pass

    try:
        general_info['fov'] = np.asarray(
            [xml_info.pop('FOV AP'),
             xml_info.pop('FOV FH'),
             xml_info.pop('FOV RL')])
    except KeyError:
        pass

    try:
        general_info['scan_resolution'] = np.asarray(
            [xml_info.pop('Scan Resolution X'),
             xml_info.pop('Scan Resolution Y')],
            dtype=int)
    except KeyError:
        pass

    # copy any excess keys not normally in the .PARREC general info
    # These will not be used by the PARREC code, but are kept for completeness
    general_info.update(xml_info)

    return general_info


# TODO: remove this function? It is currently unused, but may be useful for
#       testing roundtrip convesion.
def general_info_par_to_xml(par_info):
    """Convert general_info from PAR-style names to XML-style names."""
    general_info_nibabel_to_XML = {
        v: k for k, v in general_info_XML_to_nibabel.items()}
    par_info_init = par_info
    par_info = deepcopy(par_info)
    general_info = {}
    for k in par_info_init.keys():
        # convert all keys with a simple 1-1 name conversion
        if k in general_info_nibabel_to_XML:
            general_info[general_info_nibabel_to_XML[k]] = par_info.pop(k)
    try:
        tmp = par_info['exam_date'].split('/')
        general_info['Examination Date'] = tmp[0].strip()
        general_info['Examination Time'] = tmp[1].strip()
    except KeyError:
        pass

    try:
        general_info['Angulation AP'] = par_info['angulation'][0]
        general_info['Angulation FH'] = par_info['angulation'][1]
        general_info['Angulation RL'] = par_info['angulation'][2]
        par_info.pop('angulation')
    except KeyError:
        pass

    try:
        general_info['Off Center AP'] = par_info['off_center'][0]
        general_info['Off Center FH'] = par_info['off_center'][1]
        general_info['Off Center RL'] = par_info['off_center'][2]
        par_info.pop('off_center')
    except KeyError:
        pass

    try:
        general_info['FOV AP'] = par_info['fov'][0]
        general_info['FOV FH'] = par_info['fov'][1]
        general_info['FOV RL'] = par_info['fov'][2]
        par_info.pop('fov')
    except KeyError:
        pass

    try:
        general_info['Scan Resolution X'] = par_info['scan_resolution'][0]
        general_info['Scan Resolution Y'] = par_info['scan_resolution'][1]
        par_info.pop('scan_resolution')
    except KeyError:
        pass

    # copy any unrecognized keys as is.
    # known keys found in XML PRIDE_V5, but not in PAR v4.2 are:
    #    'Samples Per Pixel' and 'Image Planar Configuration'
    general_info.update(par_info)

    return general_info

# dictionary mapping fieldnames in the XML header to the corresponding names
# in a PAR V4.2 file
image_def_XML_to_PAR = {
    'Image Type Ed Es': 'image_type_ed_es',
    'No Averages': 'number of averages',
    'Max RR Interval': 'maximum RR-interval',
    'Type': 'image_type_mr',
    'Display Orientation': 'image_display_orientation',
    'Image Flip Angle': 'image_flip_angle',
    'Rescale Slope': 'rescale slope',
    'Label Type': 'label type',
    'fMRI Status Indication': 'fmri_status_indication',
    'TURBO Factor': 'TURBO factor',
    'Min RR Interval': 'minimum RR-interval',
    'Scale Slope': 'scale slope',
    'Inversion Delay': 'Inversion delay',
    'Window Width': 'window width',
    'Sequence': 'scanning sequence',
    'Diffusion Anisotropy Type': 'diffusion anisotropy type',
    'Index': 'index in REC file',
    'Rescale Intercept': 'rescale intercept',
    'Diffusion B Factor': 'diffusion_b_factor',
    'Trigger Time': 'trigger_time',
    'Echo': 'echo number',
    'Echo Time': 'echo_time',
    'Pixel Spacing': 'pixel spacing',
    'Slice Gap': 'slice gap',
    'Dyn Scan Begin Time': 'dyn_scan_begin_time',
    'Window Center': 'window center',
    'Contrast Type': 'contrast type',
    'Slice': 'slice number',
    'BValue': 'diffusion b value number',
    'Scan Percentage': 'scan percentage',
    'Phase': 'cardiac phase number',
    'Slice Thickness': 'slice thickness',
    'Slice Orientation': 'slice orientation',
    'Dynamic': 'dynamic scan number',
    'Pixel Size': 'image pixel size',
    'Grad Orient': 'gradient orientation number',
    'Cardiac Frequency': 'cardiac frequency'}

# copy of enums_dict but with key names converted to their PAR equivalents
enums_dict_PAR = {image_def_XML_to_PAR[k]: v for k, v in enums_dict.items()}


# TODO?: The following values have different names in the XML vs. PAR header
rename_XML_to_PAR = {
    'HFS': 'Head First Supine',
    'LR': 'Left-Right',
    'RL': 'Right-Left',
    'AP': 'Anterior-Posterior',
    'PA': 'Posterior-Anterior',
    'N': 0,
    'Y': 1,
}


def _process_gen_dict_XML(xml_root):
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
            if entry_type in ['S16', 'S32', 'S128']:
                entry_type = str
            if 'ArraySize' in a:
                val = [entry_type(i) for i in e.text.strip().split()]
            else:
                val = entry_type(e.text)
            general_info[a['Name']] = val
    return general_info


def _get_image_def_attributes(xml_root, dtype_format='xml'):
    """Get names and dtypes for all attributes defined for each image.

    called by _process_image_lines_xml

    Paramters
    ---------
    xml_root :
        TODO
    dtype_format : {'xml', 'par'}
        If 'par'', XML paramter names and dtypes are converted to their PAR
        equivalents.

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

    dtype_format = dtype_format.lower()
    if dtype_format == 'xml':
        # xml_type_dict keeps enums as their string representation
        key_attributes = [
            (k.get('Name'), _get_type(k.get('Type'), xml_type_dict))
            for k in img_keys]
    elif dtype_format == 'par':
        # par_type_dict converts enums to int as used in PAR/REC
        key_attributes = [
            (image_def_XML_to_PAR[k.get('Name')],
             _get_type(k.get('Type'), par_type_dict))
            for k in img_keys]
    else:
        raise XMLRECError("dtype_format must be 'par' or 'xml'")

    # Process other attributes that are not considered image keys
    other_attributes = []
    for element in first_def:
        a = element.attrib
        if 'Name' in a:
            # if a['Type'] == 'Enumeration':
            #     enum_type = a['EnumType']
            #     print("enum_type = {}".format(enum_type))
            name = a['Name']
            if dtype_format == 'xml':
                entry_type = _get_type(a['Type'], xml_type_dict)
            else:
                entry_type = _get_type(a['Type'], par_type_dict)
                if name in image_def_XML_to_PAR:
                    name = image_def_XML_to_PAR[name]
            if 'ArraySize' in a:
                # handle vector entries (e.g. 'Pixel Size' is length 2)
                entry = (name, entry_type, int(a['ArraySize']))
            else:
                entry = (name, entry_type)
            other_attributes.append(entry)

    return key_attributes, other_attributes


# these keys are elements of a multi-valued key in the PAR/REC image_defs
composite_PAR_keys = ['Diffusion AP', 'Diffusion FH', 'Diffusion RL',
                      'Angulation AP', 'Angulation FH', 'Angulation RL',
                      'Offcenter AP', 'Offcenter FH', 'Offcenter RL',
                      'Resolution X', 'Resolution Y']


def _composite_attributes_xml_to_par(other_attributes):
    """utility used in conversion from XML-style to PAR-style image_defs.

    called by _process_image_lines_xml
    """
    if ('Diffusion AP', np.float32) in other_attributes:
        other_attributes.remove(('Diffusion AP', np.float32))
        other_attributes.remove(('Diffusion FH', np.float32))
        other_attributes.remove(('Diffusion RL', np.float32))
        other_attributes.append(('diffusion', float, (3, )))

    if ('Angulation AP', np.float64) in other_attributes:
        other_attributes.remove(('Angulation AP', np.float64))
        other_attributes.remove(('Angulation FH', np.float64))
        other_attributes.remove(('Angulation RL', np.float64))
        other_attributes.append(('image angulation', float, (3, )))

    if ('Offcenter AP', np.float64) in other_attributes:
        other_attributes.remove(('Offcenter AP', np.float64))
        other_attributes.remove(('Offcenter FH', np.float64))
        other_attributes.remove(('Offcenter RL', np.float64))
        other_attributes.append(('image offcentre', float, (3, )))

    if ('Resolution X', np.uint16) in other_attributes:
        other_attributes.remove(('Resolution X', np.uint16))
        other_attributes.remove(('Resolution Y', np.uint16))
        other_attributes.append(('recon resolution', int, (2, )))

    return other_attributes


# TODO: remove? this one is currently unused, but perhaps useful for testing
def _composite_attributes_par_to_xml(other_attributes):
    if ('diffusion', float, (3, )) in other_attributes:
        other_attributes.append(('Diffusion AP', np.float32))
        other_attributes.append(('Diffusion FH', np.float32))
        other_attributes.append(('Diffusion RL', np.float32))
        other_attributes.remove(('diffusion', float, (3, )))

    if ('image angulation', float, (3, )) in other_attributes:
        other_attributes.append(('Angulation AP', np.float64))
        other_attributes.append(('Angulation FH', np.float64))
        other_attributes.append(('Angulation RL', np.float64))
        other_attributes.remove(('image angulation', float, (3, )))

    if ('image offcentre', float, (3, )) in other_attributes:
        other_attributes.append(('Offcenter AP', np.float64))
        other_attributes.append(('Offcenter FH', np.float64))
        other_attributes.append(('Offcenter RL', np.float64))
        other_attributes.remove(('image offcentre', float, (3, )))

    if ('recon resolution', int, (2, )) in other_attributes:
        other_attributes.append(('Resolution X', np.uint16))
        other_attributes.append(('Resolution Y', np.uint16))
        other_attributes.remove(('recon resolution', int, (2, )))
    return other_attributes


def _get_composite_key_index(key):
    """utility used in conversion from XML-style to PAR-style image_defs.

    called by _process_image_lines_xml
    """
    if 'Diffusion' in key:
        name = 'diffusion'
    elif 'Angulation' in key:
        name = 'image angulation'
    elif 'Offcenter' in key:
        name = 'image offcentre'
    elif 'Resolution' in key:
        name = 'recon resolution'
    else:
        raise ValueError("unrecognized composite element name: {}".format(key))

    if key in ['Diffusion AP', 'Angulation AP', 'Offcenter AP',
               'Resolution X']:
        index = 0
    elif key in ['Diffusion FH', 'Angulation FH', 'Offcenter FH',
                 'Resolution Y']:
        index = 1
    elif key in ['Diffusion RL', 'Angulation RL', 'Offcenter RL']:
        index = 2
    else:
        raise ValueError("unrecognized composite element name: {}".format(key))
    return (name, index)


def _process_image_lines_xml(xml_root, dtype_format='xml'):
    """Build image_defs by parsing the XML file.

    Parameters
    ----------
    xml_root :
        TODO
    dtype_format : {'xml', 'par'}
        If 'par'', XML paramter names and dtypes are converted to their PAR file
        equivalents.
    """
    image_defs_array = xml_root.find('Image_Array')
    if image_defs_array is None:
        raise RuntimeError("No 'Image_Array' found in the XML file")

    key_attributes, other_attributes = _get_image_def_attributes(xml_root, dtype_format=dtype_format)

    image_def_dtd = key_attributes + other_attributes
    # dtype dict based on the XML attribute names
    dtype_dict = {a[0]: a[1] for a in image_def_dtd}

    if dtype_format == 'par':
        # image_defs based on conversion of types in composite_PAR_keys
        image_def_dtd = _composite_attributes_xml_to_par(image_def_dtd)

    def _get_val(entry_dtype, text):
        if entry_dtype == '|S16':
            val = text[:16]
        elif entry_dtype == '|S32':
            val = text[:32]
        elif entry_dtype == '|S128':
            val = text[:128]
        else:
            val = entry_dtype(text)
        return val

    image_defs = np.zeros(len(image_defs_array), dtype=image_def_dtd)
    already_warned = []
    for i, image_def in enumerate(image_defs_array):

        if image_def.find('Key') != image_def[0]:
            raise RuntimeError("Expected first element of image_def to be Key")

        key_def = image_def[0]
        for key in key_def:
            name = key.get('Name')
            if dtype_format == 'par' and name in image_def_XML_to_PAR:
                name = image_def_XML_to_PAR[name]
            if dtype_format == 'par' and name in enums_dict_PAR:
                # string -> int
                if key.text in enums_dict_PAR[name]:
                    val = enums_dict_PAR[name][key.text]
                else:
                    if (name, key.text) not in already_warned:
                        warnings.warn(
                            ("Unknown enumerated value for {} with name {}. "
                             "Setting the value to -1.  Please contact the "
                             "nibabel developers about adding support for "
                             "this to the XML/REC reader.").format(
                                name, key.text))
                        val = -1
                        # avoid repeated warnings for this enum
                        already_warned.append((name, key.text))
            else:
                val = key.text
            image_defs[name][i] = _get_val(dtype_dict[name], val)

        # for all image properties we know about
        for element in image_def[1:]:
            a = element.attrib
            text = element.text
            if 'Name' in a:
                name = a['Name']
                if dtype_format == 'par' and name in image_def_XML_to_PAR:
                    name = image_def_XML_to_PAR[name]
                    # composite_PAR_keys
                entry_dtype = dtype_dict[name]
                if 'ArraySize' in a:
                    val = [entry_dtype(i) for i in text.strip().split()]
                else:
                    if dtype_format == 'par' and name in enums_dict_PAR:
                        # string -> int
                        if text in enums_dict_PAR[name]:
                            val = enums_dict_PAR[name][text]
                        else:
                            if (name, text) not in already_warned:
                                warnings.warn(
                                    ("Unknown enumerated value for {} with "
                                     "name {}. Setting the value to -1.  "
                                     "Please contact the nibabel developers "
                                     "about adding support for this to the "
                                     "XML/REC reader.").format(name, text))
                                val = -1
                                # avoid repeated warnings for this enum
                                already_warned.append((name, text))
                    else:
                        val = _get_val(entry_dtype, text)
                if dtype_format == 'par' and name in composite_PAR_keys:
                    # conversion of types in composite_PAR_keys
                    name, vec_index = _get_composite_key_index(name)
                    image_defs[name][i][vec_index] = val
                else:
                    image_defs[name][i] = val
    return image_defs


def parse_XML_header(fobj, dtype_format='xml'):
    """Parse a XML header and aggregate all information into useful containers.

    Parameters
    ----------
    fobj : file-object or str
        The XML header file object or file name.
    dtype_format : {'xml', 'par'}
        If 'par' the image_defs will be converted to a format matching that
        found in PARRECHeader.

    Returns
    -------
    general_info : dict
        Contains all "General Information" from the header file
    image_info : ndarray
        Structured array with fields giving all "Image information" in the
        header
    """
    # single pass through the header
    tree = ET.parse(fobj)
    root = tree.getroot()

    # _split_header() equivalent

    version = root.tag  # e.g. PRIDE_V5
    if version not in supported_xml_versions:
        warnings.warn(one_line(
            """XML version '{0}' is currently not supported.  Only PRIDE_V5 XML
            files have been tested. --making an attempt to read nevertheless.
            Please email the NiBabel mailing list, if you are interested in
            adding support for this version.
            """.format(version)))
    try:
        general_info = _process_gen_dict_XML(root)
        image_defs = _process_image_lines_xml(
            root, dtype_format=dtype_format)
    except ET.ParseError:
            raise XMLRECError(
                "A ParseError occured in the ElementTree library while "
                "reading the XML file. This may be due to a truncated XML "
                "file.")

    return general_info, image_defs


class XMLRECHeader(PARRECHeader):
    @classmethod
    def from_fileobj(klass, fileobj, permit_truncated=False,
                     strict_sort=False):
        info, image_defs = parse_XML_header(fileobj, dtype_format='par')
        # convert to PAR/REC format general_info
        info = general_info_xml_to_par(info)
        return klass(info, image_defs, permit_truncated, strict_sort)
    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise XMLRECError('Cannot create XMLRECHeader from air.')
        if type(header) == klass:
            return header.copy()
        raise XMLRECError('Cannot create XMLREC header from '
                          'non-XMLREC header.')
    def copy(self):
        return XMLRECHeader(deepcopy(self.general_info),
                            self.image_defs.copy(),
                            self.permit_truncated,
                            self.strict_sort)

class XMLRECImage(PARRECImage):
    """XML/REC image"""
    header_class = XMLRECHeader
    valid_exts = ('.REC', '.xml')
    files_types = (('image', '.REC'), ('header', '.xml'))

load = XMLRECImage.load

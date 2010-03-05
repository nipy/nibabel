import numpy as np

from nibabel.spatialimages import SpatialImage
from nibabel.eulerangles import euler2mat

# assign props to PAR header entries
# values are: (shortname[, dtype[, shape]])
_hdr_key_dict = {
    'Patient name': ('patient_name',),
    'Examination name': ('exam_name',),
    'Protocol name': ('protocol_name',),
    'Examination date/time': ('exam_date',),
    'Series Type': ('series_type',),
    'Acquisition nr': ('acq_nr', int),
    'Reconstruction nr': ('recon_nr', int),
    'Scan Duration [sec]': ('scan_duration', int),
    'Max. number of cardiac phases': ('max_cardiac_phases', int),
    'Max. number of echoes': ('max_echoes', int),
    'Max. number of slices/locations': ('max_slices', int),
    'Max. number of dynamics': ('max_dynamics', int),
    'Max. number of mixes': ('max_mixes', int),
    'Patient position': ('patient_position',),
    'Preparation direction': ('prep_direction',),
    'Technique': ('tech',),
    'Scan resolution  (x, y)': ('scan_resolution', int, (2,)),
    'Scan mode': ('san_mode',),
    'Repetition time [ms]': ('repetition_time', float),
    'FOV (ap,fh,rl) [mm]': ('fov', float, (3,)),
    'Water Fat shift [pixels]': ('water_fat_shift', float),
    'Angulation midslice(ap,fh,rl)[degr]': ('angulation', float, (3,)),
    'Off Centre midslice(ap,fh,rl) [mm]': ('off_center', float, (3,)),
    'Flow compensation <0=no 1=yes> ?': ('flow_compensation', int),
    'Presaturation     <0=no 1=yes> ?': ('presaturation', int),
    'Phase encoding velocity [cm/sec]': ('phase_enc_velocity', float, (3,)),
    'MTC               <0=no 1=yes> ?': ('mtc', int),
    'SPIR              <0=no 1=yes> ?': ('spir', int),
    'EPI factor        <0,1=no EPI>': ('epi_factor', int),
    'Dynamic scan      <0=no 1=yes> ?': ('dyn_scan', int),
    'Diffusion         <0=no 1=yes> ?': ('diffusion', int),
    'Diffusion echo time [ms]': ('diffusion_echo_time', float),
    'Max. number of diffusion values': ('max_diffusion_values', int),
    'Max. number of gradient orients': ('max_gradient_orient', int),
    'Number of label types   <0=no ASL>': ('nr_label_types', int),
    }


# header items order per image definition line
# values are: (shortname[, dtype[, shape]])
_image_props_list = [
    ('slice number', int),
    ('echo number', int,),
    ('dynamic scan number', int,),
    ('cardiac phase number', int,),
    ('image_type_mr', int,),
    ('scanning sequence', int,),
    ('index in REC file', int,),
    ('image pixel size', int,),
    ('scan percentage', int,),
    ('recon resolution', int, (2,)),
    ('rescale intercept', float),
    ('rescale slope', float),
    ('scale slope', float),
    ('window center', int,),
    ('window width', int,),
    ('image angulation', float, (3,)),
    ('image offcentre', float, (3,)),
    ('slice thickness', float),
    ('slice gap', float),
    ('image_display_orientation', int,),
    ('slice orientation', int,),
    ('fmri_status_indication', int,),
    ('image_type_ed_es', int,),
    ('pixel spacing', float, (2,)),
    ('echo_time', float),
    ('dyn_scan_begin_time', float),
    ('trigger_time', float),
    ('diffusion_b_factor', float),
    ('number of averages', int,),
    ('image_flip_angle', float),
    ('cardiac frequency', int,),
    ('minimum RR-interval', int,),
    ('maximum RR-interval', int,), 
    ('TURBO factor', int,),
    ('Inversion delay', float),
    ('diffusion b value number', int,),         # (imagekey!)
    ('gradient orientation number', int,),      # (imagekey!)
    ('contrast type',),
    ('diffusion anisotropy type',),
    ('diffusion', float, (3,)),
    ('label type', int,),                       # (imagekey!)
    ]


_slice_orientation_codes = {
    1: 'transversal',
    2: 'sagital',
    3: 'coronal'
    }

class PARRECError(Exception):
    pass


class PARFile(object):
    def __init__(self, fobj):
        hdr = {}
        image_def = dict(zip([p[0] for p in _image_props_list],
                             [[] for i in xrange(len(_image_props_list))]))
        # compute the number of props that must appear on each line
        # given the header layout defined above
        must_have_items_nr = 0
        for p in _image_props_list:
            if len(p) < 3:
                must_have_items_nr += 1
            else:
                must_have_items_nr += np.prod(p[2])
        for line in fobj:
            if line.startswith('#'):
                continue
            elif line.startswith('.'):
                # read 'general information'
                first_colon = line[1:].find(':') + 1
                key = line[1:first_colon].strip()
                value = line[first_colon + 1:].strip()
                # get props for this hdr field
                props = _hdr_key_dict[key]
                # turn values into meaningful dtype
                if len(props) == 2:
                    # only dtype spec and no shape
                    value = props[1](value)
                elif len(props) == 3:
                    # array with dtype and shape
                    value = np.fromstring(value, props[1], sep=' ')
                    value.shape = props[2]
                hdr[props[0]] = value
            elif line.strip():
                # read image definition
                items = line.split()
                if len(items) != must_have_items_nr:
                    raise PARRECError(
                        "Unexpected number of properties per image definition. "
                        "Got: %i, expected: %i"
                        % (len(items), len(_image_props_list)))
                item_counter = 0
                for i, props in enumerate(_image_props_list):
                    if len(props) == 1:
                        # simple string
                        image_def[props[0]].append(items[item_counter])
                        item_counter += 1
                    elif len(props) == 2:
                        # prop with dtype
                        image_def[props[0]].append(props[1](items[item_counter]))
                        item_counter += 1
                    elif len(props) == 3:
                        nelements = np.prod(props[2])
                        # array prop with dtype
                        # get as many elements as necessary
                        itms = items[item_counter:item_counter+nelements]
                        # convert to array with dtype
                        value = np.fromstring(" ".join(itms), props[1], sep=' ')
                        # store
                        image_def[props[0]].append(value)
                        item_counter += nelements

        # postproc image def props
        for key in image_def:
            image_def[key] = np.array(image_def[key])

        self._hdr_defs = hdr
        self._image_defs = image_def


    def _get_unqiue_image_prop(self, name):
        prop = self._image_defs[name]
        if len(prop.shape) > 1:
            uprops = [np.unique(prop[i]) for i in range(len(prop.shape))]
        else:
            uprops = [np.unique(prop)]
        if not np.prod([len(uprop) for uprop in uprops]) == 1:
            raise PARRECError('Varying %s in image sequence (%s). This is not '
                              'suppported.' % (name, uprops))
        else:
            return np.array([uprop[0] for uprop in uprops])


    def get_affine(self):
        """Just considers global rotation and offset"""
        # hdr has deg, we need radian
        # order is [ap, fh, rl]
        #           x   y   z
        ang_rad = self._hdr_defs['angulation'] * np.pi / 180.0

        # slice orientation for the whole image series
        slice_orientation = _slice_orientation_codes[
                    self._get_unqiue_image_prop('slice orientation')[0]]

        # R2AGUI approach is this, but it comes with remarks ;-)
        # % trying to incorporate AP FH RL rotation angles: determined using some 
        # % common sense, Chris Rordon's help + source code and trial and error, 
        # % this is considered EXPERIMENTAL!
        rot_rl = np.mat(
                [[1.0, 0.0, 0.0],
                 [0.0, np.cos(ang_rad[2]), -np.sin(ang_rad[2])],
                 [0.0, np.sin(ang_rad[2]), np.cos(ang_rad[2])]]
                )
        rot_ap = np.mat(
                [[np.cos(ang_rad[0]), 0.0, np.sin(ang_rad[0])],
                 [0.0, 1.0, 0.0],
                 [-np.sin(ang_rad[0]), 0.0, np.cos(ang_rad[0])]]
                )
        rot_fh = np.mat(
                [[np.cos(ang_rad[1]), -np.sin(ang_rad[1]), 0.0],
                 [np.sin(ang_rad[1]), np.cos(ang_rad[1]), 0.0],
                 [0.0, 0.0, 1.0]]
                )
        rot_r2agui = rot_rl * rot_ap * rot_fh
        # NiBabel way of doing it
        # order is [ap, fh, rl]
        #           x   y   z
        #           0   1   2
        rot_nibabel = euler2mat(ang_rad[1], ang_rad[0], ang_rad[2])

        # XXX for now put some safety net, until we have recorded proper
        # test data with oblique orientations and different readout directions
        # to verify the order of arguments of euler2mat
        assert(np.all(rot_r2agui == rot_nibabel))
        rot = rot_nibabel

        slice_thickness = self._get_unqiue_image_prop('slice thickness')[0]
        slice_gap = self._get_unqiue_image_prop('slice gap')[0]
        voxsize_inplane = self._get_unqiue_image_prop('pixel spacing')
        # voxel size (x, y, z)
        voxsize = np.ones(3)
        # come up with proper scaling
        if slice_orientation == 'sagital':
            voxsize[1:] = voxsize_inplane            # Y x Z
            voxsize[0] = slice_thickness + slice_gap # X
        elif slice_orientation == 'transversal':
            voxsize[:2] = voxsize_inplane            # X x Y
            voxsize[2] = slice_thickness + slice_gap # Z
        elif slice_orientation == 'coronal':
            voxsize[::2] = voxsize_inplane           # X x Z
            voxsize[1] = slice_thickness + slice_gap # Y
        else:
            raise PARRECError("Unknown slice orientation (%s).")

        # get the scaling by voxelsize and slice thickness (incl. gap)
        scaled = rot.T * np.mat(np.diag(voxsize))

        # rotation has order ap,fh,rl, that is y,z,x -> reorder into x,y,z
        # inplane is ap x fh
        flipit = np.mat([[0,0,1],[-1,0,0],[0,-1,0]])
        scaled = scaled * flipit
        # offset of FOV center to ijk origin
        # need to reorder fov spec into x,y,z!
        fov_origin = (self._hdr_defs['fov'][[2,0,1]] / 2.0) / voxsize
        # rotate into scanner space
        fov_origin_rot = fov_origin * scaled
        # get offset of rotated ijk origin from iso-center of the scanner
        # need to reorder offset spec into x,y,z!
        offset = fov_origin_rot - self._hdr_defs['off_center'][[2,0,1]]

        # XXX incomplete, ugly, wrong, drives me mad
        return scaled


class PARRECImage(SpatialImage):
    files_types = (('image', '.rec'), ('header', '.par'))

    @classmethod
    def from_file_map(klass, file_map):
        hdrfobj = file_map['header'].get_prepare_fileobj()
        hdr_file = PARFile(hdrfobj)
        print hdr_file.get_affine()
        return hdr_file
        #affine = minc_file.get_affine()
        #if affine.shape != (4, 4):
        #    raise MincError('Image does not have 3 spatial dimensions')
        #data_dtype = minc_file.get_data_dtype()
        #shape = minc_file.get_data_shape()
        #zooms = minc_file.get_zooms()
        #header = klass.header_class(data_dtype, shape, zooms)
        #data = klass.ImageArrayProxy(minc_file)
        #return  MincImage(data, affine, header, extra=None, file_map=file_map)


load = PARRECImage.load

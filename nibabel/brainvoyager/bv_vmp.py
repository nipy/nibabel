# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) VMP files.

for documentation on the file format see:
http://support.brainvoyager.com/installation-introduction/23-file-formats/377-users-guide-23-the-format-of-nr-vmp-files.html

Author: Thomas Emmerling
"""
from __future__ import division
from .bv import (BvFileHeader, BvFileImage, update_BV_header)
from ..spatialimages import HeaderDataError

VMP_HDR_DICT_PROTO = (
    ('magic_number', 'I', 2712847316),
    ('version', 'h', 6),
    ('document_type', 'h', 1),
    ('nr_of_submaps', 'i', 1),
    ('nr_of_timepoints', 'i', 0),
    ('nr_of_component_params', 'i', 0),
    ('show_params_range_from', 'i', 0),
    ('show_params_range_to', 'i', 0),
    ('use_for_fingerprint_params_range_from', 'i', 0),
    ('use_for_fingerprint_params_range_to', 'i', 0),
    ('x_start', 'i', 57),
    ('x_end', 'i', 231),
    ('y_start', 'i', 52),
    ('y_end', 'i', 172),
    ('z_start', 'i', 59),
    ('z_end', 'i', 197),
    ('resolution', 'i', 3),
    ('dim_x', 'i', 256),
    ('dim_y', 'i', 256),
    ('dim_z', 'i', 256),
    ('vtc_filename', 'z', b'<none>'),
    ('prt_filename', 'z', b''),
    ('voi_filename', 'z', b''),
    ('maps', (
        ('type_of_map', 'i', 1),
        ('map_threshold', 'f', 1.6500),
        ('upper_threshold', 'f', 8.0),
        ('map_name', 'z', b'New Map'),
        ('pos_min_r', 'B', 255),
        ('pos_min_g', 'B', 0),
        ('pos_min_b', 'B', 0),
        ('pos_max_r', 'B', 255),
        ('pos_max_g', 'B', 255),
        ('pos_max_b', 'B', 0),
        ('neg_min_r', 'B', 255),
        ('neg_min_g', 'B', 0),
        ('neg_min_b', 'B', 255),
        ('neg_max_r', 'B', 0),
        ('neg_max_g', 'B', 0),
        ('neg_max_b', 'B', 255),
        ('use_vmp_color', 'B', 0),
        ('lut_filename', 'z', b'<default>'),
        ('transparent_color_factor', 'f', 1.0),
        ('nr_of_lags', 'i', (0, 'type_of_map', 3)),
        ('display_min_lag', 'i', (0, 'type_of_map', 3)),
        ('display_max_lag', 'i', (0, 'type_of_map', 3)),
        ('show_correlation_or_lag', 'i', (0, 'type_of_map', 3)),
        ('cluster_size_threshold', 'i', 50),
        ('enable_cluster_size_threshold', 'B', 0),
        ('show_values_above_upper_threshold', 'i', 1),
        ('df1', 'i', 249),
        ('df2', 'i', 0),
        ('show_pos_neg_values', 'B', 3),
        ('nr_of_used_voxels', 'i', 45555),
        ('size_of_fdr_table', 'i', 0),
        ('fdr_table_info', (
            ('q', 'f', 0.0),
            ('crit_standard', 'f', 0.0),
            ('crit_conservative', 'f', 0.0),
        ), 'size_of_fdr_table'),
        ('use_fdr_table_index', 'i', 0),
    ), 'nr_of_submaps'),
    ('component_time_points', (
        ('timepoints', (('timepoint', 'f', 0.0),), 'nr_of_timepoints'),
    ), 'nr_of_submaps'),
    ('component_params', (
        ('param_name', 'z', b''),
        ('param_values', (('value', 'f', 0.0),), 'nr_of_submaps')
    ), 'nr_of_component_params')
)


class BvVmpHeader(BvFileHeader):
    ''' Class for BrainVoyager NR-VMP header
    '''

    # format defaults
    allowed_dtypes = [2]
    default_dtype = 2
    hdr_dict_proto = VMP_HDR_DICT_PROTO
    supported_fileversions = [6]

    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._hdr_dict
        # calculate dimensions
        z = (hdr['z_end'] - hdr['z_start']) / hdr['resolution']
        y = (hdr['y_end'] - hdr['y_start']) / hdr['resolution']
        x = (hdr['x_end'] - hdr['x_start']) / hdr['resolution']
        n = hdr['nr_of_submaps']
        return tuple(int(d) for d in [n, z, y, x])

    def set_data_shape(self, shape=None, zyx=None, n=None):
        ''' Set shape of data

        To conform with nibabel standards this implements shape.
        However, to fill the BvVmpHeader with sensible information use the
        zyx and the n parameter instead.

        Parameters
        ----------
        shape: sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list of integers, optional
           [[z_start,z_end],[y_start,y_end],[x_start,x_end]]
           array storing borders of data
        n: int, optional
           number of submaps

        '''
        hdr_dict_old = self._hdr_dict.copy()
        if (shape is None) and (zyx is None) and (n is None):
            raise HeaderDataError('Shape, zyx, or n needs to be specified!')

        if ((n is not None) and (n < 1)) or \
           ((shape is not None) and (shape[0] < 1)):
            raise HeaderDataError('NR-VMP files need at least one sub-map!')

        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from default coordinates.
            if len(shape) != 4:
                raise HeaderDataError(
                    'Shape for VMP files must be 4 dimensional (NZYX)!')
            self._hdr_dict['x_end'] = self._hdr_dict['x_start'] + \
                (shape[3] * self._hdr_dict['resolution'])
            self._hdr_dict['y_end'] = self._hdr_dict['y_start'] + \
                (shape[2] * self._hdr_dict['resolution'])
            self._hdr_dict['z_end'] = self._hdr_dict['z_start'] + \
                (shape[1] * self._hdr_dict['resolution'])
            self._hdr_dict['nr_of_submaps'] = int(shape[0])
            self._hdr_dict = update_BV_header(self.hdr_dict_proto,
                                              hdr_dict_old, self._hdr_dict)
            return
        if zyx is not None:
            self._hdr_dict['z_start'] = zyx[0][0]
            self._hdr_dict['z_end'] = zyx[0][1]
            self._hdr_dict['y_start'] = zyx[1][0]
            self._hdr_dict['y_end'] = zyx[1][1]
            self._hdr_dict['x_start'] = zyx[2][0]
            self._hdr_dict['x_end'] = zyx[2][1]
        if n is not None:
            self._hdr_dict['nr_of_submaps'] = int(n)
            self._hdr_dict = update_BV_header(self.hdr_dict_proto,
                                              hdr_dict_old, self._hdr_dict)

    @property
    def framing_cube(self):
        ''' Get the dimensions of the framing cube that constitutes
        the coordinate system boundaries for the bounding box.
        '''
        hdr = self._hdr_dict
        return hdr['dim_z'], hdr['dim_y'], hdr['dim_x']

    @framing_cube.setter
    def framing_cube(self, fc):
        ''' Set the dimensions of the framing cube that constitutes
        the coordinate system boundaries for the bounding box.

        For VMP files this puts the values also into the header.
        '''
        self._hdr_dict['dim_z'] = fc[0]
        self._hdr_dict['dim_y'] = fc[1]
        self._hdr_dict['dim_x'] = fc[2]
        self._framing_cube = fc


class BvVmpImage(BvFileImage):
    # Set the class of the corresponding header
    header_class = BvVmpHeader

    # Set the label ('image') and the extension ('.vmp') for a VMP file
    files_types = (('image', '.vmp'),)
    valid_exts = ('.vmp',)


load = BvVmpImage.load
save = BvVmpImage.instance_to_filename

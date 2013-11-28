# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) VMR files.

for documentation on the file format see:
http://support.brainvoyager.com/automation-aamp-development/23-file-formats/385-developer-guide-26-the-format-of-vmr-files.html

Author: Sabrina Fontanella and Thomas Emmerling
"""
from __future__ import division
from .bv import (BvError, BvFileHeader, BvFileImage, parse_BV_header,
                 pack_BV_header, calc_BV_header_size, combine_st, parse_st)
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report
import numpy as np


VMR_PRHDR_DICT_PROTO = (
    ('version', 'h', 4),
    ('dim_x', 'h', 256),
    ('dim_y', 'h', 256),
    ('dim_z', 'h', 256),
)

VMR_PSHDR_DICT_PROTO = (
    ('offset_x', 'h', 0),
    ('offset_y', 'h', 0),
    ('offset_z', 'h', 0),
    ('framing_cube', 'h', 256),
    ('pos_infos_verified', 'i', 0),
    ('coordinate_system_entry', 'i', 1),
    ('slice_first_center_x', 'f', 127.5),
    ('slice_first_center_y', 'f', 0.0),
    ('slice_first_center_z', 'f', 0.0),
    ('slice_last_center_x', 'f', -127.5),
    ('slice_last_center_y', 'f', 0.0),
    ('slice_last_center_z', 'f', 0.0),
    ('row_dir_x', 'f', 0.0),
    ('row_dir_y', 'f', 1.0),
    ('row_dir_z', 'f', 0.0),
    ('col_dir_x', 'f', 0.0),
    ('col_dir_y', 'f', 0.0),
    ('col_dir_z', 'f', -1.0),
    ('nr_rows', 'i', 256),
    ('nr_cols', 'i', 256),
    ('fov_row_dir', 'f', 256.0),
    ('fov_col_dir', 'f', 256.0),
    ('slice_thickness', 'f', 1.0),
    ('gap_thickness', 'f', 0.0),
    ('nr_of_past_spatial_trans', 'i', 0),
    ('past_spatial_trans', (
        ('name', 'z', b''),
        ('type', 'i', b''),
        ('source_file', 'z', b''),
        ('nr_of_trans_val', 'i', b''),
        ('trans_val', (('value', 'f', 0.0),), 'nr_of_trans_val')
    ), 'nr_of_past_spatial_trans'),
    ('lr_convention', 'B', 1),
    ('reference_space', 'B', 0),
    ('vox_res_x', 'f', 1.0),
    ('vox_res_y', 'f', 1.0),
    ('vox_res_z', 'f', 1.0),
    ('flag_vox_resolution', 'B', 0),
    ('flag_tal_space', 'B', 0),
    ('min_intensity', 'i', 0),
    ('mean_intensity', 'i', 127),
    ('max_intensity', 'i', 255),
)


def computeOffsetPostHDR(hdr_dict, fileobj):
    currentSeek = fileobj.tell()
    return currentSeek + (hdr_dict['dim_x'] * hdr_dict['dim_y'] *
                          hdr_dict['dim_z'])


def concatePrePos(preDict, posDict):
    temp = preDict.copy()
    temp.update(posDict)
    return temp


class BvVmrHeader(BvFileHeader):
    """Class for BrainVoyager VMR header."""

    # format defaults
    default_endianness = '<'
    allowed_dtypes = [3]
    default_dtype = 3
    hdr_dict_proto = VMR_PRHDR_DICT_PROTO + VMR_PSHDR_DICT_PROTO
    supported_fileversions = [4]

    def get_data_shape(self):
        hdr = self._hdr_dict
        # calculate dimensions
        z = hdr['dim_z']
        y = hdr['dim_y']
        x = hdr['dim_x']
        return tuple(int(d) for d in [z, y, x])

    def set_data_shape(self, shape):
        """Set shape of data.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        """
        if len(shape) != 3:
            raise HeaderDataError(
                'Shape for VMR files must be 3 dimensional (ZYX)!')
        self._hdr_dict['dim_x'] = shape[2]
        self._hdr_dict['dim_y'] = shape[1]
        self._hdr_dict['dim_z'] = shape[0]

    def set_data_offset(self, offset):
        """Set offset into data file to read data.

        The offset is always 8 for VMR files.
        """
        self._data_offset = 8

    def get_data_offset(self):
        """Return offset into data file to read data.

        The offset is always 8 for VMR files.
        """
        return 8

    def set_xflip(self, xflip):
        if xflip is True:
            self._hdr_dict['lr_convention'] = 1
        elif xflip is False:
            self._hdr_dict['lr_convention'] = 2
        else:
            self._hdr_dict['lr_convention'] = 0

    def get_xflip(self):
        xflip = int(self._hdr_dict['lr_convention'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')

    def get_base_affine(self):
        """Get affine from VMR header fields.

        Internal storage of the image is ZYXT, where (in patient coordiante/
        real world orientations):
        Z := axis increasing from right to left (R to L)
        Y := axis increasing from superior to inferior (S to I)
        X := axis increasing from anterior to posterior (A to P)
        T := volumes (if present in file format)
        """
        zooms = self.get_zooms()
        if not self.get_xflip():
            # make the BV internal Z axis neurological (left-is-left);
            # not default in BV files!
            zooms[0] *= -1

        # compute the rotation
        rot = np.zeros((3, 3))
        # make the flipped BV Z axis the new R axis
        rot[:, 0] = [-zooms[0], 0, 0]
        # make the flipped BV X axis the new A axis
        rot[:, 1] = [0, 0, -zooms[2]]
        # make the flipped BV Y axis the new S axis
        rot[:, 2] = [0, -zooms[1], 0]

        # compute the translation
        fcc = np.array(self.get_framing_cube()) / 2  # center of framing cube
        bbc = np.array(self.get_bbox_center())  # center of bounding box
        tra = np.dot((bbc - fcc), rot)

        # assemble
        M = np.eye(4, 4)
        M[0:3, 0:3] = rot
        M[0:3, 3] = tra.T

        # look for additional transformations in past_spatial_trans and combine
        # with M
        if self._hdr_dict['past_spatial_trans']:
            STarray = np.zeros((len(self._hdr_dict['past_spatial_trans']),
                               4, 4))
            for st in range(len(self._hdr_dict['past_spatial_trans'])):
                STarray[st, :, :] = \
                    parse_st(self._hdr_dict['past_spatial_trans'][st])
            combinedST = combine_st(STarray, inv=True)
            M = np.dot(M, combinedST)
        return M

    get_best_affine = get_base_affine

    get_default_affine = get_base_affine

    get_affine = get_base_affine

    @classmethod
    def from_fileobj(klass, fileobj, endianness=default_endianness,
                     check=True):
        hdr_dictPre = parse_BV_header(VMR_PRHDR_DICT_PROTO, fileobj)
        # calculate new seek for the post data header
        newSeek = computeOffsetPostHDR(hdr_dictPre, fileobj)
        fileobj.seek(newSeek)
        hdr_dictPos = parse_BV_header(VMR_PSHDR_DICT_PROTO, fileobj)
        hdr_dict = concatePrePos(hdr_dictPre, hdr_dictPos)
        # The offset is always 8 for VMR files.
        offset = 8
        return klass(hdr_dict, endianness, check, offset)

    def get_bbox_center(self):
        """Get the center coordinate of the bounding box.
           Not required for VMR files
        """
        return np.array([self.get_framing_cube() / 2 for d in range(3)])

    def get_zooms(self):
        return (self._hdr_dict['vox_res_z'], self._hdr_dict['vox_res_y'],
                self._hdr_dict['vox_res_x'])

    def set_zooms(self, zooms):
        self._hdr_dict['vox_res_z'] = float(zooms[0])
        self._hdr_dict['vox_res_y'] = float(zooms[1])
        self._hdr_dict['vox_res_x'] = float(zooms[2])

    def write_to(self, fileobj):
        binaryblock = pack_BV_header(self.hdr_dict_proto, self._hdr_dict)
        # calculate size of preDataHeader
        sizePrH = calc_BV_header_size(VMR_PRHDR_DICT_PROTO, self._hdr_dict)
        # write the preHeader
        fileobj.write(binaryblock[0:sizePrH])
        fileobj.seek(computeOffsetPostHDR(self._hdr_dict, fileobj))
        fileobj.write(binaryblock[sizePrH:])


class BvVmrImage(BvFileImage):
    """Class for BrainVoyager VMR images."""

    # Set the class of the corresponding header
    header_class = BvVmrHeader

    # Set the label ('image') and the extension ('.vtc') for a VMR file
    files_types = (('image', '.vmr'),)
    valid_exts = ('.vmr',)

load = BvVmrImage.load
save = BvVmrImage.instance_to_filename

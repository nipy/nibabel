# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) MSK files.

for documentation on the file format see:
http://www.brainvoyager.com/ubb/Forum8/HTML/000087.html

Author: Thomas Emmerling
"""
from __future__ import division
from .bv import BvFileHeader, BvFileImage
from ..spatialimages import HeaderDataError

MSK_HDR_DICT_PROTO = (
    ('resolution', 'h', 3),
    ('x_start', 'h', 57),
    ('x_end', 'h', 231),
    ('y_start', 'h', 52),
    ('y_end', 'h', 172),
    ('z_start', 'h', 59),
    ('z_end', 'h', 197),
)


class BvMskHeader(BvFileHeader):
    """Class for BrainVoyager MSK header."""

    # format defaults
    allowed_dtypes = [3]
    default_dtype = 3
    hdr_dict_proto = MSK_HDR_DICT_PROTO

    def get_data_shape(self):
        """Get shape of data."""
        hdr = self._hdr_dict
        # calculate dimensions
        z = (hdr['z_end'] -
             hdr['z_start']) / hdr['resolution']
        y = (hdr['y_end'] -
             hdr['y_start']) / hdr['resolution']
        x = (hdr['x_end'] -
             hdr['x_start']) / hdr['resolution']

        return tuple(int(d) for d in [z, y, x])

    def set_data_shape(self, shape=None, zyx=None):
        """Set shape of data.

        To conform with nibabel standards this implements shape.
        However, to fill the VtcHeader with sensible information use
        the zyxt parameter instead.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list of integers, optional
           [[z_start,z_end],[y_start,y_end],[x_start,x_end]]
           array storing borders of data
        """
        if (shape is None) and (zyx is None):
            raise HeaderDataError('Shape or zyx needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from standard coordinates.
            if len(shape) != 3:
                raise HeaderDataError('Shape for MSK files must be\
                                      3 dimensional (ZYX)!')
            self._hdr_dict['x_end'] = self._hdr_dict['x_start'] + \
                (shape[2] * self._hdr_dict['resolution'])
            self._hdr_dict['y_end'] = self._hdr_dict['y_start'] + \
                (shape[1] * self._hdr_dict['resolution'])
            self._hdr_dict['z_end'] = self._hdr_dict['z_start'] + \
                (shape[0] * self._hdr_dict['resolution'])
            return
        self._hdr_dict['z_start'] = zyx[0][0]
        self._hdr_dict['z_end'] = zyx[0][1]
        self._hdr_dict['y_start'] = zyx[1][0]
        self._hdr_dict['y_end'] = zyx[1][1]
        self._hdr_dict['x_start'] = zyx[2][0]
        self._hdr_dict['x_end'] = zyx[2][1]


class BvMskImage(BvFileImage):
    """Class for BrainVoyager MSK masks.

    MSK files are technically binary images
    """

    # Set the class of the corresponding header
    header_class = BvMskHeader

    # Set the label ('image') and the extension ('.msk') for a MSK file
    files_types = (('image', '.msk'),)
    valid_exts = ('.msk',)
    _compressed_suffixes = ()

load = BvMskImage.load
save = BvMskImage.instance_to_filename

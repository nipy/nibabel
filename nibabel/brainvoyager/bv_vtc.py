# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) VTC files.

for documentation on the file format see:
http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html

Author: Thomas Emmerling
"""
from __future__ import division
from .bv import BvError, BvFileHeader, BvFileImage
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report

VTC_HDR_DICT_PROTO = (
    ('version', 'h', 3),
    ('fmr', 'z', b''),
    ('nr_prts', 'h', 0),
    ('prts', (('filename', 'z', b''),), 'nr_prts'),
    ('current_prt', 'h', 0),
    ('datatype', 'h', 2),
    ('volumes', 'h', 0),
    ('resolution', 'h', 3),
    ('x_start', 'h', 57),
    ('x_end', 'h', 231),
    ('y_start', 'h', 52),
    ('y_end', 'h', 172),
    ('z_start', 'h', 59),
    ('z_end', 'h', 197),
    ('lr_convention', 'b', 1),
    ('ref_space', 'b', 3),
    ('tr', 'f', 2000.0),
)


class BvVtcHeader(BvFileHeader):
    """Header for Brainvoyager (BV) VTC files.

    For documentation on the file format see:
    http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html
    """

    """
    Header for Brainvoyager (BV) VTC files.

    For documentation on the file format see:
    http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html
    """

    # format defaults
    allowed_dtypes = [1, 2]
    default_dtype = 2
    hdr_dict_proto = VTC_HDR_DICT_PROTO
    supported_fileversions = [3]

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
        t = hdr['volumes']
        return tuple(int(d) for d in [z, y, x, t])

    def set_data_shape(self, shape=None, zyx=None, t=None):
        """Set shape of data.

        To conform with nibabel standards this implements shape.
        However, to fill the BvVtcHeader with sensible information
        use the zyx and the t parameter instead.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list of integers, optional
           [[z_start,z_end],[y_start,y_end],[x_start,x_end]]
           array storing borders of data
        t: int
           number of volumes
        """
        if (shape is None) and (zyx is None) and (t is None):
            raise HeaderDataError('Shape, zyx, or t needs to be specified!')
        if ((t is not None) and (t < 0)) or \
           ((shape is not None) and (shape[3] < 0)):
            raise HeaderDataError('VTC files need at least one volume!')
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from standard coordinates.
            if len(shape) != 4:
                raise HeaderDataError(
                    'Shape for VTC files must be 4 dimensional (ZYXT)!')
            self._hdr_dict['x_end'] = \
                self._hdr_dict['x_start'] + \
                (shape[2] * self._hdr_dict['resolution'])
            self._hdr_dict['y_end'] = \
                self._hdr_dict['y_start'] + \
                (shape[1] * self._hdr_dict['resolution'])
            self._hdr_dict['z_end'] = \
                self._hdr_dict['z_start'] + \
                (shape[0] * self._hdr_dict['resolution'])
            self._hdr_dict['volumes'] = shape[3]
            return
        if zyx is not None:
            self._hdr_dict['z_start'] = zyx[0][0]
            self._hdr_dict['z_end'] = zyx[0][1]
            self._hdr_dict['y_start'] = zyx[1][0]
            self._hdr_dict['y_end'] = zyx[1][1]
            self._hdr_dict['x_start'] = zyx[2][0]
            self._hdr_dict['x_end'] = zyx[2][1]
        if t is not None:
            self._hdr_dict['volumes'] = t

    def get_xflip(self):
        """Get xflip for data."""
        xflip = int(self._hdr_dict['lr_convention'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')

    def set_xflip(self, xflip):
        """Set xflip for data."""
        if xflip is True:
            self._hdr_dict['lr_convention'] = 1
        elif xflip is False:
            self._hdr_dict['lr_convention'] = 2
        else:
            self._hdr_dict['lr_convention'] = 0


class BvVtcImage(BvFileImage):
    """Class for BrainVoyager VTC images."""

    # Set the class of the corresponding header
    header_class = BvVtcHeader

    # Set the label ('image') and the extension ('.vtc') for a VTC file
    files_types = (('image', '.vtc'),)
    valid_exts = ('.vtc',)

load = BvVtcImage.load
save = BvVtcImage.instance_to_filename

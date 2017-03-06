# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for BrainVoyager file formats '''
from .test_bv_vtc import BVVTC_EXAMPLE_IMAGES, BVVTC_EXAMPLE_HDRS
from .test_bv_msk import BVMSK_EXAMPLE_IMAGES, BVMSK_EXAMPLE_HDRS
from .test_bv_vmp import BVVMP_EXAMPLE_IMAGES, BVVMP_EXAMPLE_HDRS
from .test_bv_vmr import BVVMR_EXAMPLE_IMAGES, BVVMR_EXAMPLE_HDRS

__all__ = ('BVVTC_EXAMPLE_IMAGES', 'BVMSK_EXAMPLE_IMAGES',
           'BVVMP_EXAMPLE_IMAGES', 'BVVMR_EXAMPLE_IMAGES')

# assemble example images and corresponding example headers for testing
BV_EXAMPLE_IMAGES = (BVVTC_EXAMPLE_IMAGES, BVMSK_EXAMPLE_IMAGES,
                     BVVMP_EXAMPLE_IMAGES, BVVMR_EXAMPLE_IMAGES)

BV_EXAMPLE_HDRS = (BVVTC_EXAMPLE_HDRS, BVMSK_EXAMPLE_HDRS,
                   BVVMP_EXAMPLE_HDRS, BVVMR_EXAMPLE_HDRS)

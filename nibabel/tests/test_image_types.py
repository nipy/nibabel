# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for is_image / is_header functions '''
from __future__ import division, print_function, absolute_import

import copy
from os.path import dirname, basename, join as pjoin

import numpy as np

from .. import (Nifti1Image, Nifti1Header, Nifti1Pair,
                Nifti2Image, Nifti2Header, Nifti2Pair,
                AnalyzeImage, AnalyzeHeader,
                Minc1Image, Minc2Image,
                Spm2AnalyzeImage, Spm99AnalyzeImage,
                MGHImage, all_image_classes)

from nose.tools import assert_true, assert_equal, assert_false, assert_raises

DATA_PATH = pjoin(dirname(__file__), 'data')



def test_analyze_detection():
    # Test detection of Analyze, Nifti1 and Nifti2
    # Algorithm is as described in loadsave:which_analyze_type
    def wat(hdr):
        all_analyze_header_klasses = [Nifti1Header, Nifti2Header,
                                      AnalyzeHeader]
        for klass in all_analyze_header_klasses:
            try:
                if klass.is_header(hdr.binaryblock):
                   return klass
                else:
                    print('checked completed, but failed.')
            except ValueError as ve:
                print(ve)
                continue
        return None
        # return nils.which_analyze_type(hdr.binaryblock)

    n1_hdr = Nifti1Header(b'\0' * 348, check=False)
    n2_hdr = Nifti2Header(b'\0' * 540, check=False)
    assert_equal(wat(n1_hdr), None)

    n1_hdr['sizeof_hdr'] = 540
    n2_hdr['sizeof_hdr'] = 540
    assert_equal(wat(n1_hdr), None)
    assert_equal(wat(n1_hdr.as_byteswapped()), None)
    assert_equal(wat(n2_hdr), Nifti2Header)
    assert_equal(wat(n2_hdr.as_byteswapped()), Nifti2Header)

    n1_hdr['sizeof_hdr'] = 348
    assert_equal(wat(n1_hdr), AnalyzeHeader)
    assert_equal(wat(n1_hdr.as_byteswapped()), AnalyzeHeader)

    n1_hdr['magic'] = b'n+1'
    assert_equal(wat(n1_hdr), Nifti1Header)
    assert_equal(wat(n1_hdr.as_byteswapped()), Nifti1Header)

    n1_hdr['magic'] = b'ni1'
    assert_equal(wat(n1_hdr), Nifti1Header)
    assert_equal(wat(n1_hdr.as_byteswapped()), Nifti1Header)

    # Doesn't matter what magic is if it's not a nifti1 magic
    n1_hdr['magic'] = b'ni2'
    assert_equal(wat(n1_hdr), AnalyzeHeader)

    n1_hdr['sizeof_hdr'] = 0
    n1_hdr['magic'] = b''
    assert_equal(wat(n1_hdr), None)

    n1_hdr['magic'] = 'n+1'
    assert_equal(wat(n1_hdr), Nifti1Header)

    n1_hdr['magic'] = 'ni1'
    assert_equal(wat(n1_hdr), Nifti1Header)


def test_sniff_and_guessed_image_type(img_klasses=all_image_classes):
    """
    Loop over all test cases:
      * whether a sniff is provided or not
      * randomizing the order of image classes
      * over all known image types

    For each, we expect:
       * When the file matches the expected class, things should
            either work, or fail if we're doing bad stuff.
       * When the file is a mismatch, the functions should not throw.
    """

    def test_image_class(img_path, expected_img_klass):
        """ Embedded function to compare an image of one image class to all others.

        The function should make sure that it loads the image with the expected class,
        but failing when given a bad sniff (when the sniff is used)."""

        def check_img(img_path, img_klass, sniff_mode, sniff, expect_success, msg):
            """Embedded function to do the actual checks expected."""

            if sniff_mode == 'no_sniff':
                # Don't pass any sniff--not even "None"
                is_img, new_sniff = img_klass.is_image(img_path)
            else:
                # Pass a sniff, but don't reuse across images.
                is_img, new_sniff = img_klass.is_image(img_path, sniff)

            if expect_success:
                # Check that the sniff returned is appropriate.
                new_msg = '%s returned sniff==None (%s)' % (img_klass.__name__, msg)
                expected_sizeof_hdr = getattr(img_klass.header_class, 'sizeof_hdr', 0)
                current_sizeof_hdr = len(new_sniff) if new_sniff is not None else 0
                assert_true(current_sizeof_hdr >= expected_sizeof_hdr, new_msg)

                # Check that the image type was recognized.
                new_msg = '%s (%s) image is%s a %s image.' % (
                    basename(img_path),
                    msg,
                    '' if is_img else ' not',
                    img_klass.__name__)
                assert_true(is_img, new_msg)

            if sniff_mode == 'vanilla':
                return new_sniff
            else:
                return sniff

        sizeof_hdr = getattr(expected_img_klass.header_class, 'sizeof_hdr', 0)

        for sniff_mode, sniff in dict(
                vanilla=None,  # use the sniff of the previous item
                no_sniff=None,  # Don't pass a sniff
                none=None,  # pass None as the sniff, should query in fn
                empty='',  # pass an empty sniff, should query in fn
                irrelevant='a' * (sizeof_hdr - 1),  # A too-small sniff, query
                bad_sniff='a' * sizeof_hdr).items():  # Bad sniff, should fail.

            for klass in img_klasses:
                if klass == expected_img_klass:
                    # Class will load unless you pass a bad sniff,
                    #   the header actually uses the sniff, and the
                    #   sniff check is actually something meaningful
                    #   (we're looking at you, Minc1Header...)
                    expect_success = (sniff_mode != 'bad_sniff' or
                                      sizeof_hdr == 0 or
                                      klass == Minc1Image)  # special case...
                else:
                    expect_success = False  # Not sure the relationships

                # Reuse the sniff... but it will only change for some
                # sniff_mode values.
                msg = '%s/ %s/ %s' % (expected_img_klass.__name__, sniff_mode,
                                      str(expect_success))
                sniff = check_img(img_path, klass, sniff_mode=sniff_mode,
                                  sniff=sniff, expect_success=expect_success,
                                  msg=msg)

    # Test whether we can guess the image type from example files
    for img_filename, image_klass in [('example4d.nii.gz', Nifti1Image),
                                      ('nifti1.hdr', Nifti1Pair),
                                      ('example_nifti2.nii.gz', Nifti2Image),
                                      ('nifti2.hdr', Nifti2Pair),
                                      ('tiny.mnc', Minc1Image),
                                      ('small.mnc', Minc2Image),
                                      ('test.mgz', MGHImage),
                                      ('analyze.hdr', Spm2AnalyzeImage)]:
        # print('Testing: %s %s' % (img_filename, image_klass.__name__))
        test_image_class(pjoin(DATA_PATH, img_filename), image_klass)


def test_sniff_and_guessed_image_type_randomized():
    """Re-test image classes, but in a randomized order."""
    img_klasses = copy.copy(all_image_classes)
    np.random.shuffle(img_klasses)
    test_sniff_and_guessed_image_type(img_klasses=img_klasses)

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

from .. import (Nifti1Image, Nifti1Header, Nifti1Pair, Nifti2Image, Nifti2Pair,
                Minc1Image, Minc2Image, Spm2AnalyzeImage, Spm99AnalyzeImage,
                AnalyzeImage, MGHImage, all_image_classes)

from nose.tools import assert_true, assert_equal, assert_false, assert_raises

DATA_PATH = pjoin(dirname(__file__), 'data')


def test_sniff_and_guessed_image_type(img_klasses=all_image_classes):
    """
    Loop over all test cases:
      * whether a sniff is provided or not
      * randomizing the order of image classes
      * over all known image types

    For each, we expect:
       * When the file matches the expected class, things should
            either work, or fail if we're doing bad stuff.
       * When the file is a mismatch, it should either
         * Fail to be loaded if the image type is unrelated to the expected class
         * Load or fail in a consistent manner, if there is a relationship between
            the image class and expected image class.
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
                new_msg = '%s returned sniff==None (%s)' % (img_klass.__name__, msg)
                expected_sniff_size = getattr(img_klass.header_class, 'sniff_size', 0)
                current_sniff_size = len(new_sniff) if new_sniff is not None else 0
                assert_true(current_sniff_size >= expected_sniff_size, new_msg)

            # Build a message to the user.
            new_msg = '%s (%s) image is%s a %s image.' % (
                basename(img_path),
                msg,
                '' if is_img else ' not',
                img_klass.__name__)

            if expect_success is None:
                assert_true(True, new_msg)  # No expectation, pass if no Exception
            # elif is_img != expect_success:
                # print('Failed! %s' % new_msg)
            else:
                assert_equal(is_img, expect_success, new_msg)

            if sniff_mode == 'vanilla':
                return new_sniff
            else:
                return sniff

        sniff_size = getattr(expected_img_klass.header_class, 'sniff_size', 0)

        for sniff_mode, sniff in dict(
                vanilla=None,  # use the sniff of the previous item
                no_sniff=None,  # Don't pass a sniff
                none=None,  # pass None as the sniff, should query in fn
                empty='',  # pass an empty sniff, should query in fn
                irrelevant='a' * (sniff_size - 1),  # A too-small sniff, query
                bad_sniff='a' * sniff_size).items():  # Bad sniff, should fail.

            for klass in img_klasses:
                if klass == expected_img_klass:
                    expect_success = (sniff_mode not in ['bad_sniff'] or
                                      sniff_size == 0 or
                                      klass == Minc1Image)  # special case...
                elif (issubclass(klass, expected_img_klass) or
                      issubclass(expected_img_klass, klass)):
                    expect_success = None  # Images are related; can't be sure.
                else:
                    # Usually, if the two images are unrelated, they
                    # won't be able to be loaded. But here's a
                    # list of manually confirmed special cases
                    expect_success = ((expected_img_klass == Nifti1Pair and klass == Spm99AnalyzeImage) or
                                      (expected_img_klass == Nifti2Pair and klass == Spm99AnalyzeImage))

                msg = '%s / %s / %s' % (expected_img_klass.__name__, sniff_mode, str(expect_success))
                print(msg)
                # Reuse the sniff... but it will only change for some
                # sniff_mode values.
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
        print('Testing: %s %s' % (img_filename, image_klass.__name__))
        test_image_class(pjoin(DATA_PATH, img_filename), image_klass)


def test_sniff_and_guessed_image_type_randomized():
    """Re-test image classes, but in a randomized order."""
    img_klasses = copy.copy(all_image_classes)
    np.random.shuffle(img_klasses)
    test_sniff_and_guessed_image_type(img_klasses=img_klasses)

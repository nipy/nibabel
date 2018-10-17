#!/usr/bin/env python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Build helper."""

import os
from os.path import join as pjoin
import sys
from functools import partial

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# For some commands, use setuptools.
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'install_egg_info', 'egg_info', 'easy_install', 'bdist_wheel',
            'bdist_mpkg')).intersection(sys.argv)) > 0:
    # setup_egg imports setuptools setup, thus monkeypatching distutils.
    import setup_egg  # noqa

from distutils.core import setup

# Commit hash writing, and dependency checking
from nisext.sexts import (get_comrec_build, package_check, install_scripts_bat,
                          read_vars_from)
cmdclass = {'build_py': get_comrec_build('nibabel'),
            'install_scripts': install_scripts_bat}

# Get project related strings.
INFO = read_vars_from(pjoin('nibabel', 'info.py'))

# Prepare setuptools args
if 'setuptools' in sys.modules:
    extra_setuptools_args = dict(
        tests_require=['nose'],
        test_suite='nose.collector',
        zip_safe=False,
        extras_require=dict(
            doc='Sphinx>=0.3',
            test='nose>=0.10.1'),
    )
    pkg_chk = partial(package_check, setuptools_args = extra_setuptools_args)
else:
    extra_setuptools_args = {}
    pkg_chk = package_check

# Do dependency checking
pkg_chk('numpy', INFO.NUMPY_MIN_VERSION)
pkg_chk('six', INFO.SIX_MIN_VERSION)
custom_pydicom_messages = {'missing opt': 'Missing optional package "%s"'
        ' provided by package "pydicom"'
}
pkg_chk('dicom',
        INFO.PYDICOM_MIN_VERSION,
        optional='dicom',
        messages = custom_pydicom_messages)

def main(**extra_args):
    setup(name=INFO.NAME,
          maintainer=INFO.MAINTAINER,
          maintainer_email=INFO.MAINTAINER_EMAIL,
          description=INFO.DESCRIPTION,
          long_description=INFO.LONG_DESCRIPTION,
          url=INFO.URL,
          download_url=INFO.DOWNLOAD_URL,
          license=INFO.LICENSE,
          classifiers=INFO.CLASSIFIERS,
          author=INFO.AUTHOR,
          author_email=INFO.AUTHOR_EMAIL,
          platforms=INFO.PLATFORMS,
          version=INFO.VERSION,
          requires=INFO.REQUIRES,
          provides=INFO.PROVIDES,
          packages     = ['nibabel',
                          'nibabel.externals',
                          'nibabel.externals.tests',
                          'nibabel.gifti',
                          'nibabel.gifti.tests',
                          'nibabel.cifti2',
                          'nibabel.cifti2.tests',
                          'nibabel.cmdline',
                          'nibabel.cmdline.tests',
                          'nibabel.nicom',
                          'nibabel.freesurfer',
                          'nibabel.freesurfer.tests',
                          'nibabel.nicom.tests',
                          'nibabel.testing',
                          'nibabel.tests',
                          'nibabel.benchmarks',
                          'nibabel.streamlines',
                          'nibabel.streamlines.tests',
                          # install nisext as its own package
                          'nisext',
                          'nisext.tests'],
          # The package_data spec has no effect for me (on python 2.6) -- even
          # changing to data_files doesn't get this stuff included in the source
          # distribution -- not sure if it has something to do with the magic
          # above, but distutils is surely the worst piece of code in all of
          # python -- duplicating things into MANIFEST.in but this is admittedly
          # only a workaround to get things started -- not a solution
          package_data = {'nibabel':
                          [pjoin('tests', 'data', '*'),
                           pjoin('externals', 'tests', 'data', '*'),
                           pjoin('nicom', 'tests', 'data', '*'),
                           pjoin('gifti', 'tests', 'data', '*'),
                           pjoin('streamlines', 'tests', 'data', '*'),
                          ]},
          scripts      = [pjoin('bin', 'parrec2nii'),
                          pjoin('bin', 'nib-ls'),
                          pjoin('bin', 'nib-dicomfs'),
                          pjoin('bin', 'nib-nifti-dx'),
                          pjoin('bin', 'nib-tck2trk'),
                          pjoin('bin', 'nib-trk2tck'),
                          pjoin('bin', 'nib-diff'),
                          ],
          cmdclass = cmdclass,
          **extra_args
         )


if __name__ == "__main__":
    main(**extra_setuptools_args)

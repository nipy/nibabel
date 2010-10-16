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

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from distutils.core import setup

# For some commands, use setuptools.  
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    # setup_egg imports setuptools setup, thus monkeypatching distutils. 
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

from nisext.sexts import get_comrec_build, package_check
cmdclass = {'build_py': get_comrec_build('nibabel')}

# Get version and release info, which is all stored in nibabel/info.py
ver_file = os.path.join('nibabel', 'info.py')
execfile(ver_file)

# Do dependency checking
package_check('numpy', NUMPY_MIN_VERSION)
package_check('dicom', PYDICOM_MIN_VERSION, optional=True)
if 'setuptools' in sys.modules:
    extra_setuptools_args['extras_require'] = dict(
        doc='Sphinx>=0.3',
        test='nose>=0.10.1',
        nicom = 'dicom>=' + PYDICOM_MIN_VERSION)

def main(**extra_args):
    setup(name=NAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          url=URL,
          download_url=DOWNLOAD_URL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          version=VERSION,
          requires=REQUIRES,
          provides=PROVIDES,
          packages     = ['nibabel',
                          'nibabel.externals',
                          'nibabel.gifti',
                          'nibabel.nicom',
                          'nibabel.nicom.tests',
                          'nibabel.testing',
                          'nibabel.tests',
                          # required in setup.py, hence needs to go into source
                          # dist
                          'nisext'],
          # The package_data spec has no effect for me (on python 2.6) -- even
          # changing to data_files doesn't get this stuff included in the source
          # distribution -- not sure if it has something to do with the magic
          # above, but distutils is surely the worst piece of code in all of
          # python -- duplicating things into MANIFEST.in but this is admittedly
          # only a workaround to get things started -- not a solution
          package_data = {'nibabel':
                          [pjoin('tests', 'data', '*'),
                           pjoin('nicom', 'tests', 'data', '*'),
                          ]},
          scripts      = [pjoin('bin', 'parrec2nii')],
          cmdclass = cmdclass,
          **extra_args
         )


if __name__ == "__main__":
    main(**extra_setuptools_args)

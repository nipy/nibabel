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
from ConfigParser import ConfigParser

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

from distutils.command.build_py import build_py

class MyBuildPy(build_py):
    ''' Subclass to write commit data into installation tree '''
    def run(self):
        build_py.run(self)
        import subprocess
        proc = subprocess.Popen('git rev-parse --short HEAD',
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)
        repo_commit, _ = proc.communicate()
        # We write the installation commit even if it's empty
        cfg_parser = ConfigParser()
        cfg_parser.read(os.path.join('nibabel', 'COMMIT_INFO.txt'))
        cfg_parser.set('commit hash', 'install_hash', repo_commit)
        out_pth = pjoin(self.build_lib, 'nibabel', 'COMMIT_INFO.txt')
        cfg_parser.write(open(out_pth, 'wt'))

cmdclass = {'build_py': MyBuildPy}

# Get version and release info, which is all stored in nibabel/version.py
ver_file = os.path.join('nibabel', 'version.py')
execfile(ver_file)

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
          packages     = ['nibabel',
                          'nibabel.externals',
                          'nibabel.nicom',
                          'nibabel.nicom.tests',
                          'nibabel.gifti',
                          'nibabel.testing',
                          'nibabel.tests'],
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

#!/usr/bin/env python
""" Refresh README.rst file from long description

Should be run from nibabel root (containing setup.py)
"""
from __future__ import print_function

import os
import runpy

readme_lines = []
with open('README.rst', 'rt') as fobj:
    for line in fobj:
        readme_lines.append(line)
        if line.startswith('.. Following contents should be'):
            break
    else:
        raise ValueError('Expected comment not found')

rel = runpy.run_path(os.path.join('nibabel', 'info.py'))

readme = ''.join(readme_lines) + '\n' + rel['LONG_DESCRIPTION']

with open('README.rst', 'wt') as fobj:
    fobj.write(readme)

print('Done')

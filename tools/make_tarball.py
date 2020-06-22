#!/usr/bin/env python
"""Simple script to create a tarball with proper git info.
"""

import commands
import os
import sys
import shutil

from  toollib import *

tag = commands.getoutput('git describe')
base_name = f'nibabel-{tag}'
tar_name = f'{base_name}.tgz'

# git archive is weird:  Even if I give it a specific path, it still won't
# archive the whole tree.  It seems the only way to get the whole tree is to cd
# to the top of the tree.  There are long threads (since 2007) on the git list
# about this and it still doesn't work in a sensible way...

start_dir = os.getcwd()
cd('..')
c(f'git archive --format=tar --prefix={base_name}/ HEAD | gzip > {tar_name}')
c(f'mv {tar_name} tools/')

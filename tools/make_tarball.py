#!/usr/bin/env python
"""Simple script to create a tarball with proper git info."""

import commands
import os

from toollib import cd, c

tag = commands.getoutput('git describe')
base_name = 'nibabel-%s' % tag
tar_name = '%s.tgz' % base_name

# git archive is weird:  Even if I give it a specific path, it still won't
# archive the whole tree.  It seems the only way to get the whole tree is to cd
# to the top of the tree.  There are long threads (since 2007) on the git list
# about this and it still doesn't work in a sensible way...

start_dir = os.getcwd()
cd('..')
git_tpl = 'git archive --format=tar --prefix={0}/ HEAD | gzip > {1}'
c(git_tpl.format(base_name, tar_name))
c('mv {0} tools/'.format(tar_name))

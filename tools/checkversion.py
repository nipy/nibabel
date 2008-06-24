#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Check version string for identity."""

__docformat__ = 'restructuredtext'

import sys
import re

checkver = None

for line in sys.stdin:
    file = re.findall('^[^:]*', line)[0]
    print file, '::',
    # look for versions
    ver = re.findall('([\d]+)\.([\d]+)\.([\d]+)', line)

    if len(ver):
        ver_str = '.'.join(ver[0])
        print ver_str
        if checkver == None:
            checkver = ver_str
        else:
            if not checkver == ver_str:
                print 'Deviant version string detected.'
                sys.exit(1)
    else:
        print 'WARNING: Version string might be missed!\n', line

print 'Versions seem to be in sync.'
sys.exit(0)

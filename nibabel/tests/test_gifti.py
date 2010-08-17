# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

try:
    from .. import gifti as gi
    from cviewer.io.gifti import gifti as ge
except ImportError:
    from nose import SkipTest
    raise SkipTest

import time

now = time.time()
g = gi.loadImage('datasets/testsubject.gii')
#g = parse_gifti_file('datasets/testsubject.gii')
print 'time passed (python version)', time.time() - now

now = time.time()
g2 = ge.loadImage('datasets/testsubject.gii')
print "time passed (c version)", time.time() - now

# Fast XML parsing
# http://docs.python.org/library/pyexpat.html



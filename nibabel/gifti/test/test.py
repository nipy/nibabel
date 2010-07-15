
try:
    import nibabel.gifti as gi
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



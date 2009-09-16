''' Tests for Euler angles '''

import numpy as np

import nifti.eulerangles as nea

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

# Example rotations '''
eg_rots = []
params = (-np.pi,np.pi,np.pi/3)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            eg_rots.append((x, y, z))


def diebel_euler(x, y, z):
    co = np.cos(x)
    so = np.sin(x)
    ct = np.cos(y)
    st = np.cos(y)
    cu = np.cos(z)
    su = np.cos(z)
    return np.array(
        [[ct*cu, ct*su, -st],
         [so*st*cu - co*su, so*st*su-co*cu, ct*so],
         [co*st*cu + so*su, co*st*su-so*cu, ct*co]])


def test_convention():
    M = nea.euler2mat()
    yield assert_array_equal, M, np.eye(4)
    for x, y, z in eg_rots:
        yield (assert_array_almost_equal,
               diebel_euler(x, y, z),
               nea.euler2mat(x, y, z)
               )

''' Tests for Euler angles '''

import numpy as np

import nifti.eulerangles as nea
import nifti.quaternions as nq

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal, dec

# Example rotations '''
eg_rots = []
params = (-np.pi,np.pi,np.pi/2)
xs = np.arange(*params)
ys = np.arange(*params)
zs = np.arange(*params)
for x in xs:
    for y in ys:
        for z in zs:
            eg_rots.append((x, y, z))


def x_only(x):
    cosx = np.cos(x)
    sinx = np.sin(x)
    return np.array(
        [[1, 0, 0],
         [0, cosx, sinx],
         [0, -sinx, cosx]])

                 
def y_only(y):
    cosy = np.cos(y)
    siny = np.sin(y)
    return np.array(
        [[cosy, 0, -siny],
         [0, 1, 0],
         [siny, 0, cosy]])


def z_only(z):
    cosz = np.cos(z)
    sinz = np.sin(z)
    return np.array(
                [[cosz, sinz, 0],
                 [-sinz, cosz, 0],
                 [0, 0, 1]])


def diebel_euler(x, y, z):
    # The whole matrix formula for 1,2,3 from Diebel (2006)
    co = np.cos(x)
    so = np.sin(x)
    ct = np.cos(y)
    st = np.sin(y)
    cu = np.cos(z)
    su = np.sin(z)
    return np.array(
        [[ct*cu, ct*su, -st],
         [so*st*cu - co*su, so*st*su+co*cu, ct*so],
         [co*st*cu + so*su, co*st*su-so*cu, ct*co]])


def test_convention():
    M = nea.euler2mat()
    yield assert_array_equal, M, np.eye(3)
    for x, y, z in eg_rots:
        M1 = diebel_euler(x, y, z)
        M2 = nea.euler2mat(x, y, z)
        yield assert_array_almost_equal, M1, M2
        M3 = np.dot(x_only(x), np.dot(y_only(y), z_only(z)))
        yield assert_array_almost_equal, M1, M3
        xp, yp, zp = nea.mat2euler(M1)
        # The parameters may not be the same as input, but they give the
        # same rotation matrix
        M4 = nea.euler2mat(xp, yp, zp)
        yield assert_array_almost_equal, M1, M4


# Quaternion conversion seems to have the complex part inverted
@dec.knownfailureif(True)
def test_quats():
    for x, y, z in eg_rots:
        M1 = nea.euler2mat(x, y, z)
        quatM = nq.mat2quat(M1)
        quat = nea.euler2quat(x, y, z)
        yield assert_array_almost_equal, quatM, quat

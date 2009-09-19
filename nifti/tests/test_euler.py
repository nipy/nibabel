''' Tests for Euler angles '''

import math
import numpy as np

import nifti.eulerangles as nea
import nifti.quaternions as nq

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

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
         [0, cosx, -sinx],
         [0, sinx, cosx]])

                 
def y_only(y):
    cosy = np.cos(y)
    siny = np.sin(y)
    return np.array(
        [[cosy, 0, siny],
         [0, 1, 0],
         [-siny, 0, cosy]])


def z_only(z):
    cosz = np.cos(z)
    sinz = np.sin(z)
    return np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]])


def sympy_euler(z, y, x):
    # The whole matrix formula for z,y,x rotations from Sympy
    cos = math.cos
    sin = math.sin
    # the following copy / pasted from Sympy - see derivations subdirectory
    return [
        [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
        [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
        [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
        ]


def test_euler_mat():
    M = nea.euler2mat()
    yield assert_array_equal, M, np.eye(3)
    for x, y, z in eg_rots:
        M1 = nea.euler2mat(z, y, x)
        M2 = sympy_euler(z, y, x)
        yield assert_array_almost_equal, M1, M2
        M3 = np.dot(x_only(x), np.dot(y_only(y), z_only(z)))
        yield assert_array_almost_equal, M1, M3
        zp, yp, xp = nea.mat2euler(M1)
        # The parameters may not be the same as input, but they give the
        # same rotation matrix
        M4 = nea.euler2mat(zp, yp, xp)
        yield assert_array_almost_equal, M1, M4


def sympy_euler2quat(z=0, y=0, x=0):
    # direct formula for z,y,x quaternion rotations using sympy
    # see derivations subfolder
    cos = math.cos
    sin = math.sin
    # the following copy / pasted from Sympy output
    return (cos(0.5*x)*cos(0.5*y)*cos(0.5*z) - sin(0.5*x)*sin(0.5*y)*sin(0.5*z),
            cos(0.5*x)*sin(0.5*y)*sin(0.5*z) + cos(0.5*y)*cos(0.5*z)*sin(0.5*x),
            cos(0.5*x)*cos(0.5*z)*sin(0.5*y) - cos(0.5*y)*sin(0.5*x)*sin(0.5*z),
            cos(0.5*x)*cos(0.5*y)*sin(0.5*z) + cos(0.5*z)*sin(0.5*x)*sin(0.5*y))

            
def test_quat2euler():
    # Test for a numerical error in euler2mat
    z, y, x = -np.pi / 2, -np.pi / 2, -np.pi
    M1 = nea.euler2mat(z, y, x)
    zp, yp, xp = nea.mat2euler(M1)
    M2 = nea.euler2mat(zp, yp, xp)
    yield assert_array_almost_equal, M1, M2
    quat = nea.euler2quat(z, y, x)
    M3 = nq.quat2mat(quat)
    yield assert_array_almost_equal, M1, M3
    zp, yp, xp = nea.mat2euler(M3)
    M4 = nea.euler2mat(zp, yp, xp)
    yield assert_array_almost_equal, M1, M4
    
            
def test_quats():
    for x, y, z in eg_rots:
        M1 = nea.euler2mat(z, y, x)
        quatM = nq.mat2quat(M1)
        quat = nea.euler2quat(z, y, x)
        yield nq.nearly_equivalent, quatM, quat
        quatS = sympy_euler2quat(z, y, x)
        yield nq.nearly_equivalent, quat, quatS
        zp, yp, xp = nea.quat2euler(quat)
        # The parameters may not be the same as input, but they give the
        # same rotation matrix
        M2 = nea.euler2mat(zp, yp, xp)
        yield assert_array_almost_equal, M1, M2
        

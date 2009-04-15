''' Test quaternion calculations '''

import numpy as np
from numpy import pi

from numpy.testing import dec, assert_raises

import nifti.quaternions as vq

def euler2mat(z=0, y=0, x=0):
    ''' Return matrices for give rotations around z, y and x axes
    '''
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        M = np.array([
            [cosz, sinz, 0],
            [-sinz, cosz, 0],
            [0, 0, 1]])
    else:
        M = np.eye(3)
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        My = np.array([
            [cosy, 0, -siny],
            [0, 1, 0],
            [siny, 0, cosy]])
        M = np.dot(My, M)
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Mx = np.array([
            [1, 0, 0],
            [0, cosx, sinx],
            [0, -sinx, cosx]])
        M = np.dot(Mx, M)
    return M

# Example rotations '''
eg_rots = []
params = (-pi,pi,pi/4)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            eg_rots.append(euler2mat(z,y,x))
# Example quaternions (from rotations)
eg_quats = []
for M in eg_rots:
    eg_quats.append(vq.mat2quat(M))
# M, quaternion pairs
eg_pairs = zip(eg_rots, eg_quats)

# Set of arbitrary unit quaternions
unit_quats = set()
params = (-3,4)
for w in range(*params):
    for x in range(*params):
        for y in range(*params):
            for z in range(*params):
                q = (w, x, y, z)
                Nq = np.sqrt(np.dot(q, q))
                if not Nq == 0:
                    q = tuple([e / Nq for e in q])
                    unit_quats.add(q)

def test_fillpos():
    # Takes np array
    xyz = np.zeros((3,))
    w,x,y,z = vq.fillpositive(xyz)
    assert w == 1
    # Or lists
    xyz = [0] * 3
    w,x,y,z = vq.fillpositive(xyz)
    assert w == 1
    # Errors with wrong number of values
    assert_raises(ValueError, vq.fillpositive, [0, 0])
    assert_raises(ValueError, vq.fillpositive, [0]*4)
    # Errors with negative w2
    assert_raises(ValueError, vq.fillpositive, [1.0]*3)
    # Test corner case where w is near zero
    wxyz = vq.fillpositive([1,0,0])
    assert wxyz[0] == 0.0

def test_conjugate():
    # Takes sequence
    cq = vq.conjugate((1, 0, 0, 0))
    # Returns float type
    assert cq.dtype.kind == 'f'

def test_inverse():
    # Takes sequence
    iq = vq.inverse((1, 0, 0, 0))
    # Returns float type
    assert iq.dtype.kind == 'f'
    for M, q in eg_pairs:
        iq = vq.inverse(q)
        iqM = vq.quat2mat(iq)
        iM = np.linalg.inv(M)
        assert np.allclose(iM, iqM)

def test_eye():
    qi = vq.eye()
    assert qi.dtype.kind == 'f'
    assert np.all([1,0,0,0]==qi)
    assert np.allclose(vq.quat2mat(qi), np.eye(3))

def test_norm():
    qi = vq.eye()
    assert vq.norm(qi) == 1
    assert vq.isunit(qi)
    qi[1] = 0.2
    assert not vq.isunit(qi)

@dec.slow
def test_mult():
    ''' Test that quaternion * same as matrix * '''
    for M1, q1 in eg_pairs[0::4]:
        for M2, q2 in eg_pairs[1::4]:
            q12 = vq.mult(q1, q2)
    assert np.allclose(np.dot(M2,M1), vq.quat2mat(q12))

def test_inverse():
    for M, q in eg_pairs:
        iq = vq.inverse(q)
        iqM = vq.quat2mat(iq)
        iM = np.linalg.inv(M)
        assert np.allclose(iM, iqM)

def test_eye():
    qi = vq.eye()
    assert np.all([1,0,0,0]==qi)
    assert np.allclose(vq.quat2mat(qi), np.eye(3))

@dec.slow
def test_quaternion_reconstruction():
    ''' Test reconstruction of arbitrary unit quaternions '''
    for q in unit_quats:
        M = vq.quat2mat(q)
        qt = vq.mat2quat(M)
        # Accept positive or negative match
        posm = np.allclose(q, qt)
        negm = np.allclose(q, -qt)
        assert posm or negm

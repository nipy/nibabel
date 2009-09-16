''' Test quaternion calculations '''

import numpy as np
from numpy import pi

# Recent (1.2) versions of numpy have this decorator
try:
    from numpy.testing.decorators import slow
except ImportError:
    def slow(t):
        t.slow = True
        return t

from nose.tools import assert_raises, assert_true, assert_false, \
    assert_equal

from numpy.testing import assert_array_almost_equal

import nifti.quaternions as nq
import nifti.eulerangles as nea

def euler2mat(z=0, y=0, x=0):
    ''' Return matrices for give rotations around z, y and x axes
    '''
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
                [[cosz, sinz, 0],
                 [-sinz, cosz, 0],
                 [0, 0, 1]])
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
                    [[cosy, 0, -siny],
                     [0, 1, 0],
                     [siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
                    [[1, 0, 0],
                     [0, cosx, sinx],
                     [0, -sinx, cosx]])
    if Ms:
        return reduce(np.dot, Ms)
    return np.eye(4)

# Example rotations '''
eg_rots = []
params = (-pi,pi,pi/3)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            eg_rots.append(nea.euler2mat(z,y,x))
# Example quaternions (from rotations)
eg_quats = []
for M in eg_rots:
    eg_quats.append(nq.mat2quat(M))
# M, quaternion pairs
eg_pairs = zip(eg_rots, eg_quats)

# Set of arbitrary unit quaternions
unit_quats = set()
params = (-2,3)
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
    w,x,y,z = nq.fillpositive(xyz)
    yield assert_true, w == 1
    # Or lists
    xyz = [0] * 3
    w,x,y,z = nq.fillpositive(xyz)
    yield assert_true, w == 1
    # Errors with wrong number of values
    yield assert_raises, ValueError, nq.fillpositive, [0, 0]
    yield assert_raises, ValueError, nq.fillpositive, [0]*4
    # Errors with negative w2
    yield assert_raises, ValueError, nq.fillpositive, [1.0]*3
    # Test corner case where w is near zero
    wxyz = nq.fillpositive([1,0,0])
    yield assert_true, wxyz[0] == 0.0


def test_conjugate():
    # Takes sequence
    cq = nq.conjugate((1, 0, 0, 0))
    # Returns float type
    yield assert_true, cq.dtype.kind == 'f'


def test_inverse():
    # Takes sequence
    iq = nq.inverse((1, 0, 0, 0))
    # Returns float type
    yield assert_true, iq.dtype.kind == 'f'
    for M, q in eg_pairs:
        iq = nq.inverse(q)
        iqM = nq.quat2mat(iq)
        iM = np.linalg.inv(M)
        yield assert_true, np.allclose(iM, iqM)


def test_eye():
    qi = nq.eye()
    yield assert_true, qi.dtype.kind == 'f'
    yield assert_true, np.all([1,0,0,0]==qi)
    yield assert_true, np.allclose(nq.quat2mat(qi), np.eye(3))


def test_norm():
    qi = nq.eye()
    yield assert_true, nq.norm(qi) == 1
    yield assert_true, nq.isunit(qi)
    qi[1] = 0.2
    yield assert_true, not nq.isunit(qi)


@slow
def test_mult():
    # Test that quaternion * same as matrix * 
    for M1, q1 in eg_pairs[0::4]:
        for M2, q2 in eg_pairs[1::4]:
            q21 = nq.mult(q2, q1)
            yield assert_array_almost_equal, np.dot(M2,M1), nq.quat2mat(q21)


def test_inverse():
    for M, q in eg_pairs:
        iq = nq.inverse(q)
        iqM = nq.quat2mat(iq)
        iM = np.linalg.inv(M)
        yield assert_true, np.allclose(iM, iqM)


def test_eye():
    qi = nq.eye()
    yield assert_true, np.all([1,0,0,0]==qi)
    yield assert_true, np.allclose(nq.quat2mat(qi), np.eye(3))


@slow
def test_qrotate():
    vecs = np.eye(3)
    for vec in np.eye(3):
        for M, q in eg_pairs:
            vdash = nq.rotate_vector(vec, q)
            vM = np.dot(M, vec.reshape(3,1))[:,0]
            yield assert_array_almost_equal, vdash, vM


@slow
def test_quaternion_reconstruction():
    # Test reconstruction of arbitrary unit quaternions
    for q in unit_quats:
        M = nq.quat2mat(q)
        qt = nq.mat2quat(M)
        # Accept positive or negative match
        posm = np.allclose(q, qt)
        negm = np.allclose(q, -qt)
        yield assert_true, posm or negm

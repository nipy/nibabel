# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
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

from numpy.testing import assert_array_almost_equal, assert_array_equal

from .. import quaternions as nq
from .. import eulerangles as nea

# Example rotations '''
eg_rots = []
params = (-pi, pi, pi / 2)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            eg_rots.append(nea.euler2mat(z, y, x))
# Example quaternions (from rotations)
eg_quats = []
for M in eg_rots:
    eg_quats.append(nq.mat2quat(M))
# M, quaternion pairs
eg_pairs = list(zip(eg_rots, eg_quats))

# Set of arbitrary unit quaternions
unit_quats = set()
params = range(-2, 3)
for w in params:
    for x in params:
        for y in params:
            for z in params:
                q = (w, x, y, z)
                Nq = np.sqrt(np.dot(q, q))
                if not Nq == 0:
                    q = tuple([e / Nq for e in q])
                    unit_quats.add(q)


def test_fillpos():
    # Takes np array
    xyz = np.zeros((3,))
    w, x, y, z = nq.fillpositive(xyz)
    yield assert_true, w == 1
    # Or lists
    xyz = [0] * 3
    w, x, y, z = nq.fillpositive(xyz)
    yield assert_true, w == 1
    # Errors with wrong number of values
    yield assert_raises, ValueError, nq.fillpositive, [0, 0]
    yield assert_raises, ValueError, nq.fillpositive, [0] * 4
    # Errors with negative w2
    yield assert_raises, ValueError, nq.fillpositive, [1.0] * 3
    # Test corner case where w is near zero
    wxyz = nq.fillpositive([1, 0, 0])
    yield assert_true, wxyz[0] == 0.0


def test_conjugate():
    # Takes sequence
    cq = nq.conjugate((1, 0, 0, 0))
    # Returns float type
    yield assert_true, cq.dtype.kind == 'f'


def test_quat2mat():
    # also tested in roundtrip case below
    M = nq.quat2mat([1, 0, 0, 0])
    yield assert_array_almost_equal, M, np.eye(3)
    M = nq.quat2mat([3, 0, 0, 0])
    yield assert_array_almost_equal, M, np.eye(3)
    M = nq.quat2mat([0, 1, 0, 0])
    yield assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = nq.quat2mat([0, 2, 0, 0])
    yield assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = nq.quat2mat([0, 0, 0, 0])
    yield assert_array_almost_equal, M, np.eye(3)


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
    yield assert_true, np.all([1, 0, 0, 0] == qi)
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
            yield assert_array_almost_equal, np.dot(M2, M1), nq.quat2mat(q21)


def test_inverse():
    for M, q in eg_pairs:
        iq = nq.inverse(q)
        iqM = nq.quat2mat(iq)
        iM = np.linalg.inv(M)
        yield assert_true, np.allclose(iM, iqM)


def test_eye():
    qi = nq.eye()
    yield assert_true, np.all([1, 0, 0, 0] == qi)
    yield assert_true, np.allclose(nq.quat2mat(qi), np.eye(3))


def test_qrotate():
    for vec in np.eye(3):
        for M, q in eg_pairs:
            vdash = nq.rotate_vector(vec, q)
            vM = np.dot(M, vec)
            yield assert_array_almost_equal, vdash, vM


def test_quaternion_reconstruction():
    # Test reconstruction of arbitrary unit quaternions
    for q in unit_quats:
        M = nq.quat2mat(q)
        qt = nq.mat2quat(M)
        # Accept positive or negative match
        posm = np.allclose(q, qt)
        negm = np.allclose(q, -qt)
        yield assert_true, posm or negm


def test_angle_axis2quat():
    q = nq.angle_axis2quat(0, [1, 0, 0])
    yield assert_array_equal, q, [1, 0, 0, 0]
    q = nq.angle_axis2quat(np.pi, [1, 0, 0])
    yield assert_array_almost_equal, q, [0, 1, 0, 0]
    q = nq.angle_axis2quat(np.pi, [1, 0, 0], True)
    yield assert_array_almost_equal, q, [0, 1, 0, 0]
    q = nq.angle_axis2quat(np.pi, [2, 0, 0], False)
    yield assert_array_almost_equal, q, [0, 1, 0, 0]


def test_angle_axis():
    for M, q in eg_pairs:
        theta, vec = nq.quat2angle_axis(q)
        q2 = nq.angle_axis2quat(theta, vec)
        yield nq.nearly_equivalent, q, q2
        aa_mat = nq.angle_axis2mat(theta, vec)
        yield assert_array_almost_equal, aa_mat, M

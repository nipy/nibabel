# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Test quaternion calculations """

import numpy as np
from numpy import pi

import pytest

from numpy.testing import assert_array_almost_equal, assert_array_equal, dec

from .. import quaternions as nq
from .. import eulerangles as nea

# Example rotations
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
    assert w == 1
    # Or lists
    xyz = [0] * 3
    w, x, y, z = nq.fillpositive(xyz)
    assert w == 1
    # Errors with wrong number of values
    with pytest.raises(ValueError):
        nq.fillpositive([0, 0])
    with pytest.raises(ValueError):
        nq.fillpositive([0] * 4)
    # Errors with negative w2
    with pytest.raises(ValueError):
        nq.fillpositive([1.0] * 3)
    # Test corner case where w is near zero
    wxyz = nq.fillpositive([1, 0, 0])
    assert wxyz[0] == 0.0


def test_conjugate():
    # Takes sequence
    cq = nq.conjugate((1, 0, 0, 0))
    # Returns float type
    assert cq.dtype.kind == 'f'


def test_quat2mat():
    # also tested in roundtrip case below
    M = nq.quat2mat([1, 0, 0, 0])
    assert_array_almost_equal, M, np.eye(3)
    M = nq.quat2mat([3, 0, 0, 0])
    assert_array_almost_equal, M, np.eye(3)
    M = nq.quat2mat([0, 1, 0, 0])
    assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = nq.quat2mat([0, 2, 0, 0])
    assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = nq.quat2mat([0, 0, 0, 0])
    assert_array_almost_equal, M, np.eye(3)


def test_inverse_0():
    # Takes sequence
    iq = nq.inverse((1, 0, 0, 0))
    # Returns float type
    assert iq.dtype.kind == 'f'


@pytest.mark.parametrize("M, q", eg_pairs)
def test_inverse_1(M, q):
    iq = nq.inverse(q)
    iqM = nq.quat2mat(iq)
    iM = np.linalg.inv(M)
    assert np.allclose(iM, iqM)


def test_eye():
    qi = nq.eye()
    assert qi.dtype.kind == 'f'
    assert np.all([1, 0, 0, 0] == qi)
    assert np.allclose(nq.quat2mat(qi), np.eye(3))


def test_norm():
    qi = nq.eye()
    assert nq.norm(qi) == 1
    assert nq.isunit(qi)
    qi[1] = 0.2
    assert not nq.isunit(qi)


@dec.slow
@pytest.mark.parametrize("M1, q1", eg_pairs[0::4])
@pytest.mark.parametrize("M2, q2", eg_pairs[1::4])
def test_mult(M1, q1, M2, q2):
    # Test that quaternion * same as matrix *
    q21 = nq.mult(q2, q1)
    assert_array_almost_equal, np.dot(M2, M1), nq.quat2mat(q21)


@pytest.mark.parametrize("M, q", eg_pairs)
def test_inverse(M, q):
    iq = nq.inverse(q)
    iqM = nq.quat2mat(iq)
    iM = np.linalg.inv(M)
    assert np.allclose(iM, iqM)


def test_eye():
    qi = nq.eye()
    assert np.all([1, 0, 0, 0] == qi)
    assert np.allclose(nq.quat2mat(qi), np.eye(3))


@pytest.mark.parametrize("vec", np.eye(3))
@pytest.mark.parametrize("M, q", eg_pairs)
def test_qrotate(vec, M, q):
    vdash = nq.rotate_vector(vec, q)
    vM = np.dot(M, vec)
    assert_array_almost_equal(vdash, vM)


@pytest.mark.parametrize("q", unit_quats)
def test_quaternion_reconstruction(q):
    # Test reconstruction of arbitrary unit quaternions
    M = nq.quat2mat(q)
    qt = nq.mat2quat(M)
    # Accept positive or negative match
    posm = np.allclose(q, qt)
    negm = np.allclose(q, -qt)
    assert (posm or negm)


def test_angle_axis2quat():
    q = nq.angle_axis2quat(0, [1, 0, 0])
    assert_array_equal(q, [1, 0, 0, 0])
    q = nq.angle_axis2quat(np.pi, [1, 0, 0])
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = nq.angle_axis2quat(np.pi, [1, 0, 0], True)
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = nq.angle_axis2quat(np.pi, [2, 0, 0], False)
    assert_array_almost_equal(q, [0, 1, 0, 0])


def test_angle_axis():
    for M, q in eg_pairs:
        theta, vec = nq.quat2angle_axis(q)
        q2 = nq.angle_axis2quat(theta, vec)
        nq.nearly_equivalent(q, q2)
        aa_mat = nq.angle_axis2mat(theta, vec)
        assert_array_almost_equal(aa_mat, M)
        unit_vec = vec / np.sqrt(vec.dot(vec))
        aa_mat2 = nq.angle_axis2mat(theta, unit_vec, is_normalized=True)
        assert_array_almost_equal(aa_mat2, M)

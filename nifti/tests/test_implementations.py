import numpy as np

import nifti.quaternions as nq
import nifti.eulerangles as nea

# Import transformations module
import os
import sys
_my_path, _ = os.path.split(__file__)
sys.path.append(_my_path)
from transformations import rotation_matrix_from_quaternion, \
    quaternion_from_rotation_matrix, rotation_matrix_from_euler, \
    quaternion_from_euler
sys.path.remove(_my_path)

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal, dec

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


# Example rotations '''
eg_rots = []
params = (-np.pi,np.pi,np.pi/2)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            eg_rots.append((x, y, z))


def quat_equal(q1, q2, rtol=1e-05, atol=1e-08):
    # q * -1 is same transformation as q
    q1 = np.array(q1)
    q2 = np.array(q2)
    if np.allclose(q1, q2, rtol, atol):
        return True
    return np.allclose(q1 * -1, q2, rtol, atol)


def trans_quat(quat):
    # converts back and forth between transformations / nifti quat
    w, x, y, z = quat
    return [x, y, z, w]
    

def test_quaternion_imps():
    for x, y, z in eg_rots:
        M = nea.euler2mat(x, y, z)
        quat = nq.mat2quat(M)
        # Against transformations code
        tM = rotation_matrix_from_quaternion(trans_quat(quat))
        yield assert_array_almost_equal, M, tM[:3,:3]
        M44 = np.eye(4)
        M44[:3,:3] = M
        tQ = quaternion_from_rotation_matrix(M44)
        yield assert_true, quat_equal(trans_quat(quat), tQ)


def test_euler_imps():
    for x, y, z in eg_rots:
        M1 = rotation_matrix_from_euler(z, y, x,'szyx')[:3,:3]
        M2 = nea.euler2mat(x, y, z)
        yield assert_array_almost_equal, M1, M2
        q1 = quaternion_from_euler(z, y, x, 'szyx')
        q2 = nea.euler2quat(x, y, z)
        yield assert_true, quat_equal(trans_quat(q1), q2)

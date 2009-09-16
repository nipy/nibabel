import numpy as np

import nifti.quaternions as nq

import os
import sys
_my_path, _ = os.path.split(__file__)
sys.path.append(_my_path)
from transformations import rotation_matrix_from_quaternion, \
    quaternion_from_rotation_matrix
sys.path.remove(_my_path)

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal

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
    for quat in unit_quats:
        M = nq.quat2mat(quat)
        q_back = nq.mat2quat(M)
        yield assert_true, quat_equal(quat, q_back)
        # Against transformations code
        tM = rotation_matrix_from_quaternion(trans_quat(quat))
        yield assert_array_almost_equal, M, tM[:3,:3]
        M44 = np.eye(4)
        M44[:3,:3] = M
        tQ = quaternion_from_rotation_matrix(M44)
        yield assert_true, quat_equal(trans_quat(quat), tQ)

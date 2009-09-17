''' Module implementing Euler angle rotations and their conversions

See: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134

See also:

* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html

Euler angles for rotation in 3 dimensions consist of three rotations.
Following Diebel (2006), they follow the sequence:

#. rotation of angle ``psi`` around axis ``k``, followed by
#. rotation of angle ``theta`` around axis ``j`` followed by
#. rotation of angle ``phi`` around axis ``i``.

Given a triplet of angles - or *Euler angle vector* - [ ``phi``,
``theta``, ``psi`` ], we need to know the respective axes of rotation [
``i``, ``j``, ``k`` ] (remember that the "``psi`` around ``k``" rotation
is applied first).

There are various conventions mapping [ ``i``, ``j``, ``k`` ] to the
``x``, ``y`` and ``z`` axes; here we follow:

* ``i`` is the ``x`` axis
* ``j`` is the ``y`` axis
* ``k`` is the ``z`` axis

Thus (to repeat), the Euler angle vector [ ``phi``, ``theta``, ``psi``
], in our convention, leads to a rotation of ``psi`` around the ``z``
axis, followed by a rotation of ``theta`` around the ``y`` axis,
followed by a rotation of ``psi`` around the ``x`` axis.

We also specify, for our rotations, that the axes are in a static frame
(the axes stay in the same place as the successive rotations are
applied, rather than moving with the rotations). 

In published and online descriptions of Euler angles, the order in which
the rotations are specified in the Euler angle vector, differs from
source to source.  For example, sometimes the first angle in the vector
is the first to be applied.  Here we follow Diebel (2006) in applying
the rotation angles / axes in the reverse order to that specified in the
vector.  That is just to repeat that, for an Euler angle vector [
``phi``, ``theta``, ``psi`` ], we apply ``psi`` followed by ``theta``
followed by ``phi``).

The convention of rotation around ``z``, followed by rotation around
``y``, followed by rotation around ``x``, is known as "xyz",
pitch-roll-yaw, Cardan angles, or Tait-Bryan angles.

'''

import math
import numpy as np


def euler2mat(x=0, y=0, z=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    y : scalar
       Rotation angle in radians around y-axis
    z : scalar
       Rotation angle in radians around z-axis (performed first)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)

    Returns
    -------
    x : scalar
    y : scalar
    z : scalar
       Rotations in radians around x, y, z axes, respectively

    Notes
    -----
    From Diebel (2006) page 12 (but note transpose)

    '''
    M = np.asarray(M)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    return math.atan2(-r23, r33), math.asin(r13), math.atan2(-r12, r11)

    
def euler2quat(x=0, y=0, z=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    y : scalar
       Rotation angle in radians around y-axis
    z : scalar
       Rotation angle in radians around z-axis (performed first)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    From Diebel 2006 page 12
    '''
    x = x/2.0
    y = y/2.0
    z = z/2.0
    co = math.cos(x)
    so = math.sin(x)
    ct = math.cos(y)
    st = math.sin(y)
    cu = math.cos(z)
    su = math.sin(z)
    return np.array([
             co*ct*cu + so*st*su,
            -co*st*su + ct*cu*so,
             co*cu*st + so*ct*su,
             co*ct*su-so*cu*st])

             
def euler2quat2(x=0, y=0, z=0):
    ''' Euler to quaternion using axis / angle quaternions for xyz '''
    import nifti.quaternions as nq
    # Make 4x3 array, where columns are x, y, z axis quaternions, and
    # rows are (real, vector 0,1,2) of quaternion.
    xyz_half = np.array([x, y, z]) / 2.0
    quat_vectors = np.diag(np.sin(xyz_half)) # vectors already normalized
    quat_scalars = np.cos(xyz_half)
    quats = np.concatenate(np.atleast_2d(quat_scalars, quat_vectors))
    # Reduce quaternion multiply over (z, y, x) ordering
    return reduce(nq.mult, quats.T[::-1])

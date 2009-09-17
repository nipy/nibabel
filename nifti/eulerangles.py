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


def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

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
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    Derived using Sympy expression for z then y then x rotation matrix,
    see ``eulerangles.py`` in ``derivations`` subdirectory
    '''
    M = np.asarray(M)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    return math.atan2(-r12, r11), math.asin(r13), math.atan2(-r23, r33)

    
def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    Formula from Sympy - see ``eulerangles.py`` in ``derivations``
    subdirectory
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])

             

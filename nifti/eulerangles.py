''' Module implementing Euler angle rotations and their conversions

See:

* http://en.wikipedia.org/wiki/Rotation_matrix
* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html

See also: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134

Euler's rotation theorem tells us that any rotation in 3D can be
described by 3 angles.  Let's call the 3 angles the *Euler angle vector*
and call the angles in the vector :math:`alpha`, :math:`beta` and
:math:`gamma`.  The vector is [ :math:`alpha`,
:math:`beta`. :math:`gamma` ] and, in this description, the order of the
parameters specifies the order in which the rotations occur (so the
rotation corresponding to :math:`alpha` is applied first).

In order to specify the meaning of an *Euler angle vector* we need to
specify the axes around which each of the rotations corresponding to
:math:`alpha`, :math:`beta` and :math:`gamma` will occur.

There are therefore three axes for the rotations :math:`alpha`,
:math:`beta` and :math:`gamma`; let's call them :math:`i` :math:`j`,
:math:`k`. 

Let us express the rotation :math:`alpha` around axis `i` as a 3 by 3
rotation matrix `A`.  Similarly :math:`beta` around `j` becomes 3 x 3
matrix `B` and :math:`gamma` around `k` becomes matrix `G`.  Then the
whole rotation expressed by the Euler angle vector [ :math:`alpha`,
:math:`beta`. :math:`gamma` ], `R` is given by::

   R = np.dot(G, np.dot(B, A))

See http://mathworld.wolfram.com/EulerAngles.html

The order :math:`G B A` expresses the fact that the rotations are
performed in the order of the vector (:math:`alpha` around axis `i` =
`A` first).

To convert a given Euler angle vector to a meaningful rotation, and a
rotation matrix, we need to define:

* the axes `i`, `j`, `k`
* whether a rotation matrix should be applied on the left of a vector to
  be transformed (vectors are column vectors) or on the right (vectors
  are row vectors).
* whether the rotations move the axes as they are applied (intrinsic
  rotations) - compared the situation where the axes stay fixed and the
  vectors move within the axis frame (extrinsic)
* the handedness of the coordinate system

See: http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

We are using the following conventions:

* axes `i`, `j`, `k` are the `z`, `y`, and `x` axes respectively.  Thus
  an Euler angle vector [ :math:`alpha`, :math:`beta`. :math:`gamma` ]
  in our convention implies a :math:`alpha` radian rotation around the
  `z` axis, followed by a :math:`beta` rotation around the `y` axis,
  followed by a :math:`gamma` rotation around the `x` axis.
* the rotation matrix applies on the left, to column vectors on the
  right, so if `R` is the rotation matrix, and `v` is a 3 x N matrix
  with N column vectors, the transformed vector set `vdash` is given by
  ``vdash = np.dot(R, v)``.
* extrinsic rotations - the axes are fixed, and do not move with the
  rotations.
* a right-handed coordinate system

The convention of rotation around ``z``, followed by rotation around
``y``, followed by rotation around ``x``, is known (confusingly) as
"xyz", pitch-roll-yaw, Cardan angles, or Tait-Bryan angles.

'''

import math
import numpy as np


_FLOAT_EPS = np.finfo(float).eps * 4.0


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


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default) estimate from precision
       of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, see
    ``eulerangles.py`` in ``derivations`` subdirectory, as::

       math.atan2(-r12, r11), math.asin(r13), math.atan2(-r23, r33)

    The difference here is from ``transformations.py`` by Christoph
    Gohlke, http://www.lfd.uci.edu/~gohlke/ released under a BSD license. 
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # by inspection, cy is
    # sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh:
        z = math.atan2(-r12,  r11)
        y = math.atan2(r13,  cy)
        x = math.atan2(-r23, r33)
    else:
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy)
        x = 0.0
    return z, y, x

    
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


def quat2euler(q):
    ''' Return Euler angles corresponding to quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion

    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    '''
    # delayed import to avoid cyclic dependencies
    import nifti.quaternion as nq
    return mat2euler(nq.quat2mat(q))

    
def angle_axis2euler(theta, vector, is_normalized=False):
    ''' Convert angle, axis pair to Euler angles

    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Examples
    --------
    >>> angle_axis2euler(0, [1, 0, 0])
    
    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``angle_axis2mat`` and ``mat2euler``
    functions, but the reduction in computation is small, and the code
    repetition is large.
    '''
    # delayed import to avoid cyclic dependencies
    import nifti.quaternions as nq
    M = nq.angle_axis2mat(theta, vector, is_normalized)
    return mat2euler(M)

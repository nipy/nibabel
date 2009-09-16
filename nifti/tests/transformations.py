# -*- coding: utf-8 -*-
# transformations.py

# Copyright (c) 2006-2007, Christoph Gohlke
# Copyright (c) 2006-2007, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of the copyright holders nor the names of any
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Homogeneous Transformation Matrices and Quaternions.

Functions for calculating 4x4 matrices for translating, rotating, mirroring,
scaling, projecting, orthogonalization, and superimposition of arrays of
homogenous coordinates as well as for converting between rotation matrices,
Euler angles, and quaternions.

Matrices can be inverted using numpy.linalg.inv(M), concatenated using
numpy.dot(M0,M1), or used to transform homogeneous points using
numpy.dot(v, M) for shape (4,*) "point of arrays", respectively
numpy.dot(v, M.T) for shape (*,4) "array of points".

Quaternions ix+jy+kz+w are represented as [x, y, z, w].

Angles are in radians unless specified otherwise.

The 24 ways of specifying rotations for a given triple of Euler angles,
can be represented by a 4 character string or encoded 4-tuple:

  Axes 4-string:
    first character -- rotations are applied to 's'tatic or 'r'otating frame
    remaining characters -- successive rotation axis 'x', 'y', or 'z'
  Axes 4-tuple:
    inner axis -- code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
    parity -- even (0) if inner axis 'x' is followed by 'y', 'y' is followed
        by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
    repetition -- first and last axis are same (1) or different (0).
    frame -- rotations are applied to static (0) or rotating (1) frame.
  Examples of tuple codes:
    'sxyz' <-> (0, 0, 0, 0)
    'ryxy' <-> (1, 1, 1, 1)

Author:
    Christoph Gohlke, http://www.lfd.uci.edu/~gohlke/
    Laboratory for Fluorescence Dynamics, University of California, Irvine

Requirements:
    Python 2.5 (http://www.python.org)
    Numpy 1.0 (http://numpy.scipy.org)

References:
(1) Matrices and Transformations. Ronald Goldman.
    In "Graphics Gems I", pp 472-475. Morgan Kaufmann, 1990.
(2) Euler Angle Conversion. Ken Shoemake.
    In "Graphics Gems IV", pp 222-229. Morgan Kaufmann, 1994.
(3) Arcball Rotation Control. Ken Shoemake.
    In "Graphics Gems IV", pp 175-192. Morgan Kaufmann, 1994.

"""

from __future__ import division

import math
import numpy

_EPS = numpy.finfo(float).eps * 4.0

def concatenate_transforms(*args):
    """Merge multiple transformation matrices into one."""
    M = numpy.identity(4, dtype=numpy.float64)
    for i in args:
        M = numpy.dot(M, i)
    return M

def translation_matrix(direction):
    """Return matrix to translate by direction vector."""
    M = numpy.identity(4, dtype=numpy.float64)
    M[0:3,3] = direction[0:3]
    return M

def mirror_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal
vector."""
    M = numpy.identity(4, dtype=numpy.float64)
    n = numpy.array(normal[0:3], dtype=numpy.float64, copy=True)
    n /= math.sqrt(numpy.dot(n,n))
    M[0:3,0:3] -= 2.0 * numpy.outer(n, n)
    M[0:3,3] = 2.0 * numpy.dot(point[0:3], n) * n
    return M

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction."""
    sa, ca = math.sin(angle), math.cos(angle)
    # unit vector of direction
    u = numpy.array(direction[0:3], dtype=numpy.float64, copy=True)
    u /= math.sqrt(numpy.dot(u, u))
    # rotation matrix around unit vector
    R = numpy.identity(3, dtype=numpy.float64)
    R *= ca
    R += numpy.outer(u, u) * (1.0 - ca)
    R += sa * numpy.array(((  0.0,-u[2], u[1]),
                           ( u[2],  0.0,-u[0]),
                           (-u[1], u[0],  0.0)), dtype=numpy.float64)
    M = numpy.identity(4, dtype=numpy.float64)
    M[0:3,0:3] = R
    if point is not None:
        # rotation not around origin
        M[0:3,3] = point[0:3] - numpy.dot(R, point[0:3])
    return M

def scaling_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Point Symmetry: factor = -1.0

    """
    if origin is None: origin = [0,0,0]
    o = numpy.array(origin[0:3], dtype=numpy.float64, copy=False)
    if direction is None:
        # uniform scaling
        M = factor*numpy.identity(4, dtype=numpy.float64)
        M[0:3,3] = (1.0-factor) * o
        M[3,3] = 1.0
    else:
        # nonuniform scaling
        M = numpy.identity(4, dtype=numpy.float64)
        u = numpy.array(direction[0:3], dtype=numpy.float64, copy=True)
        u /= math.sqrt(numpy.dot(u, u)) # unit vector of direction
        M[0:3,0:3] -= (1.0-factor) * numpy.outer(u, u)
        M[0:3,3] = ((1.0-factor) * numpy.dot(o, u)) * u
    return M

def projection_matrix(point, normal, direction=None, perspective=None):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    """
    M = numpy.identity(4, dtype=numpy.float64)
    n = numpy.array(normal[0:3], dtype=numpy.float64, copy=True)
    n /= math.sqrt(numpy.dot(n,n)) # unit vector of normal
    if perspective is not None:
        # perspective projection
        r = numpy.array(perspective[0:3], dtype=numpy.float64)
        M[0:3,0:3] *= numpy.dot((r-point), n)
        M[0:3,0:3] -= numpy.outer(n, r)
        M[0:3,3] = numpy.dot(point, n) * r
        M[3,0:3] = -n
        M[3,3] = numpy.dot(perspective, n)
    elif direction is not None:
        # parallel projection
        w = numpy.array(direction[0:3], dtype=numpy.float64)
        s = 1.0 / numpy.dot(w, n)
        M[0:3,0:3] -= numpy.outer(n, w) * s
        M[0:3,3] = w * (numpy.dot(point[0:3], n) * s)
    else:
        # orthogonal projection
        M[0:3,0:3] -= numpy.outer(n, n)
        M[0:3,3] = numpy.dot(point[0:3], n) * n
    return M

def orthogonalization_matrix(a=10.0, b=10.0, c=10.0,
                             alpha=90.0, beta=90.0, gamma=90.0):
    """Return orthogonalization matrix for crystallographic cell
coordinates.

    Angles are in degrees.

    """
    al = math.radians(alpha)
    be = math.radians(beta)
    ga = math.radians(gamma)
    sia = math.sin(al)
    sib = math.sin(be)
    coa = math.cos(al)
    cob = math.cos(be)
    co = (coa * cob - math.cos(ga)) / (sia * sib)
    return numpy.array((
        (a*sib*math.sqrt(1.0-co*co), 0.0,   0.0, 0.0),
        (-a*sib*co,                  b*sia, 0.0, 0.0),
        (a*cob,                      b*coa, c,   0.0),
        (0.0,                        0.0,   0.0, 1.0)),
        dtype=numpy.float64)

def superimpose_matrix(v0, v1, compute_rmsd=False):
    """Return matrix to transform given vector set into second vector set.

    Minimize weighted sum of squared deviations according to W. Kabsch.
    v0 and v1 are shape (*,3) or (*,4) arrays of at least 3 vectors.

    """
    v0 = numpy.array(v0, dtype=numpy.float64)
    v1 = numpy.array(v1, dtype=numpy.float64)

    assert v0.ndim==2 and v0.ndim==v1.ndim and \
           v0.shape[0]>2 and v0.shape[1] in (3,4)

    # vectors might be homogeneous coordinates
    if v0.shape[1] == 4:
        v0 = v0[:,0:3]
        v1 = v1[:,0:3]

    # move centroids to origin
    t0 = numpy.mean(v0, axis=0)
    t1 = numpy.mean(v1, axis=0)
    v0 = v0 - t0
    v1 = v1 - t1

    # Singular Value Decomposition of covariance matrix
    u, s, vh = numpy.linalg.svd(numpy.dot(v1.T, v0))

    # rotation matrix from SVD orthonormal bases
    R = numpy.dot(u, vh)
    if numpy.linalg.det(R) < 0.0:
        # R does not constitute right handed system
        rc = vh[2,:] * 2.0
        R -= numpy.vstack((u[0,2]*rc, u[1,2]*rc, u[2,2]*rc))
        s[-1] *= -1.0

    # homogeneous transformation matrix
    M = numpy.identity(4, dtype=numpy.float64)
    T = numpy.identity(4, dtype=numpy.float64)
    M[0:3,0:3] = R
    T[0:3,3] = t1
    M = numpy.dot(T, M)
    T[0:3,3] = -t0
    M = numpy.dot(M, T)

    # compute root mean square error from SVD sigma
    if compute_rmsd:
        r = numpy.cumsum(v0*v0) + numpy.cumsum(v1*v1)
        rmsd = numpy.sqrt(abs(r - (2.0 * sum(s)) / len(v0)))
        return M, rmsd
    else:
        return M

def rotation_matrix_from_euler(ai, aj, ak, axes):
    """Return homogeneous rotation matrix from Euler angles and axis
sequence.

    ai, aj, ak -- Euler's roll, pitch and yaw angles
    axes -- One of 24 axis sequences as string or encoded tuple

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame: ai, ak = ak, ai
    if parity: ai, aj, ak = -ai, -aj, -ak

    si, sj, sh = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ch = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ch, ci*sh
    sc, ss = si*ch, si*sh

    M = numpy.identity(4, dtype=numpy.float64)
    if repetition:
        M[i,i] =  cj;     M[i,j] =  sj*si;    M[i,k] =  sj*ci
        M[j,i] =  sj*sh;  M[j,j] = -cj*ss+cc; M[j,k] = -cj*cs-sc
        M[k,i] = -sj*ch;  M[k,j] =  cj*sc+cs; M[k,k] =  cj*cc-ss
    else:
        M[i,i] =  cj*ch;  M[i,j] =  sj*sc-cs; M[i,k] =  sj*cc+ss
        M[j,i] =  cj*sh;  M[j,j] =  sj*ss+cc; M[j,k] =  sj*cs-sc
        M[k,i] = -sj;     M[k,j] =  cj*si;    M[k,k] =  cj*ci
    return M

def euler_from_rotation_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    matrix -- 3x3 or 4x4 rotation matrix
    axes -- One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64)[0:3, 0:3]
    if repetition:
        sy = math.sqrt(M[i,j]*M[i,j] + M[i,k]*M[i,k])
        if sy > _EPS:
            ax = math.atan2( M[i,j],  M[i,k])
            ay = math.atan2( sy,      M[i,i])
            az = math.atan2( M[j,i], -M[k,i])
        else:
            ax = math.atan2(-M[j,k],  M[j,j])
            ay = math.atan2( sy,      M[i,i])
            az = 0.0
    else:
        cy = math.sqrt(M[i,i]*M[i,i] + M[j,i]*M[j,i])
        if cy > _EPS:
            ax = math.atan2( M[k,j],  M[k,k])
            ay = math.atan2(-M[k,i],  cy)
            az = math.atan2( M[j,i],  M[i,i])
        else:
            ax = math.atan2(-M[j,k],  M[j,j])
            ay = math.atan2(-M[k,i],  cy)
            az = 0.0

    if parity: ax, ay, az = -ax, -ay, -az
    if frame: ax, az = az, ax
    return ax, ay, az

def quaternion_from_euler(ai, aj, ak, axes):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak -- Euler's roll, pitch and yaw angles
    axes -- One of 24 axis sequences as string or encoded tuple

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame: ai, ak = ak, ai
    if parity: aj = -aj

    ti = ai*0.5
    tj = aj*0.5
    tk = ak*0.5
    ci = math.cos(ti)
    si = math.sin(ti)
    cj = math.cos(tj)
    sj = math.sin(tj)
    ck = math.cos(tk)
    sk = math.sin(tk)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = numpy.empty((4,), dtype=numpy.float64)
    if repetition:
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
        q[3] = cj*(cc - ss)
    else:
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
        q[3] = cj*cc + sj*ss

    if parity: q[j] *= -1
    return q

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    quaternion -- Sequence of x, y, z, w
    axes -- One of 24 valid axis sequences as string or encoded tuple

    """
    return euler_from_rotation_matrix(
        rotation_matrix_from_quaternion(quaternion), axes)

def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis."""
    u = numpy.zeros((4,), dtype=numpy.float64)
    u[0:3] = axis[0:3]
    u *= math.sin(angle/2) / math.sqrt(numpy.dot(u, u))
    u[3] = math.cos(angle/2)
    return u

def rotation_matrix_from_quaternion(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = numpy.array(quaternion, dtype=numpy.float64)[0:4]
    nq = numpy.dot(q, q)
    if nq == 0.0:
        return numpy.identity(4, dtype=numpy.float64)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1,1]-q[2,2],     q[0,1]-q[2,3],     q[0,2]+q[1,3], 0.0),
        (    q[0,1]+q[2,3], 1.0-q[0,0]-q[2,2],     q[1,2]-q[0,3], 0.0),
        (    q[0,2]-q[1,3],     q[1,2]+q[0,3], 1.0-q[0,0]-q[1,1], 0.0),
        (              0.0,               0.0,               0.0, 1.0)
        ), dtype=numpy.float64)

def quaternion_from_rotation_matrix(matrix):
    """Return quaternion from rotation matrix."""
    q = numpy.empty((4,), dtype=numpy.float64)
    M = numpy.array(matrix, dtype=numpy.float64)[0:4,0:4]
    t = numpy.trace(M)
    if t > M[3,3]:
        q[3] = t
        q[2] = M[1,0] - M[0,1]
        q[1] = M[0,2] - M[2,0]
        q[0] = M[2,1] - M[1,2]
    else:
        i,j,k = 0,1,2
        if M[1,1] > M[0,0]:
            i,j,k = 1,2,0
        if M[2,2] > M[i,i]:
            i,j,k = 2,0,1
        t = M[i,i] - (M[j,j] + M[k,k]) + M[3,3]
        q[i] = t
        q[j] = M[i,j] + M[j,i]
        q[k] = M[k,i] + M[i,k]
        q[3] = M[k,j] - M[j,k]
    q *= 0.5 / math.sqrt(t * M[3,3])
    return q

def quaternion_multiply(q1, q0):
    """Multiply two quaternions."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return numpy.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0))

def quaternion_from_sphere_points(v0, v1):
    """Return quaternion from two points on unit sphere.

    v0 -- E.g. sphere coordinates of cursor at mouse down
    v1 -- E.g. current sphere coordinates of cursor

    """
    x, y, z = numpy.cross(v0, v1)
    return x, y, z, numpy.dot(v0, v1)

def quaternion_to_sphere_points(q):
    """Return two points on unit sphere from quaternion."""
    l = math.sqrt(q[0]*q[0] + q[1]*q[1])
    v0 = numpy.array((0.0, 1.0, 0.0) if l==0.0 else \
                     (-q[1]/l, q[0]/l, 0.0), dtype=numpy.float64)
    v1 = numpy.array((q[3]*v0[0] - q[3]*v0[1],
                      q[3]*v0[1] + q[3]*v0[0],
                      q[0]*v0[1] - q[1]*v0[0]), dtype=numpy.float64)
    if q[3] < 0.0:
        v0 *= -1.0
    return v0, v1


class Arcball(object):
    """Virtual Trackball Control."""

    def __init__(self, center=None, radius=1.0, initial=None):
        """Initializes virtual trackball control.

        center -- Window coordinates of trackball center
        radius -- Radius of trackball in window coordinates
        initial -- Initial quaternion or rotation matrix

        """
        self.axis = None
        self.center = numpy.zeros((3,), dtype=numpy.float64)
        self.place(center, radius)
        self.v0 = numpy.array([0., 0., 1.], dtype=numpy.float64)
        if initial is None:
            self.q0 = numpy.array([0., 0., 0., 1.], dtype=numpy.float64)
        else:
            try:
                self.q0 = quaternion_from_rotation_matrix(initial)
            except:
                self.q0 = initial
        self.qnow = self.q0

    def place(self, center=[0., 0.], radius=1.0):
        """Place Arcball, e.g. when window size changes."""
        self.radius = float(radius)
        self.center[0:2] = center[0:2]

    def click(self, position, axis=None):
        """Set axis constraint and initial window coordinates of cursor."""
        self.axis = axis
        self.q0 = self.qnow
        self.v0 = self._map_to_sphere(position, self.center, self.radius)

    def drag(self, position):
        """Return rotation matrix from updated window coordinates of
cursor."""
        v0 = self.v0
        v1 = self._map_to_sphere(position, self.center, self.radius)

        if self.axis is not None:
            v0 = self._constrain_to_axis(v0, self.axis)
            v1 = self._constrain_to_axis(v1, self.axis)

        t = numpy.cross(v0, v1)
        if numpy.dot(t, t) < _EPS:
            # v0 and v1 coincide. no additional rotation
            self.qnow = self.q0
        else:
            q1 = [t[0], t[1], t[2], numpy.dot(v0, v1)]
            self.qnow = quaternion_multiply(q1, self.q0)

        return rotation_matrix_from_quaternion(self.qnow)

    def _map_to_sphere(self, position, center, radius):
        """Map window coordinates to unit sphere coordinates."""
        v = numpy.array([position[0], position[1], 0.0],
dtype=numpy.float64)
        v -= center
        v /= radius
        v[1] *= -1
        l = numpy.dot(v, v)
        if l > 1.0:
            v /= math.sqrt(l) # position outside of sphere
        else:
            v[2] = math.sqrt(1.0 - l)
        return v

    def _constrain_to_axis(self, point, axis):
        """Return sphere point perpendicular to axis."""
        v = numpy.array(point, dtype=numpy.float64)
        a = numpy.array(axis, dtype=numpy.float64, copy=True)
        a /= numpy.dot(a, v)
        v -= a # on plane
        n = numpy.dot(v, v)
        if n > 0.0:
            v /= math.sqrt(n)
            return v
        if a[2] == 1.0:
            return numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
        v[:] = -a[1], a[0], 0.0
        v /= math.sqrt(numpy.dot(v, v))
        return v

    def _nearest_axis(self, point, *axes):
        """Return axis, which arc is nearest to point."""
        nearest = None
        max = -1.0
        for axis in axes:
            t = numpy.dot(self._constrain_to_axis(point, axis), point)
            if t > max:
                nearest = axis
                max = d
        return nearest

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = { # axes string -> (inner axis, parity, repetition, frame)
    "sxyz": (0, 0, 0, 0), "sxyx": (0, 0, 1, 0), "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0), "syzx": (1, 0, 0, 0), "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0), "syxy": (1, 1, 1, 0), "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0), "szyx": (2, 1, 0, 0), "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1), "rxyx": (0, 0, 1, 1), "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1), "rxzy": (1, 0, 0, 1), "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1), "ryxy": (1, 1, 1, 1), "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1), "rxyz": (2, 1, 0, 1), "rzyz": (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

try:
    # import faster functions from C extension module
    import _vlfdlib
    rotation_matrix_from_quaternion = _vlfdlib.quaternion_to_matrix
    quaternion_from_rotation_matrix = _vlfdlib.quaternion_from_matrix
    quaternion_multiply = _vlfdlib.quaternion_multiply
except ImportError:
    pass

def test_transformations_module(*args, **kwargs):
    """Test transformation module."""
    v = numpy.array([6.67,3.69,4.82], dtype=numpy.float64)
    I = numpy.identity(4, dtype=numpy.float64)
    T = translation_matrix(-v)
    M = mirror_matrix([0,0,0], v)
    R = rotation_matrix(math.pi, [1,0,0], point=v)
    P = projection_matrix(-v, v, perspective=10*v)
    P = projection_matrix(-v, v, direction=v)
    P = projection_matrix(-v, v)
    S = scaling_matrix(0.25, v, direction=None) # reduce size
    S = scaling_matrix(-1.0) # point symmetry
    S = scaling_matrix(10.0)
    O = orthogonalization_matrix(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    assert numpy.allclose(S, O)

    angles = (1, -2, 3)
    for axes in sorted(_AXES2TUPLE.keys()):
        assert axes == _TUPLE2AXES[_AXES2TUPLE[axes]]
        Me = rotation_matrix_from_euler(axes=axes, *angles)
        em = euler_from_rotation_matrix(Me, axes)
        Mf = rotation_matrix_from_euler(axes=axes, *em)
        assert numpy.allclose(Me, Mf)

    axes = 'sxyz'
    euler = (0.0, 0.2, 0.0)
    qe = quaternion_from_euler(axes=axes, *euler)
    Mr = rotation_matrix(0.2, [0,1,0])
    Me = rotation_matrix_from_euler(axes=axes, *euler)
    Mq = rotation_matrix_from_quaternion(qe)
    assert numpy.allclose(Mr, Me)
    assert numpy.allclose(Mr, Mq)
    em = euler_from_rotation_matrix(Mr, axes)
    eq = euler_from_quaternion(qe, axes)
    assert numpy.allclose(em, eq)
    qm = quaternion_from_rotation_matrix(Mq)
    assert numpy.allclose(qe, qm)

    assert numpy.allclose(
        rotation_matrix_from_quaternion(
            quaternion_about_axis(1.0, [1,-0.5,1])),
        rotation_matrix(1.0, [1,-0.5,1]))

    ball = Arcball([320,320], 320)
    ball.click([500,250])
    R = ball.drag([475, 275])
    assert numpy.allclose(R, [[ 0.9787566,  0.03798976, -0.20147528, 0.],
                              [-0.06854143, 0.98677273, -0.14690692, 0.],
                              [ 0.19322935, 0.15759552,  0.9684142,  0.],
                              [ 0.,         0.,          0.,         1.]])

if __name__ == "__main__":
    test_transformations_module()

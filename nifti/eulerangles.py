''' Module implementing Euler angle rotations and their conversions

See: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134

See also:

* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html

Euler angles for rotation in 3 dimensions consist of three rotations.
Following Diebel above, they follow the sequence:

#. rotation of angle ``psi`` around axis ``k``, followed by
#. rotation of angle ``theta`` around axis ``j`` followed by
#. rotation of angle ``phi`` around axis ``i``.

Given a triplet of angles - or *Euler axis vector* - [ ``phi``, ``theta``,
``psi`` ], we need to know the respective axes of rotation [ ``i``,
``j``, ``k`` ] (remember that the "``psi`` around ``k``" rotation is
applied first).

There are various conventions mapping [ ``i``, ``j``, ``k`` ] to the
``x``, ``y`` and ``z`` axes; here we follow:

* ``i`` is the ``x`` axis
* ``j`` is the ``y`` axis
* ``k`` is the ``z`` axis

Thus (to repeat), the Euler axis vector [ ``phi``, ``theta``, ``psi`` ],
in our convention, leads to a rotation of ``psi`` around the ``z`` axis,
followed by a rotation of ``theta`` around the ``y`` axis, followed by a
rotation of ``psi`` around the ``x`` axis.

This convention is known as Cardan angles, or Tait-Bryan angles.

We also specify, for our rotations, that the axes are static (stay in
the same place when the rotations are applied, with the body moving
within the axes) and therefore not dynamic (dynamic means that the axes
move with the body).

'''

import numpy as np


def euler2mat(x=0, y=0, z=0):
    ''' Return matrices for give rotations around z, y and x axes
    '''
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
                [[cosz, sinz, 0],
                 [-sinz, cosz, 0],
                 [0, 0, 1]]))
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
                 [0, -sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms)
    return np.eye(4)


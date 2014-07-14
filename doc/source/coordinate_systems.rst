##############################
Coordinate systems and affines
##############################

A nibabel (and nipy) image is the association of three things:

* The *image data array*: a 3D or 4D *array* of image data
* An *affine array* that tells you the position of the image array data in
  a *reference space*.
* *image metadata* (data about the data) describing the image, usually in the
  form of an image *header*.

This document describes how the *affine array* describes the position of the
image data in a reference space.  On the way we will define what we mean by
reference space, and the reference spaces that Nibabel uses.

*******************
Introducing Someone
*******************

We have scanned someone called "Someone", and we have a two MRI images of
their brain, a single EPI volume, and a structural scan.  In general we never
use the person's name in the image filenames, but we make an
exception in this case: 

* :download:`somones_epi.nii.gz </downloads/someones_epi.nii.gz>`.
* :download:`somones_anatomy.nii.gz </downloads/someones_anatomy.nii.gz>`.

We can load up the EPI image to get the image data array:

.. plot::
    :context:
    :nofigs:

    >>> import nibabel as nib
    >>> epi_img = nib.load('downloads/someones_epi.nii.gz')
    >>> epi_img_data = epi_img.get_data()
    >>> epi_img_data.shape
    (53, 61, 33)

Then we have a look at slices over the first, second and third dimensions of
the array.

.. plot::
    :context:

    >>> import matplotlib.pyplot as plt
    >>> def show_slices(slices):
    ...    """ Function to display row of image slices """
    ...    fig, axes = plt.subplots(1, len(slices))
    ...    for i, slice in enumerate(slices):
    ...        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    >>>
    >>> slice_0 = epi_img_data[26, :, :]
    >>> slice_1 = epi_img_data[:, 30, :]
    >>> slice_2 = epi_img_data[:, :, 16]
    >>> show_slices([slice_0, slice_1, slice_2])

We collected an anatomical image in the same session.  We can load that image
and look at slices in the three axes:

.. plot::
    :nofigs:
    :include-source: false

    # This guy just closes the previous figure so we don't get it twice
    plt.close('all')

.. plot::
    :context:

    >>> anat_img = nib.load('downloads/someones_anatomy.nii.gz')
    >>> anat_img_data = anat_img.get_data()
    >>> anat_img_data.shape
    (57, 67, 56)
    >>> show_slices([anat_img_data[28, :, :],
    ...              anat_img_data[:, 33, :],
    ...              anat_img_data[:, :, 28]])

As is usually the case, we had a different field of view for the anatomical
scan, and so the anatomical image has a different shape, size, and orientation
in the magnet.

*********************************************************
Voxel coordinates are coordinates in the image data array
*********************************************************

As y'all know, a voxel is a pixel with volume.

In the code above, ``slice_0`` from the EPI data is a 2D slice from a 3D
image.  The plot of the EPI slices displays the slices in grayscale (graded
between black for the minimum value, white for the maximum).  Each pixel in
the slice grayscale image also represents a voxel, because this 2D image
represents a slice from the 3D image with a certain thickness.

The 3D array is therefore also a voxel array.  As for any array, we can select
particular values by indexing.  For example, we can get the value for the
middle voxel in the EPI data array like this:

.. plot::
    :context:
    :nofigs:

    >>> n_i, n_j, n_k = epi_img_data.shape
    >>> center_i = (n_i - 1) / 2.
    >>> center_j = (n_j - 1) / 2.
    >>> center_k = (n_k - 1) / 2.
    >>> center_i, center_j, center_k
    (26.0, 30.0, 16.0)
    >>> center_vox_value = epi_img_data[center_i, center_j, center_k]
    >>> center_vox_value
    81.549287796020508

The values (26, 30, 16) are indices into the data array ``epi_img_data``. (26,
30, 16) is therefore a 'voxel coordinate' - a coordinate into the voxel array.

A coordinate is a set of numbers giving positions relative to a set of *axes*.
In this case 26 is a position on the first array axis, where the axis is of
length ``epi_img_data.shape[0]``, and therefore goes from 0 to 52
(``epi_img_data.shape == (53, 61, 33)``).  Similarly 30 gives a position on
the second axis (0 to 60) and 16 is the position on the third axis (0 to 32).

*************************************
Voxel coordinates and points in space
*************************************

The voxel coordinate tells us almost nothing about where the data came from
in terms of position in the scanner.  For example, let's say we have the voxel
coordinate (26, 30, 16).  Without more information we have no idea whether
this voxel position is on the left or right of the brain, or came from the
left or right of the scanner.

This is because the scanner allows us to collect voxel data in almost any
arbitrary position and orientation within the magnet.

In the case of Someone's EPI, we took transverse slices at a moderate angle to
the floor to ceiling direction.  This localizer image from the scanner console
has a red box that shows the position of the slice block for
``someones_epi.nii.gz`` and a blue box for the slice block of
``someones_anatomy.nii.gz``:

.. image:: /images/localizer.png

The localizer is oriented to the magnet, so that the left and right borders of
the image are parallel to the floor of the scanner room, with the left border
being towards the floor and the right border towards the ceiling.

You will see from the labels on the localizer that the center of the EPI voxel
data block (at 26, 30, 16 in ``epi_img_data``) is not quite at the center of
magnet bore (the magnet *isocenter*).

We have an anatomical and an EPI scan, and later on we will surely want to be
able to relate the data from ``someones_epi.nii.gz`` to
``someones_anatomy.nii.gz``.  We can't easily do this at the moment, because
we collected the anatomical image with a different field of view and
orientation to the EPI image, so the voxel coordinates in the EPI image refer
to different locations in the space to the voxel coordinates in the anatomical
image.

We solve this problem by keeping track of the relationship of voxel
coordinates to some *reference space*.  In particular, the *affine array*
stores the relationship between voxel coordinates in the image data array and
coordinates in the reference space.  We store the relationship of voxel
coordinates from ``someones_epi.nii.gz`` and the reference space, and also the
(different) relationship of voxel coordinates in ``someones_anatomy.nii.gz``
to the *same* reference space.  Because we know the relationship of (voxel
coordinates to the reference space) for both images, we can use this
information to relate voxel coordinates in ``someones_epi.nii.gz`` to spatially
equivalent voxel coordinates in ``someones_anatomy.nii.gz``.

*******************
The reference space
*******************

What does "space" mean in the phrase "reference space"?  The space is defined
by a set of axes. For our 3D spatial world, it is a set of 3 independent axes.

We can decide what space we want to use, by choosing these axes.  We need to
choose the origin of the axes, their direction and their units.

To start with, we are going to use the following reference axes:

* The origin of the axes is at the magnet isocenter. This is coordinate (0, 0,
  0) in our reference space. All three axes pass through the isocenter.
* The units for all three axes are millimeters.
* The first axis goes from negative values to the left of isocenter to
  positive values to the right of isocenter. Left is the subject's left, and
  also the left of an observer behind the scanner looking down the bore at the
  top of the subject's head. The axis is parallel to the floor of the scanning
  room. Call this the *X* axis.
* The second axis goes from negative values below isocenter to positive values
  above isocenter.  "Below" means towards the floor, and "above" means towards
  the ceiling. The axis is parallel to the left and right walls of the scanner
  room. Call this the *Y* axis. The Y axis is at right angles to the X and Z
  axes.
* The third axis goes from negative values towards the foot of the scanner bed
  to positive values at the opposite side of the magnet bore.  The axis is
  parallel to the floor of the scanner room and to the scanner bed.  Call this
  the *Z* axis.  The Z axis is at right angles to the X and Y axes.

This convention is usually called *scanner RAS*, because the origin comes from
the scanner isocenter, the axes are oriented relative to the magnet, and the
X, Y, Z axes go from left to Right; posterior to Anterior; and inferior to
Superior respectively, relative to the subject in the scanner.  Yes, it is
confusing that right, anterior, superior refer to the subject rather than the
scanner, but that's the convention we are stuck with.

Although it is common to call this convention "RAS", it is not quite
universal, because some use "R" in "RAS" to mean that the X axis *starts* on
the right, rather than ends on the right. To be safe, we'll call this
convention "scanner RAS+", meaning that Right, Anterior, Posterior are all
positive values on these axes.

Some people also use "right" to mean the right hand side when an observer
looks at the front of the scanner, from the foot of the subject.
Unfortunately, this means that you have to read people's definitions carefully
when you are trying to work out what space they are referring to. We nibabel /
nipy agree with most other brain imaging folk in that we always use "right" to
mean the subject's right.

****************************************************
The affine matrix as a transformation between spaces
****************************************************

So far we've dealt with voxel coordinates, which are indices into the image
data array.

Now we have defined the reference space, we can define coordinates in that
space too.  For example, coordinate (0, 0, 0) would be at the magnet
isocenter, and (10, -5, -3) would be 10 millimeters to the right of isocenter,
5mm posterior (towards the floor) and 3mm inferior (towards the foot of the
scanner bed).

Now the question is, how do we record the relationship of voxel coordinates to
coordinates in the reference space?

Specifically, if we have some voxel coordinate $(i, j, k)$, how do we generate
the reference space coordinate $(x, y, z)$?

Imagine we had solved this, and we had some function $f$ that accepts a voxel
coordinate and returns a coordinate in the reference space:

.. math::

    (x, y, z) = f(i, j, k)

$f$ is a function that transforms coordinates in voxel space (voxel
coordinates) into coordinates in the reference space (reference coordinates).
So our question is, what is the function $f$?

In general $f$ could be a complicated non-linear function, but in practice, we
know that the scanner collects data on a regular grid.  This means that the
relationship between $(i, j, k)$ and $(x, y, z)$ is linear (actually
*affine*), and can be encoded with linear (actually affine) transformations
comprising translations, rotations and zooms (`wikipedia linear transform`_,
`wikipedia affine transform`_).

A rotation in three dimensions can be represented as a 3 by 3 *rotation
matrix* (`wikipedia rotation matrix`_).  For example, here's a rotation by
$\theta$ radians around the third array axis:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} =
    \begin{bmatrix}
    \cos(\theta) &  -\sin(\theta) & 0 \\
    \sin(\theta) & \cos(\theta) & 0 \\
    0 & 0 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix}

This is a rotation by $\phi$ radians around the second array axis:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} =
    \begin{bmatrix}
    \cos(\phi) & 0 & \sin(\phi) \\
    0 & 1 & 0 \\
    -\sin(\phi) & 0 & \cos(\phi) \\
    \end{bmatrix}
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix}

Scaling (zooming) in three dimensions can be represented by a diagonal 3 by 3
matrix.  Here's how to zoom the first dimension by $p$, the second by $q$ and
the third by $r$ units:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} =
    \begin{bmatrix}
    p & 0 & 0 \\
    0 & q & 0 \\
    0 & 0 & r \\
    \end{bmatrix}
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix}

Rotations and zoom matrices can by combined by matrix multiplication.  Here's
a rotation of $\theta$ radians around the third axis followed by a rotation of
$\phi$ radians around the second axis, followed by scaling of $p, q, r$ units:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} =
    \begin{bmatrix}
    p & 0 & 0 \\
    0 & q & 0 \\
    0 & 0 & r \\
    \end{bmatrix}
    \begin{bmatrix}
    \cos(\phi) & 0 & \sin(\phi) \\
    0 & 1 & 0 \\
    -\sin(\phi) & 0 & \cos(\phi) \\
    \end{bmatrix}
    \begin{bmatrix}
    \cos(\theta) &  -\sin(\theta) & 0 \\
    \sin(\theta) & \cos(\theta) & 0 \\
    0 & 0 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix}

This can also be written:

.. math::

    M =
    \begin{bmatrix}
    p & 0 & 0 \\
    0 & q & 0 \\
    0 & 0 & r \\
    \end{bmatrix}
    \begin{bmatrix}
    \cos(\phi) & 0 & \sin(\phi) \\
    0 & 1 & 0 \\
    -\sin(\phi) & 0 & \cos(\phi) \\
    \end{bmatrix}
    \begin{bmatrix}
    \cos(\theta) &  -\sin(\theta) & 0 \\
    \sin(\theta) & \cos(\theta) & 0 \\
    0 & 0 & 1 \\
    \end{bmatrix}

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} = M
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix}

This might be obvious because the matrix multiplication is the result of
applying each transformation in turn on the coordinates output from the
previous transformation. Combining the transformations works because matrix
multiplication is associative.

A translation in three dimensions can be represented as a length 3 vector to
be added to the length 3 coordinate.  For example, a translation of $a$ units
on the first axis, $b$ on the second and $c$ on the third might be written
as:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} =
    \begin{bmatrix}
    a \\
    b \\
    c \\
    \end{bmatrix} +
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix}

We can write our function $f$ as a combination of matrix multiplication by
some 3 by 3 rotation / zoom matrix $M$ followed by addition of a 3 by 1
translation vector $(a, b, c)$

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} = M
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix} +
    \begin{bmatrix}
    a\\
    b\\
    c\\
    \end{bmatrix}

We could record the parameters necessary for $f$ as the 3 by 3 matrix, $M$
and the 3 by 1 vector $(a, b, c)$.

In fact, the 4 by 4 image *affine array* does include exactly this
information. If $m_{ij}$ is the value in row $i$ column $j$ of matrix $M$,
then the image affine matrix $A$ is:

.. math::

    A =
    \begin{bmatrix}
    m_{11} & m_{12} & m_{13} & a \\
    m_{21} & m_{22} & m_{23} & b \\
    m_{31} & m_{32} & m_{33} & c \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}

Why the extra row of $[0, 0, 0, 1]$?  Wee need this row because we have
rephrased the combination of rotations / zooms and translations as a
transformation in *homogenous coordinates* (see `wikipedia homogenous
coordinates`_).  This is a trick that allows us to put the translation part
into the same matrix as the rotations / zooms, so that both translations and
rotations / zooms can be applied by matrix multiplication.  In order to make
this work, we have to add an extra 1 to our input and output coordinate
vectors:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    1\\
    \end{bmatrix} =
    \begin{bmatrix}
    m_{11} & m_{12} & m_{13} & a \\
    m_{21} & m_{22} & m_{23} & b \\
    m_{31} & m_{32} & m_{33} & c \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
    i\\
    j\\
    k\\
    1\\
    \end{bmatrix}

This results in the same transformation as applying $M$ and $(a, b, c)$
separately. One advantage of encoding transformations this way is that we can
combine two sets of [rotations, zooms, translations] by matrix multiplication
of the two corresponding affine matrices.

In practice, although it is common to combine 3D transformations using 4 by 4
affine matrices, we nearly always *apply* the transformations by breaking up
the affine matrix into its component $M$ matrix and $(a, b, c)$ vector and
doing:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    \end{bmatrix} = M
    \begin{bmatrix}
    i\\
    j\\
    k\\
    \end{bmatrix} +
    \begin{bmatrix}
    a\\
    b\\
    c\\
    \end{bmatrix}

As long as the last row of the 4 by 4 is $[0, 0, 0, 1]$, applying the
transformations in this way is mathematically the same as using the full 4 by
4 form, without the inconvenience of adding the extra 1 to our input and
output vectors.

*********************************************************************
The inverse of the affine gives the mapping from scanner to voxel
*********************************************************************

The affine arrays we have described so far have another pleasant property |--|
they are usually invertible.  As y'all know, the inverse of a matrix $A$ is
the matrix $A^{-1}$ such that $I = A^{-1} A$, where $I$ is the identity
matrix.  Put another way:

.. math::

    \begin{bmatrix}
    x\\
    y\\
    z\\
    1\\
    \end{bmatrix} = A
    \begin{bmatrix}
    i\\
    j\\
    k\\
    1\\
    \end{bmatrix}

    \begin{bmatrix}
    i\\
    j\\
    k\\
    1\\
    \end{bmatrix} = A^{-1}
    \begin{bmatrix}
    x\\
    y\\
    z\\
    1\\
    \end{bmatrix}

That means that the inverse of the affine matrix gives the transformation from
scanner RAS+ coordinates to voxel coordinates in the image data.

Now imagine we have affine array $A$ for ``someones_epi.nii.gz``, and affine array
$B$ for ``someones_anatomy.nii.gz``.  $A$ gives the mapping from voxels in the
image data array of ``someones_epi.nii.gz`` to millimeters in scanner RAS+.  $B$
gives the mapping from voxels in image data array of
``someones_anatomy.nii.gz`` to *the same* scanner RAS+.  Now let's say we have
a particular voxel coordinate $(i, j, k)$ in the data array of
``someones_epi.nii.gz``, and we want to find the voxel in
``someones_anatomy.nii.gz`` that is in the same spatial position.  Call this
matching voxel coordinate $(i', j', k')$ . We first apply the transform from
``someones_epi.nii.gz`` voxels to scanner RAS+ ($A$) and then apply the transform
from scanner RAS+ to voxels in ``someones_anatomy.nii.gz`` ($B^{-1}$):

.. math::

    \begin{bmatrix}
    i'\\
    j'\\
    k'\\
    1\\
    \end{bmatrix} = B^{-1} A
    \begin{bmatrix}
    i\\
    j\\
    k\\
    1\\
    \end{bmatrix}

*********************
The affine by example
*********************

We can get the affine from the nibabel image object.  Here is the affine for
the EPI scan:

.. plot::
    :context:
    :nofigs:

    >>> import numpy as np
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> epi_img.affine
    array([[  3.   ,   0.   ,   0.   , -78.   ],
           [  0.   ,   2.866,  -0.887, -76.   ],
           [  0.   ,   0.887,   2.866, -64.   ],
           [  0.   ,   0.   ,   0.   ,   1.   ]])

As you see, the last row is $[0, 0, 0, 1]$

For simplicity of application, we split this into $M$ and $(a, b, c)$:

.. plot::
    :context:
    :nofigs:

    >>> M = epi_img.affine[:3, :3]
    >>> abc = epi_img.affine[:3, 3]

Then we can define our function $f$:

.. plot::
    :context:
    :nofigs:

    >>> def f(i, j, k):
    ...    """ Return X, Y, Z coordinates for i, j, k """
    ...    return M.dot((i, j, k)) + abc

The labels on the localizer image give the impression that the voxel
coordinate for center of ``someones_epi.nii.gz`` was a little above the magnet
isocenter.  Now we can check:

.. plot::
    :context:
    :nofigs:

    >>> epi_vox_center = (np.array(epi_img_data.shape) - 1) / 2.
    >>> f(epi_vox_center[0], epi_vox_center[1], epi_vox_center[2])
    array([ 0.   , -4.205,  8.453])

That means the center of the image field of view is at the isocenter of the
magnet on the left to right axis, and is around 4.2mm posterior to the
isocenter and ~8.5 mm above the isocenter.

We get the same result from applying the affine directly instead of using $M$
and $(a, b, c)$ in our function. As above, we need to add a 1
to the end of the vector to apply the 4 by 4 affine matrix.

.. plot::
    :context:
    :nofigs:

    >>> epi_img.affine.dot(list(epi_vox_center) + [1])
    array([ 0.   , -4.205,  8.453,  1.   ])

We can apply this affine function to give the position of any voxel, relative
to the scanner RAS+ reference space.

We can use matrix inversion on the affines to map between voxels in the EPI
image and voxels in the structural image.

.. plot::
    :context:
    :nofigs:

    >>> import numpy.linalg as npl
    >>> def vox_epi2anat(i, j, k):
    ...    """ voxel coordinates in structural for input EPI coords """
    ...    anat_vox2anat_mm = anat_img.affine
    ...    anat_mm2anat_vox = npl.inv(anat_vox2anat_mm)
    ...    epi_vox2epi_mm = epi_img.affine
    ...    epi_vox2anat_vox = anat_mm2anat_vox.dot(epi_vox2epi_mm)
    ...    vv_M = epi_vox2anat_vox[:3, :3]
    ...    vv_abc = epi_vox2anat_vox[:3, 3]
    ...    return vv_M.dot((i, j, k)) + vv_abc

So, what is the voxel coordinate in the anatomical corresponding to the voxel
center of the EPI image?

.. plot::
    :context:
    :nofigs:

    >>> vox_epi2anat(epi_vox_center[0], epi_vox_center[1], epi_vox_center[2])
    array([ 28.364,  31.562,  36.165])


The voxel coordinate of the center voxel of the anatomical image is:

.. plot::
    :context:
    :nofigs:

    >>> anat_vox_center = (np.array(anat_img_data.shape) - 1) / 2.
    >>> anat_vox_center
    array([ 28. ,  33. ,  27.5])

The voxel location in the anatomical image that matches the center voxel of the EPI image is nearly
exactly half way across the first axis, a voxel or two back from the
anatomical voxel center on the second axis, and about 9 voxels above the
anatomical voxel center.  We can check the localizer image by eye to see
whether this makes sense, by seeing how the red EPI field of view center
relates to the blue anatomical field of view center and the blue anatomical
image field of view.

**********************
Other reference spaces
**********************

The scanner RAS+ reference space is a "real-world" space, in the sense that a
coordinate in this space refers to a position in the real world, in a
particular scanner in a particular room.

Imagine that we used some fancy software to register ``someones_epi.nii.gz``
to a template image, such as the Montreal Neurological Institute (MNI)
template brain.  The registration has moved the voxels around in complicated
ways |--| the image has changed shape to match the template brain.  We
probably do not want to know how the voxel locations relate to the original
scanner, but how they relate to the template brain. So, what reference space
should we use?

In this case we use a space defined in terms of the template brain |--| the MNI
reference space.

* The origin (0, 0, 0) point is defined to be the point that the anterior
  commissure of the MNI template brain crosses the midline (the AC point).
* Axis units are millimeters.
* The Y axis follows the midline of the MNI brain between the left and right
  hemispheres, going from posterior (negative) to anterior (positive), passing
  through the AC point.  The template defines this line.
* The Z axis is at right angles to the Y axis, going from inferior (negative)
  to superior (positive), with the superior part of the line passing between
  the two hemispheres.
* The X axis is a line going from the left side of the brain (negative) to
  right side of the brain (positive), passing through the AC point, and at
  right angles to the Y and Z axes.

These axes are defined with reference to the template.  The exact position of
the Y axis, for example, is somewhat arbitrary, as is the definition of the
origin. Left and right are left and right as defined by the template. These
are the axes and the space that MNI defines for its template.

A coordinate in this reference system gives a position relative to the
particular brain template.  It is not a real-world space because it does not
refer to any particular place but to a position relative to a template.

The axes are still left to right, posterior to anterior and inferior to
superior.  This is still an RAS+ space |--| the MNI RAS+ space.

An image aligned to this template will therefore have an affine giving the
relationship between voxels in the aligned image and the MNI RAS+ space.

There are other reference spaces. For example, we might align an image to the
Talairach atlas brain.  This brain has a different shape and size than the MNI
brain.  The origin is the AC point, but the Y axis passes through the point
that the posterior commissure crosses the midline (the PC point), giving a
slightly different trajectory from the MNI Y axis.  Like the MNI RAS+ space,
the Talairach axes also run left to right, posterior to anterior and inferior
superior, so this is the Talairach RAS+ space.

***********************
Input and output spaces
***********************

In the examples thus far we have talked about voxel space and output RAS+
spaces such as scanner RAS+ and MNI RAS+.  The affine maps input voxel
coordinates to output RAS+ coordinates.

There are conventions other than RAS+ for the output space.  For example,
DICOM files map input voxel coordinates to coordinates in scanner LPS+ space.
LPS+ space has the origin at the scanner isocenter, but the X axis goes from
right to Left, the Y axis goes from anterior to Posterior, and the Z axis
goes from inferior to Superior.  A positive X coordinate in this space would
mean the point was to the *left* of magnet isocenter.

****************************************
Nibabel always uses an RAS+ output space
****************************************

Nibabel images always use RAS+ output coordinates, regardless of the preferred
output coordinates of the underlying format.  For example, we convert affines
for DICOM images to output RAS+ coordinates instead of LPS+ coordinates. We
chose this convention because it is by the far the most popular in
neuroimaging; it is the standard used by NIfTI_ and MINC_ formats.

Nibabel does not enforce a particular RAS+ space. For example, NIfTI images
contain codes that specify whether the affine maps to scanner or MNI or
Talairach RAS+ space.  For the moment, you have to consult the specifics of
each format to find which RAS+ space the affine maps to.

.. include:: links_names.txt

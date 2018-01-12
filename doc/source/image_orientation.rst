#######################
Image voxel orientation
#######################

It is sometimes useful to know the approximate world-space orientations of the
image voxel axes.

See :doc:`coordinate_systems` for background on voxel and world axes.

For example, let's say we had an image with an identity affine:

>>> import numpy as np
>>> import nibabel as nib
>>> affine = np.eye(4)  # identity affine
>>> voxel_data = np.random.normal(size=(10, 11, 12))
>>> img = nib.Nifti1Image(voxel_data, affine)

Because the affine is an identity affine, the voxel axes align with the world
axes.  By convention, nibabel world axes are always in RAS+ orientation (left
to Right, posterior to Anterior, inferior to Superior).

Let's say we took a single line of voxels along the first voxel axis:

>>> single_line_axis_0 = voxel_data[:, 0, 0]

The first voxel axis is aligned to the left to Right world axes.  This means
that the first voxel is towards the left of the world, and the last voxel is
towards the right of the world.

Here is a single line in the second axis:

>>> single_line_axis_1 = voxel_data[0, :, 0]

The first voxel in this line is towards the posterior of the world, and the
last towards the anterior.

>>> single_line_axis_2 = voxel_data[0, 0, :]

The first voxel in this line is towards the inferior of the world, and the
last towards the superior.

This image therefore has RAS+ *voxel* axes.

In other cases, it is not so obvious what the orientations of the axes are.
For example, here is our example NIfTI 1 file again:

>>> import os
>>> from nibabel.testing import data_path
>>> example_file = os.path.join(data_path, 'example4d.nii.gz')
>>> img = nib.load(example_file)

Here is the affine (to two digits decimal precision):

>>> np.set_printoptions(precision=2, suppress=True)
>>> img.affine
array([[ -2.  ,   0.  ,   0.  , 117.86],
       [ -0.  ,   1.97,  -0.36, -35.72],
       [  0.  ,   0.32,   2.17,  -7.25],
       [  0.  ,   0.  ,   0.  ,   1.  ]])

What are the orientations of the voxel axes here?

Nibabel has a routine to tell you, called ``aff2axcodes``.

>>> nib.aff2axcodes(img.affine)
('L', 'A', 'S')

The voxel orientations are nearest to:

#. First voxel axis goes from right to Left;
#. Second voxel axis goes from posterior to Anterior;
#. Third voxel axis goes from inferior to Superior.

Sometimes you may want to rearrange the image voxel axes to make them as close
as possible to RAS+ orientation.  We refer to this voxel orientation as
*canonical* voxel orientation, because RAS+ is our canonical world
orientation. Rearranging the voxel axes means reversing and / or reordering
the voxel axes.

You can do the arrangement with ``as_closest_canonical``:

>>> canonical_img = nib.as_closest_canonical(img)
>>> canonical_img.affine
array([[   2.  ,    0.  ,    0.  , -136.14],
       [   0.  ,    1.97,   -0.36,  -35.72],
       [  -0.  ,    0.32,    2.17,   -7.25],
       [   0.  ,    0.  ,    0.  ,    1.  ]])
>>> nib.aff2axcodes(canonical_img.affine)
('R', 'A', 'S')

.. include:: links_names.txt

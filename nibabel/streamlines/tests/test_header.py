import numpy as np

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal

from nibabel.streamlines.header import TractogramHeader


def test_streamlines_header():
    header = TractogramHeader()
    assert_true(header.nb_streamlines is None)
    assert_true(header.nb_scalars_per_point is None)
    assert_true(header.nb_properties_per_streamline is None)
    assert_array_equal(header.voxel_sizes, (1, 1, 1))
    assert_array_equal(header.to_world_space, np.eye(4))
    assert_equal(header.extra, {})

    # Modify simple attributes
    header.nb_streamlines = 1
    header.nb_scalars_per_point = 2
    header.nb_properties_per_streamline = 3
    assert_equal(header.nb_streamlines, 1)
    assert_equal(header.nb_scalars_per_point, 2)
    assert_equal(header.nb_properties_per_streamline, 3)

    # Modifying voxel_sizes should be reflected in to_world_space
    header.voxel_sizes = (2, 3, 4)
    assert_array_equal(header.voxel_sizes, (2, 3, 4))
    assert_array_equal(np.diag(header.to_world_space), (2, 3, 4, 1))

    # Modifying scaling of to_world_space should be reflected in voxel_sizes
    header.to_world_space = np.diag([4, 3, 2, 1])
    assert_array_equal(header.voxel_sizes, (4, 3, 2))
    assert_array_equal(header.to_world_space, np.diag([4, 3, 2, 1]))

    # Test that we can run __repr__ without error.
    repr(header)

import numpy as np
from nibabel.orientations import aff2axcodes
from collections import OrderedDict


class Field:
    """ Header fields common to multiple streamlines file formats.

    In IPython, use `nibabel.streamlines.Field??` to list them.
    """
    NB_STREAMLINES = "nb_streamlines"
    STEP_SIZE = "step_size"
    METHOD = "method"
    NB_SCALARS_PER_POINT = "nb_scalars_per_point"
    NB_PROPERTIES_PER_STREAMLINE = "nb_properties_per_streamline"
    NB_POINTS = "nb_points"
    VOXEL_SIZES = "voxel_sizes"
    DIMENSIONS = "dimensions"
    MAGIC_NUMBER = "magic_number"
    ORIGIN = "origin"
    VOXEL_TO_WORLD = "voxel_to_world"
    VOXEL_ORDER = "voxel_order"
    ENDIAN = "endian"


class StreamlinesHeader(object):
    def __init__(self):
        self._nb_streamlines = None
        self._nb_scalars_per_point = None
        self._nb_properties_per_streamline = None
        self._voxel_to_world = np.eye(4)
        self.extra = OrderedDict()

    @property
    def voxel_to_world(self):
        return self._voxel_to_world

    @voxel_to_world.setter
    def voxel_to_world(self, value):
        self._voxel_to_world = np.array(value, dtype=np.float32)

    @property
    def voxel_sizes(self):
        """ Get voxel sizes from voxel_to_world. """
        return np.sqrt(np.sum(self.voxel_to_world[:3, :3]**2, axis=0))

    @voxel_sizes.setter
    def voxel_sizes(self, value):
        scaling = np.r_[np.array(value), [1]]
        old_scaling = np.r_[np.array(self.voxel_sizes), [1]]
        # Remove old scaling and apply new one
        self.voxel_to_world = np.dot(np.diag(scaling/old_scaling), self.voxel_to_world)

    @property
    def voxel_order(self):
        """ Get voxel order from voxel_to_world. """
        return "".join(aff2axcodes(self.voxel_to_world))

    @property
    def nb_streamlines(self):
        return self._nb_streamlines

    @nb_streamlines.setter
    def nb_streamlines(self, value):
        self._nb_streamlines = int(value)

    @property
    def nb_scalars_per_point(self):
        return self._nb_scalars_per_point

    @nb_scalars_per_point.setter
    def nb_scalars_per_point(self, value):
        self._nb_scalars_per_point = int(value)

    @property
    def nb_properties_per_streamline(self):
        return self._nb_properties_per_streamline

    @nb_properties_per_streamline.setter
    def nb_properties_per_streamline(self, value):
        self._nb_properties_per_streamline = int(value)

    @property
    def extra(self):
        return self._extra

    @extra.setter
    def extra(self, value):
        self._extra = OrderedDict(value)

    def __eq__(self, other):
        return (np.allclose(self.voxel_to_world, other.voxel_to_world) and
                self.nb_streamlines == other.nb_streamlines and
                self.nb_scalars_per_point == other.nb_scalars_per_point and
                self.nb_properties_per_streamline == other.nb_properties_per_streamline and
                repr(self.extra) == repr(other.extra))  # Not the robust way, but will do!

    def __repr__(self):
        txt = "Header{\n"
        txt += "nb_streamlines: " + repr(self.nb_streamlines) + '\n'
        txt += "nb_scalars_per_point: " + repr(self.nb_scalars_per_point) + '\n'
        txt += "nb_properties_per_streamline: " + repr(self.nb_properties_per_streamline) + '\n'
        txt += "voxel_to_world: " + repr(self.voxel_to_world) + '\n'
        txt += "voxel_sizes: " + repr(self.voxel_sizes) + '\n'

        txt += "Extra fields: {\n"
        for key in sorted(self.extra.keys()):
            txt += "  " + repr(key) + ": " + repr(self.extra[key]) + "\n"

        txt += "  }\n"
        return txt + "}"

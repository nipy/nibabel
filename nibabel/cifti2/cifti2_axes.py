import numpy as np
from . import cifti2
from six import string_types, add_metaclass, integer_types
from operator import xor
import abc


def from_mapping(mim):
    """
    Parses the MatrixIndicesMap to find the appropriate CIFTI axis describing the rows or columns

    Parameters
    ----------
    mim : cifti2.Cifti2MatrixIndicesMap

    Returns
    -------
    subtype of Axis
    """
    return_type = {'CIFTI_INDEX_TYPE_SCALARS': Scalar,
                   'CIFTI_INDEX_TYPE_LABELS': Label,
                   'CIFTI_INDEX_TYPE_SERIES': Series,
                   'CIFTI_INDEX_TYPE_BRAIN_MODELS': BrainModel,
                   'CIFTI_INDEX_TYPE_PARCELS': Parcels}
    return return_type[mim.indices_map_to_data_type].from_mapping(mim)


def to_header(axes):
    """
    Converts the axes describing the rows/columns of a CIFTI vector/matrix to a Cifti2Header

    Parameters
    ----------
    axes : iterable[Axis]
        one or more axes describing each dimension in turn

    Returns
    -------
    cifti2.Cifti2Header
    """
    axes = list(axes)
    mims_all = []
    matrix = cifti2.Cifti2Matrix()
    for dim, ax in enumerate(axes):
        if ax in axes[:dim]:
            dim_prev = axes.index(ax)
            mims_all[dim_prev].applies_to_matrix_dimension.append(dim)
            mims_all.append(mims_all[dim_prev])
        else:
            mim = ax.to_mapping(dim)
            mims_all.append(mim)
            matrix.append(mim)
    return cifti2.Cifti2Header(matrix)


@add_metaclass(abc.ABCMeta)
class Axis(object):
    """
    Abstract class for any object describing the rows or columns of a CIFTI vector/matrix

    Mainly used for type checking.
    """

    @property
    def size(self):
        return len(self)

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Compares whether two Axes are equal

        Parameters
        ----------
        other : Axis
            other axis to compare to

        Returns
        -------
        False if the axes don't have the same type or if their content differs
        """
        pass

    @abc.abstractmethod
    def __add__(self, other):
        """
        Concatenates two Axes of the same type

        Parameters
        ----------
        other : Axis
            axis to be appended to the current one

        Returns
        -------
        Axis of the same subtype as self and other
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        """
        Extracts definition of single row/column or new Axis describing a subset of the rows/columns
        """


class BrainModel(Axis):
    """
    Each row/column in the CIFTI vector/matrix represents a single vertex or voxel

    This Axis describes which vertex/voxel is represented by each row/column.
    """

    def __init__(self, name, voxel=None, vertex=None, affine=None,
                 volume_shape=None, nvertices=None):
        """
        New BrainModel axes can be constructed by passing on the greyordinate brain-structure
        names and voxel/vertex indices to the constructor or by one of the
        factory methods:

        - :py:meth:`~BrainModel.from_mask`: creates surface or volumetric BrainModel axis
        from respectively 1D or 3D masks
        - :py:meth:`~BrainModel.from_surface`: creates a surface BrainModel axis

        The resulting BrainModel axes can be concatenated by adding them together.

        Parameters
        ----------
        name : str or np.ndarray
            brain structure name or (N, ) string array with the brain structure names
        voxel : np.ndarray
            (N, 3) array with the voxel indices (can be omitted for CIFTI files only
            covering the surface)
        vertex :  np.ndarray
            (N, ) array with the vertex indices (can be omitted for volumetric CIFTI files)
        affine : np.ndarray
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI files only
            covering the surface)
        volume_shape : Tuple[int, int, int]
            shape of the volume in which the voxels were defined (not needed for CIFTI files only
            covering the surface)
        nvertices : dict[String -> int]
            maps names of surface elements to integers (not needed for volumetric CIFTI files)
        """
        if voxel is None:
            if vertex is None:
                raise ValueError("At least one of voxel or vertex indices should be defined")
            nelements = len(vertex)
            self.voxel = np.full((nelements, 3), fill_value=-1, dtype=int)
        else:
            nelements = len(voxel)
            self.voxel = np.asanyarray(voxel, dtype=int)

        if vertex is None:
            self.vertex = np.full(nelements, fill_value=-1, dtype=int)
        else:
            self.vertex = np.asanyarray(vertex, dtype=int)

        if isinstance(name, string_types):
            name = [self.to_cifti_brain_structure_name(name)] * self.vertex.size
        self.name = np.asanyarray(name, dtype='U')

        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = {self.to_cifti_brain_structure_name(name): number
                              for name, number in nvertices.items()}

        for name in list(self.nvertices.keys()):
            if name not in self.name:
                del self.nvertices[name]

        is_surface = self.is_surface
        if is_surface.all():
            self.affine = None
            self.volume_shape = None
        else:
            if affine is None or volume_shape is None:
                raise ValueError("Affine and volume shape should be defined " +
                                 "for BrainModel containing voxels")
            self.affine = affine
            self.volume_shape = volume_shape

        if np.any(self.vertex[is_surface] < 0):
            raise ValueError('Undefined vertex indices found for surface elements')
        if np.any(self.voxel[~is_surface] < 0):
            raise ValueError('Undefined voxel indices found for volumetric elements')

        for check_name in ('name', 'voxel', 'vertex'):
            shape = (self.size, 3) if check_name == 'voxel' else (self.size, )
            if getattr(self, check_name).shape != shape:
                raise ValueError("Input {} has incorrect shape ({}) for BrainModel axis".format(
                        check_name, getattr(self, check_name).shape))

    @classmethod
    def from_mask(cls, mask, name='other', affine=None):
        """
        Creates a new BrainModel axis describing the provided mask

        Parameters
        ----------
        mask : np.ndarray
            all non-zero voxels will be included in the BrainModel axis
            should be (Nx, Ny, Nz) array for volume mask or (Nvertex, ) array for surface mask
        name : str
            Name of the brain structure (e.g. 'CortexRight', 'thalamus_left' or 'brain_stem')
        affine : np.ndarray
            (4, 4) array with the voxel to mm transformation (defaults to identity matrix)
            Argument will be ignored for surface masks

        Returns
        -------
        BrainModel which covers the provided mask
        """
        if affine is None:
            affine = np.eye(4)
        else:
            affine = np.asanyarray(affine)
        if affine.shape != (4, 4):
            raise ValueError("Affine transformation should be a 4x4 array or None, not %r" % affine)
        if mask.ndim == 1:
            return cls.from_surface(np.where(mask != 0)[0], mask.size, name=name)
        elif mask.ndim == 3:
            voxels = np.array(np.where(mask != 0)).T
            return cls(name, voxel=voxels, affine=affine, volume_shape=mask.shape)
        else:
            raise ValueError("Mask should be either 1-dimensional (for surfaces) or "
                             "3-dimensional (for volumes), not %i-dimensional" % mask.ndim)

    @classmethod
    def from_surface(cls, vertices, nvertex, name='Cortex'):
        """
        Creates a new BrainModel axis describing the vertices on a surface

        Parameters
        ----------
        vertices : np.ndarray
            indices of the vertices on the surface
        nvertex : int
            total number of vertices on the surface
        name : str
            Name of the brain structure (e.g. 'CortexLeft' or 'CortexRight')

        Returns
        -------
        BrainModel which covers (part of) the surface
        """
        cifti_name = cls.to_cifti_brain_structure_name(name)
        return cls(cifti_name, vertex=vertices,
                   nvertices={cifti_name: nvertex})

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new BrainModel axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        BrainModel
        """
        nbm = sum(bm.index_count for bm in mim.brain_models)
        voxel = np.full((nbm, 3), fill_value=-1, dtype=int)
        vertex = np.full(nbm, fill_value=-1, dtype=int)
        name = []

        nvertices = {}
        affine, shape = None, None
        for bm in mim.brain_models:
            index_end = bm.index_offset + bm.index_count
            is_surface = bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
            name.extend([bm.brain_structure] * bm.index_count)
            if is_surface:
                vertex[bm.index_offset: index_end] = bm.vertex_indices
                nvertices[bm.brain_structure] = bm.surface_number_of_vertices
            else:
                voxel[bm.index_offset: index_end, :] = bm.voxel_indices_ijk
                if affine is None:
                    shape = mim.volume.volume_dimensions
                    affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        return cls(name, voxel, vertex, affine, shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the brain model axis to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
        for name, to_slice, bm in self.iter_structures():
            is_surface = name in self.nvertices.keys()
            if is_surface:
                voxels = None
                vertices = cifti2.Cifti2VertexIndices(bm.vertex)
                nvertex = self.nvertices[name]
            else:
                voxels = cifti2.Cifti2VoxelIndicesIJK(bm.voxel)
                vertices = None
                nvertex = None
                if mim.volume is None:
                    affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, self.affine)
                    mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
            cifti_bm = cifti2.Cifti2BrainModel(
                    to_slice.start, len(bm),
                    'CIFTI_MODEL_TYPE_SURFACE' if is_surface else 'CIFTI_MODEL_TYPE_VOXELS',
                    name, nvertex, voxels, vertices
            )
            mim.append(cifti_bm)
        return mim

    def iter_structures(self):
        """
        Iterates over all brain structures in the order that they appear along the axis

        Yields
        ------
        tuple with 3 elements:
        - CIFTI brain structure name
        - slice to select the data associated with the brain structure from the tensor
        - brain model covering that specific brain structure
        """
        idx_start = 0
        start_name = self.name[idx_start]
        for idx_current, name in enumerate(self.name):
            if start_name != name:
                yield start_name, slice(idx_start, idx_current), self[idx_start: idx_current]
                idx_start = idx_current
                start_name = self.name[idx_start]
        yield start_name, slice(idx_start, None), self[idx_start:]

    @staticmethod
    def to_cifti_brain_structure_name(name):
        """
        Attempts to convert the name of an anatomical region in a format recognized by CIFTI

        This function returns:

        - the name if it is in the CIFTI format already
        - if the name is a tuple the first element is assumed to be the structure name while
          the second is assumed to be the hemisphere (left, right or both). The latter will default
          to both.
        - names like left_cortex, cortex_left, LeftCortex, or CortexLeft will be converted to
          CIFTI_STRUCTURE_CORTEX_LEFT

        see :py:func:`nibabel.cifti2.tests.test_name` for examples of
        which conversions are possible

        Parameters
        ----------
        name: (str, tuple)
            input name of an anatomical region

        Returns
        -------
        CIFTI2 compatible name

        Raises
        ------
        ValueError: raised if the input name does not match a known anatomical structure in CIFTI
        """
        if name in cifti2.CIFTI_BRAIN_STRUCTURES:
            return name
        if not isinstance(name, string_types):
            if len(name) == 1:
                structure = name[0]
                orientation = 'both'
            else:
                structure, orientation = name
                if structure.lower() in ('left', 'right', 'both'):
                    orientation, structure = name
        else:
            orient_names = ('left', 'right', 'both')
            for poss_orient in orient_names:
                idx = len(poss_orient)
                if poss_orient == name.lower()[:idx]:
                    orientation = poss_orient
                    if name[idx] in '_ ':
                        structure = name[idx + 1:]
                    else:
                        structure = name[idx:]
                    break
                if poss_orient == name.lower()[-idx:]:
                    orientation = poss_orient
                    if name[-idx - 1] in '_ ':
                        structure = name[:-idx - 1]
                    else:
                        structure = name[:-idx]
                    break
            else:
                orientation = 'both'
                structure = name
        if orientation.lower() == 'both':
            proposed_name = 'CIFTI_STRUCTURE_%s' % structure.upper()
        else:
            proposed_name = 'CIFTI_STRUCTURE_%s_%s' % (structure.upper(), orientation.upper())
        if proposed_name not in cifti2.CIFTI_BRAIN_STRUCTURES:
            raise ValueError('%s was interpreted as %s, which is not a valid CIFTI brain structure'
                             % (name, proposed_name))
        return proposed_name

    @property
    def is_surface(self):
        """
        (N, ) boolean array which is true for any element on the surface
        """
        return np.vectorize(lambda name: name in self.nvertices.keys())(self.name)

    _affine = None

    @property
    def affine(self):
        """
        Affine of the volumetric image in which the greyordinate voxels were defined
        """
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asanyarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value

    _volume_shape = None

    @property
    def volume_shape(self):
        """
        Shape of the volumetric image in which the greyordinate voxels were defined
        """
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError("Volume shape should be a tuple of length 3")
            if not all(isinstance(v, integer_types) for v in value):
                raise ValueError("All elements of the volume shape should be integers")
        self._volume_shape = value

    _name = None

    @property
    def name(self):
        """The brain structure to which the voxel/vertices of belong
        """
        return self._name

    @name.setter
    def name(self, values):
        self._name = np.array([self.to_cifti_brain_structure_name(name) for name in values])

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        if not isinstance(other, BrainModel) or len(self) != len(other):
            return False
        if xor(self.affine is None, other.affine is None):
            return False
        return (
            (self.affine is None or
             np.allclose(self.affine, other.affine) and
             self.volume_shape == other.volume_shape) and
            self.nvertices == other.nvertices and
            np.array_equal(self.name, other.name) and
            np.array_equal(self.voxel[~self.is_surface], other.voxel[~other.is_surface]) and
            np.array_equal(self.vertex[self.is_surface], other.vertex[other.is_surface])
        )

    def __add__(self, other):
        """
        Concatenates two BrainModels

        Parameters
        ----------
        other : BrainModel
            brain model to be appended to the current one

        Returns
        -------
        BrainModel
        """
        if not isinstance(other, BrainModel):
            return NotImplemented
        if self.affine is None:
            affine, shape = other.affine, other.volume_shape
        else:
            affine, shape = self.affine, self.volume_shape
            if other.affine is not None and (
                    not np.allclose(other.affine, affine) or
                    other.volume_shape != shape
            ):
                raise ValueError("Trying to concatenate two BrainModels defined " +
                                 "in a different brain volume")

        nvertices = dict(self.nvertices)
        for name, value in other.nvertices.items():
            if name in nvertices.keys() and nvertices[name] != value:
                raise ValueError("Trying to concatenate two BrainModels with inconsistent " +
                                 "number of vertices for %s" % name)
            nvertices[name] = value
        return self.__class__(
                np.append(self.name, other.name),
                np.concatenate((self.voxel, other.voxel), 0),
                np.append(self.vertex, other.vertex),
                affine, shape, nvertices
        )

    def __getitem__(self, item):
        """
        Extracts part of the brain structure

        Parameters
        ----------
        item : anything that can index a 1D array

        Returns
        -------
        If `item` is an integer returns a tuple with 3 elements:
        - boolean, which is True if it is a surface element
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from

        Otherwise returns a new BrainModel
        """
        if isinstance(item, integer_types):
            return self.get_element(item)
        if isinstance(item, string_types):
            raise IndexError("Can not index an Axis with a string (except for Parcels)")
        return self.__class__(self.name[item], self.voxel[item], self.vertex[item],
                              self.affine, self.volume_shape, self.nvertices)

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 3 elements
        - boolean, which is True if it is a surface element
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from
        """
        is_surface = self.name[index] in self.nvertices.keys()
        struct = self.vertex if is_surface else self.voxel
        return is_surface, struct[index], self.name[index]


class Parcels(Axis):
    """
    Each row/column in the CIFTI vector/matrix represents a parcel of voxels/vertices

    This Axis describes which parcel is represented by each row/column.

    Individual parcels can be accessed based on their name, using
    ``parcel = parcel_axis[name]``
    """

    def __init__(self, name, voxels, vertices, affine=None, volume_shape=None, nvertices=None):
        """
        Use of this constructor is not recommended. New Parcels axes can be constructed more easily
        from a sequence of BrainModel axes using :py:meth:`~Parcels.from_brain_models`

        Parameters
        ----------
        name : np.ndarray
            (N, ) string array with the parcel names
        voxels :  np.ndarray
            (N, ) object array each containing a sequence of voxels.
            For each parcel the voxels are represented by a (M, 3) index array
        vertices :  np.ndarray
            (N, ) object array each containing a sequence of vertices.
            For each parcel the vertices are represented by a mapping from brain structure name to
            (M, ) index array
        affine : np.ndarray
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI files only
            covering the surface)
        volume_shape : Tuple[int, int, int]
            shape of the volume in which the voxels were defined (not needed for CIFTI files only
            covering the surface)
        nvertices : dict[String -> int]
            maps names of surface elements to integers (not needed for volumetric CIFTI files)
        """
        self.name = np.asanyarray(name, dtype='U')
        self.voxels = np.asanyarray(voxels, dtype='object')
        self.vertices = np.asanyarray(vertices, dtype='object')
        self.affine = affine
        self.volume_shape = volume_shape
        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = {BrainModel.to_cifti_brain_structure_name(name): number
                              for name, number in nvertices.items()}

        for check_name in ('name', 'voxels', 'vertices'):
            if getattr(self, check_name).shape != (self.size, ):
                raise ValueError("Input {} has incorrect shape ({}) for Parcel axis".format(
                        check_name, getattr(self, check_name).shape))

    @classmethod
    def from_brain_models(cls, named_brain_models):
        """
        Creates a Parcel axis from a list of BrainModel axes with names

        Parameters
        ----------
        named_brain_models : List[Tuple[String, BrainModel]]
            list of (parcel name, brain model representation) pairs defining each parcel

        Returns
        -------
        Parcels
        """
        nparcels = len(named_brain_models)
        affine = None
        volume_shape = None
        all_names = []
        all_voxels = np.zeros(nparcels, dtype='object')
        all_vertices = np.zeros(nparcels, dtype='object')
        nvertices = {}
        for idx_parcel, (parcel_name, bm) in enumerate(named_brain_models):
            all_names.append(parcel_name)

            voxels = bm.voxel[~bm.is_surface]
            if voxels.shape[0] != 0:
                if affine is None:
                    affine = bm.affine
                    volume_shape = bm.volume_shape
                else:
                    if not np.allclose(affine, bm.affine) or (volume_shape != bm.volume_shape):
                        raise ValueError("Can not combine brain models defined in different " +
                                         "volumes into a single Parcel axis")
            all_voxels[idx_parcel] = voxels

            vertices = {}
            for name, _, bm_part in bm.iter_structures():
                if name in bm.nvertices.keys():
                    if name in nvertices.keys() and nvertices[name] != bm.nvertices[name]:
                        raise ValueError("Got multiple conflicting number of " +
                                         "vertices for surface structure %s" % name)
                    nvertices[name] = bm.nvertices[name]
                    vertices[name] = bm_part.vertex
            all_vertices[idx_parcel] = vertices
        return Parcels(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new Parcels axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Parcels
        """
        nparcels = len(list(mim.parcels))
        all_names = []
        all_voxels = np.zeros(nparcels, dtype='object')
        all_vertices = np.zeros(nparcels, dtype='object')

        volume_shape = None if mim.volume is None else mim.volume.volume_dimensions
        affine = None
        if mim.volume is not None:
            affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        nvertices = {}
        for surface in mim.surfaces:
            nvertices[surface.brain_structure] = surface.surface_number_of_vertices
        for idx_parcel, parcel in enumerate(mim.parcels):
            nvoxels = 0 if parcel.voxel_indices_ijk is None else len(parcel.voxel_indices_ijk)
            voxels = np.zeros((nvoxels, 3), dtype='i4')
            if nvoxels != 0:
                voxels[:] = parcel.voxel_indices_ijk
            vertices = {}
            for vertex in parcel.vertices:
                name = vertex.brain_structure
                vertices[vertex.brain_structure] = np.array(vertex)
                if name not in nvertices.keys():
                    raise ValueError("Number of vertices for surface structure %s not defined" %
                                     name)
            all_voxels[idx_parcel] = voxels
            all_vertices[idx_parcel] = vertices
            all_names.append(parcel.name)
        return cls(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the Parcel to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_PARCELS')
        if self.affine is not None:
            affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, matrix=self.affine)
            mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
        for name, nvertex in self.nvertices.items():
            mim.append(cifti2.Cifti2Surface(name, nvertex))
        for name, voxels, vertices in zip(self.name, self.voxels, self.vertices):
            cifti_voxels = cifti2.Cifti2VoxelIndicesIJK(voxels)
            element = cifti2.Cifti2Parcel(name, cifti_voxels)
            for name, idx_vertices in vertices.items():
                element.vertices.append(cifti2.Cifti2Vertices(name, idx_vertices))
            mim.append(element)
        return mim

    _affine = None

    @property
    def affine(self):
        """
        Affine of the volumetric image in which the greyordinate voxels were defined
        """
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asanyarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value

    _volume_shape = None

    @property
    def volume_shape(self):
        """
        Shape of the volumetric image in which the greyordinate voxels were defined
        """
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError("Volume shape should be a tuple of length 3")
            if not all(isinstance(v, integer_types) for v in value):
                raise ValueError("All elements of the volume shape should be integers")
        self._volume_shape = value

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        if (self.__class__ != other.__class__ or len(self) != len(other) or
                not np.array_equal(self.name, other.name) or self.nvertices != other.nvertices or
                any((vox1 != vox2).any() for vox1, vox2 in zip(self.voxels, other.voxels))):
            return False
        if self.affine is not None:
            if (
                    other.affine is None or
                    abs(self.affine - other.affine).max() > 1e-8 or
                    self.volume_shape != other.volume_shape
            ):
                return False
        elif other.affine is not None:
            return False
        for vert1, vert2 in zip(self.vertices, other.vertices):
            if len(vert1) != len(vert2):
                return False
            for name in vert1.keys():
                if name not in vert2 or (vert1[name] != vert2[name]).all():
                    return False
        return True

    def __add__(self, other):
        """
        Concatenates two Parcels

        Parameters
        ----------
        other : Parcels
            parcel to be appended to the current one

        Returns
        -------
        Parcel
        """
        if not isinstance(other, Parcels):
            return NotImplemented
        if self.affine is None:
            affine, shape = other.affine, other.volume_shape
        else:
            affine, shape = self.affine, self.volume_shape
            if other.affine is not None and (not np.allclose(other.affine, affine) or
                                             other.volume_shape != shape):
                raise ValueError("Trying to concatenate two Parcels defined " +
                                 "in a different brain volume")
        nvertices = dict(self.nvertices)
        for name, value in other.nvertices.items():
            if name in nvertices.keys() and nvertices[name] != value:
                raise ValueError("Trying to concatenate two Parcels with inconsistent "
                                 "number of vertices for %s"
                                 % name)
            nvertices[name] = value
        return self.__class__(
                np.append(self.name, other.name),
                np.append(self.voxels, other.voxels),
                np.append(self.vertices, other.vertices),
                affine, shape, nvertices
        )

    def __getitem__(self, item):
        """
        Extracts subset of the axes based on the type of ``item``:

        - `int`: 3-element tuple of (parcel name, parcel voxels, parcel vertices)
        - `string`: 2-element tuple of (parcel voxels, parcel vertices
        - other object that can index 1D arrays: new Parcel axis
        """
        if isinstance(item, string_types):
            idx = np.where(self.name == item)[0]
            if len(idx) == 0:
                raise IndexError("Parcel %s not found" % item)
            if len(idx) > 1:
                raise IndexError("Multiple parcels with name %s found" % item)
            return self.voxels[idx[0]], self.vertices[idx[0]]
        if isinstance(item, integer_types):
            return self.get_element(item)
        return self.__class__(self.name[item], self.voxels[item], self.vertices[item],
                          self.affine, self.volume_shape, self.nvertices)

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 3 elements
        - unicode name of the parcel
        - (M, 3) int array with voxel indices
        - Dict[String -> (K, ) int array] with vertex indices for a specific surface brain structure
        """
        return self.name[index], self.voxels[index], self.vertices[index]


class Scalar(Axis):
    """
    Along this axis of the CIFTI vector/matrix each row/column has been given
    a unique name and optionally metadata
    """

    def __init__(self, name, meta=None):
        """
        Parameters
        ----------
        name : np.ndarray
            (N, ) string array with the parcel names
        meta :  np.ndarray
            (N, ) object array with a dictionary of metadata for each row/column.
            Defaults to empty dictionary
        """
        self.name = np.asanyarray(name, dtype='U')
        if meta is None:
            meta = [{} for _ in range(self.name.size)]
        self.meta = np.asanyarray(meta, dtype='object')

        for check_name in ('name', 'meta'):
            if getattr(self, check_name).shape != (self.size, ):
                raise ValueError("Input {} has incorrect shape ({}) for Scalar axis".format(
                        check_name, getattr(self, check_name).shape))

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new Scalar axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Scalar
        """
        names = [nm.map_name for nm in mim.named_maps]
        meta = [{} if nm.metadata is None else dict(nm.metadata) for nm in mim.named_maps]
        return cls(names, meta)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SCALARS')
        for name, meta in zip(self.name, self.meta):
            meta = None if len(meta) == 0 else meta
            named_map = cifti2.Cifti2NamedMap(name, cifti2.Cifti2MetaData(meta))
            mim.append(named_map)
        return mim

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        """
        Compares two Scalars

        Parameters
        ----------
        other : Scalar
            scalar axis to be compared

        Returns
        -------
        bool : False if type, length or content do not match
        """
        if not isinstance(other, Scalar) or self.size != other.size:
            return False
        return (self.name == other.name).all() and (self.meta == other.meta).all()

    def __add__(self, other):
        """
        Concatenates two Scalars

        Parameters
        ----------
        other : Scalar
            scalar axis to be appended to the current one

        Returns
        -------
        Scalar
        """
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(
                np.append(self.name, other.name),
                np.append(self.meta, other.meta),
        )

    def __getitem__(self, item):
        if isinstance(item, integer_types):
            return self.get_element(item)
        return self.__class__(self.name[item], self.meta[item])

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the row/column
        - dictionary with the element metadata
        """
        return self.name[index], self.meta[index]


class Label(Axis):
    """
    Defines CIFTI axis for label array.

    Along this axis of the CIFTI vector/matrix each row/column has been given a unique name,
    label table, and optionally metadata
    """

    def __init__(self, name, label, meta=None):
        """
        Parameters
        ----------
        name : np.ndarray
            (N, ) string array with the parcel names
        label : np.ndarray
            single dictionary or (N, ) object array with dictionaries mapping
            from integers to (name, (R, G, B, A)), where name is a string and R, G, B, and A are
            floats between 0 and 1 giving the colour and alpha (i.e., transparency)
        meta :  np.ndarray
            (N, ) object array with a dictionary of metadata for each row/column
        """
        self.name = np.asanyarray(name, dtype='U')
        if isinstance(label, dict):
            label = [label] * self.name.size
        self.label = np.asanyarray(label, dtype='object')
        if meta is None:
            meta = [{} for _ in range(self.name.size)]
        self.meta = np.asanyarray(meta, dtype='object')

        for check_name in ('name', 'meta', 'label'):
            if getattr(self, check_name).shape != (self.size, ):
                raise ValueError("Input {} has incorrect shape ({}) for Label axis".format(
                        check_name, getattr(self, check_name).shape))

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new Label axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Label
        """
        tables = [{key: (value.label, value.rgba) for key, value in nm.label_table.items()}
                  for nm in mim.named_maps]
        rest = Scalar.from_mapping(mim)
        return Label(rest.name, tables, rest.meta)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_LABELS')
        for name, label, meta in zip(self.name, self.label, self.meta):
            label_table = cifti2.Cifti2LabelTable()
            for key, value in label.items():
                label_table[key] = (value[0],) + tuple(value[1])
            if len(meta) == 0:
                meta = None
            named_map = cifti2.Cifti2NamedMap(name, cifti2.Cifti2MetaData(meta),
                                              label_table)
            mim.append(named_map)
        return mim

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        """
        Compares two Labels

        Parameters
        ----------
        other : Label
            label axis to be compared

        Returns
        -------
        bool : False if type, length or content do not match
        """
        if not isinstance(other, Label) or self.size != other.size:
            return False
        return (
                (self.name == other.name).all() and
                (self.meta == other.meta).all() and
                (self.label == other.label).all()
        )

    def __add__(self, other):
        """
        Concatenates two Labels

        Parameters
        ----------
        other : Label
            label axis to be appended to the current one

        Returns
        -------
        Label
        """
        if not isinstance(other, Label):
            return NotImplemented
        return Label(
                np.append(self.name, other.name),
                np.append(self.label, other.label),
                np.append(self.meta, other.meta),
        )

    def __getitem__(self, item):
        if isinstance(item, integer_types):
            return self.get_element(item)
        return self.__class__(self.name[item], self.label[item], self.meta[item])

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the row/column
        - dictionary with the label table
        - dictionary with the element metadata
        """
        return self.name[index], self.label[index], self.meta[index]


class Series(Axis):
    """
    Along this axis of the CIFTI vector/matrix the rows/columns increase monotonously in time

    This Axis describes the time point of each row/column.

    Attributes
    ----------
    start : float
        starting time point
    step :  float
        sampling time (TR)
    size : int
        number of time points
    """
    size = None

    def __init__(self, start, step, size, unit="SECOND"):
        """
        Creates a new Series axis

        Parameters
        ----------
        start : float
            Time of the first datapoint
        step : float
            Step size between data points
        size : int
            Number of data points
        unit : str
            Unit of the step size (one of 'second', 'hertz', 'meter', or 'radian')
        """
        self.unit = unit
        self.start = start
        self.step = step
        self.size = size

    @property
    def time(self):
        return np.arange(self.size) * self.step + self.start

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new Series axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Series
        """
        start = mim.series_start * 10 ** mim.series_exponent
        step = mim.series_step * 10 ** mim.series_exponent
        return cls(start, step, mim.number_of_series_points, mim.series_unit)

    def to_mapping(self, dim):
        """
        Converts the Series to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SERIES')
        mim.series_exponent = 0
        mim.series_start = self.start
        mim.series_step = self.step
        mim.number_of_series_points = self.size
        mim.series_unit = self.unit
        return mim

    _unit = None

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value.upper() not in ("SECOND", "HERTZ", "METER", "RADIAN"):
            raise ValueError("Series unit should be one of " +
                             "('second', 'hertz', 'meter', or 'radian'")
        self._unit = value.upper()

    def extend(self, other_axis):
        """
        Concatenates two Series

        Note: this will ignore the start point of the other axis

        Parameters
        ----------
        other_axis : Series
            other axis

        Returns
        -------
        Series
        """
        if other_axis.step != self.step:
            raise ValueError('Can only concatenate Series with the same step size')
        if other_axis.unit != self.unit:
            raise ValueError('Can only concatenate Series with the same unit')
        return Series(self.start, self.step, self.size + other_axis.size, self.unit)

    def __len__(self):
        return self.size

    def __eq__(self, other):
        """
        True if start, step, size, and unit are the same.
        """
        return (
                isinstance(other, Series) and
                self.start == other.start and
                self.step == other.step and
                self.size == other.size and
                self.unit == other.unit
        )

    def __add__(self, other):
        """
        Concatenates two Series

        Parameters
        ----------
        other : Series
            Time Series to append at the end of the current time Series.
            Note that the starting time of the other time Series is ignored.

        Returns
        -------
        Series
            New time Series with the concatenation of the two

        Raises
        ------
        ValueError
            raised if the repetition time of the two time Series is different
        """
        if isinstance(other, Series):
            return self.extend(other)
        return NotImplemented

    def __getitem__(self, item):
        if isinstance(item, slice):
            step = 1 if item.step is None else item.step
            idx_start = ((self.size - 1 if step < 0 else 0)
                         if item.start is None else
                         (item.start if item.start >= 0 else self.size + item.start))
            idx_end = ((-1 if step < 0 else self.size)
                       if item.stop is None else
                       (item.stop if item.stop >= 0 else self.size + item.stop))
            if idx_start > self.size and step < 0:
                idx_start = self.size - 1
            if idx_end > self.size:
                idx_end = self.size
            nelements = (idx_end - idx_start) // step
            if nelements < 0:
                nelements = 0
            return Series(idx_start * self.step + self.start, self.step * step,
                          nelements, self.unit)
        elif isinstance(item, integer_types):
            return self.get_element(item)
        raise IndexError('Series can only be indexed with integers or slices ' +
                         'without breaking the regular structure')

    def get_element(self, index):
        """
        Gives the time point of a specific row/column

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        float
        """
        if index < 0:
            index = self.size + index
        if index >= self.size or index < 0:
            raise IndexError("index %i is out of range for Series with size %i" %
                             (index, self.size))
        return self.start + self.step * index

import numpy as np
from nibabel.cifti2 import cifti2
from six import string_types
from operator import xor


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


class Axis(object):
    """
    Abstract class for any object describing the rows or columns of a CIFTI vector/matrix

    Attributes
    ----------
    arr : np.ndarray
        (N, ) typed array with the actual information on each row/column
    """
    _use_dtype = None
    arr = None

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=self._use_dtype)

    def get_element(self, index):
        """
        Extracts a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        Description of the row/column
        """
        return self.arr[index]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_element(item)
        if isinstance(item, string_types):
            raise IndexError("Can not index an Axis with a string (except for Parcels)")
        return type(self)(self.arr[item])

    @property
    def size(self, ):
        return self.arr.size

    def __len__(self):
        return self.size

    def __eq__(self, other):
        return (type(self) == type(other) and
                len(self) == len(other) and
                (self.arr == other.arr).all())

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
        if type(self) == type(other):
            return type(self)(np.append(self.arr, other.arr))
        return NotImplemented


class BrainModel(Axis):
    """
    Each row/column in the CIFTI vector/matrix represents a single vertex or voxel

    This Axis describes which vertex/voxel is represented by each row/column.

    Attributes
    ----------
    voxel : np.ndarray
        (N, 3) array with the voxel indices
    vertex :  np.ndarray
        (N, ) array with the vertex indices
    name : np.ndarray
        (N, ) array with the brain structure objects
    """
    _use_dtype = np.dtype([('vertex', 'i4'), ('voxel', ('i4', 3)),
                           ('name', 'U%i' % max(len(name) for name in cifti2.CIFTI_BRAIN_STRUCTURES))])
    _affine = None
    _volume_shape = None

    def __init__(self, arr, affine=None, volume_shape=None, nvertices=None):
        """
        Creates a new BrainModel axis defining the vertices and voxels represented by each row/column

        Parameters
        ----------
        arr : np.ndarray
            (N, ) structured array with for every element a tuple with 3 elements:
            - vertex index (-1 for voxels)
            - 3 voxel indices (-1 for vertices)
            - string (name of brain structure)
        affine : np.ndarray
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI files only covering the surface)
        volume_shape : Tuple[int, int, int]
            shape of the volume in which the voxels were defined (not needed for CIFTI files only covering the surface)
        nvertices : dict[String -> int]
            maps names of surface elements to integers
        """
        super(BrainModel, self).__init__(arr)
        self.name = self.name  # correct names to CIFTI brain structures
        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = dict(nvertices)
        for name in list(self.nvertices.keys()):
            if name not in self.name:
                del self.nvertices[name]
        if self.is_surface.all():
            self.affine = None
            self.volume_shape = None
        else:
            self.affine = affine
            self.volume_shape = volume_shape

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
        nbm = np.sum([bm.index_count for bm in mim.brain_models])
        arr = np.zeros(nbm, dtype=cls._use_dtype)
        arr['voxel'] = -1
        arr['vertex'] = -1
        nvertices = {}
        affine, shape = None, None
        for bm in mim.brain_models:
            index_end = bm.index_offset + bm.index_count
            is_surface = bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
            arr['name'][bm.index_offset: index_end] = bm.brain_structure
            if is_surface:
                arr['vertex'][bm.index_offset: index_end] = bm.vertex_indices
                nvertices[bm.brain_structure] = bm.surface_number_of_vertices
            else:
                arr['voxel'][bm.index_offset: index_end, :] = bm.voxel_indices_ijk
                if affine is None:
                    shape = mim.volume.volume_dimensions
                    affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
                else:
                    if shape != mim.volume.volume_dimensions:
                        raise ValueError("All volume masks should be defined in the same volume")
                    if (affine != mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix).any():
                        raise ValueError("All volume masks should have the same affine")
        return cls(arr, affine, shape, nvertices)

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
        if np.asarray(affine).shape != (4, 4):
            raise ValueError("Affine transformation should be a 4x4 array or None, not %r" % affine)
        if mask.ndim == 1:
            return cls.from_surface(np.where(mask != 0)[0], mask.size, name=name)
        elif mask.ndim == 3:
            voxels = np.array(np.where(mask != 0)).T
            arr = np.zeros(len(voxels), dtype=cls._use_dtype)
            arr['vertex'] = -1
            arr['voxel'] = voxels
            arr['name'] = cls.to_cifti_brain_structure_name(name)
            return cls(arr, affine=affine, volume_shape=mask.shape)
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
        arr = np.zeros(len(vertices), dtype=cls._use_dtype)
        arr['voxel'] = -1
        arr['vertex'] = vertices
        arr['name'] = cls.to_cifti_brain_structure_name(name)
        return cls(arr, nvertices={arr['name'][0]: nvertex})

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
        elem = self.arr[index]
        is_surface = elem['name'] in self.nvertices.keys()
        name = 'vertex' if is_surface else 'voxel'
        return is_surface, elem[name], elem['name']

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
                    affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, matrix=self.affine)
                    mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
            cifti_bm = cifti2.Cifti2BrainModel(to_slice.start, len(bm),
                                               'CIFTI_MODEL_TYPE_SURFACE' if is_surface else 'CIFTI_MODEL_TYPE_VOXELS',
                                               name, nvertex, voxels, vertices)
            mim.append(cifti_bm)
        return mim

    def iter_structures(self, ):
        """
        Iterates over all brain structures in the order that they appear along the axis

        Yields
        ------
        tuple with
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

    @property
    def affine(self, ):
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value

    @property
    def volume_shape(self, ):
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError("Volume shape should be a tuple of length 3")
        self._volume_shape = value

    @property
    def is_surface(self, ):
        """True for any element on the surface
        """
        return np.vectorize(lambda name: name in self.nvertices.keys())(self.name)

    @property
    def voxel(self, ):
        """The voxel represented by each row or column
        """
        return self.arr['voxel']

    @voxel.setter
    def voxel(self, values):
        self.arr['voxel'] = values

    @property
    def vertex(self, ):
        """The vertex represented by each row or column
        """
        return self.arr['vertex']

    @vertex.setter
    def vertex(self, values):
        self.arr['vertex'] = values

    @property
    def name(self, ):
        """The brain structure to which the voxel/vertices of belong
        """
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = [self.to_cifti_brain_structure_name(name) for name in values]

    @staticmethod
    def to_cifti_brain_structure_name(name):
        """
        Attempts to convert the name of an anatomical region in a format recognized by CIFTI

        This function returns:
        * the name if it is in the CIFTI format already
        * if the name is a tuple the first element is assumed to be the structure name while
        the second is assumed to be the hemisphere (left, right or both). The latter will default
        to both.
        * names like left_cortex, cortex_left, LeftCortex, or CortexLeft will be converted to
        CIFTI_STRUCTURE_CORTEX_LEFT

        see ``nibabel.cifti2.tests.test_name`` for examples of which conversions are possible

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
            raise ValueError('%s was interpreted as %s, which is not a valid CIFTI brain structure' %
                             (name, proposed_name))
        return proposed_name

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_element(item)
        if isinstance(item, string_types):
            raise IndexError("Can not index an Axis with a string (except for Parcels)")
        return type(self)(self.arr[item], self.affine, self.volume_shape, self.nvertices)

    def __eq__(self, other):
        if type(self) != type(other) or len(self) != len(other):
            return False
        if xor(self.affine is None, other.affine is None):
            return False
        return (((self.affine is None and other.affine is None) or
                 (abs(self.affine - other.affine).max() < 1e-8 and
                  self.volume_shape == other.volume_shape)) and
                (self.nvertices == other.nvertices) and
                (self.arr == other.arr).all())

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
        if type(self) == type(other):
            if self.affine is None:
                affine, shape = other.affine, other.volume_shape
            else:
                affine, shape = self.affine, self.volume_shape
                if other.affine is not None and ((other.affine != affine).all() or
                                                  other.volume_shape != shape):
                    raise ValueError("Trying to concatenate two BrainModels defined in a different brain volume")
            nvertices = dict(self.nvertices)
            for name, value in other.nvertices.items():
                if name in nvertices.keys() and nvertices[name] != value:
                    raise ValueError("Trying to concatenate two BrainModels with inconsistent number of vertices for %s"
                                     % name)
                nvertices[name] = value
            return type(self)(np.append(self.arr, other.arr), affine, shape, nvertices)
        return NotImplemented


class Parcels(Axis):
    """
    Each row/column in the CIFTI vector/matrix represents a parcel of voxels/vertices

    This Axis describes which parcel is represented by each row/column.

    Attributes
    ----------
    name : np.ndarray
        (N, ) string array with the parcel names
    parcel :  np.ndarray
        (N, ) array with the actual get_parcels (each of which is a BrainModel object)

    Individual get_parcels can also be accessed based on their name, using
    ``parcel = parcel_axis[name]``
    """
    _use_dtype = np.dtype([('name', 'U60'), ('voxels', 'object'), ('vertices', 'object')])
    _affine = None
    _volume_shape = None

    def __init__(self, arr, affine=None, volume_shape=None, nvertices=None):
        """
        Creates a new BrainModel axis defining the vertices and voxels represented by each row/column

        Parameters
        ----------
        arr : np.ndarray
            (N, ) structured array with for every element a tuple with 3 elements:
            - string (name of parcel)
            - (M, 3) int array with the M voxel indices in the parcel
            - Dict[String -> (K, ) int array] mapping surface brain structure names to vertex indices
        affine : np.ndarray
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI files only covering the surface)
        volume_shape : Tuple[int, int, int]
            shape of the volume in which the voxels were defined (not needed for CIFTI files only covering the surface)
        nvertices : dict[String -> int]
            maps names of surface elements to integers
        """
        super(Parcels, self).__init__(arr)
        self.affine = affine
        self.volume_shape = volume_shape
        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = dict(nvertices)

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
        affine = None
        volume_shape = None
        arr = np.zeros(len(named_brain_models), dtype=cls._use_dtype)
        nvertices = {}
        for idx_parcel, (parcel_name, bm) in enumerate(named_brain_models):
            voxels = bm.voxel[~bm.is_surface]
            if voxels.shape[0] != 0:
                if affine is None:
                    affine = bm.affine
                    volume_shape = bm.volume_shape
                else:
                    if (affine != bm.affine).any() or (volume_shape != bm.volume_shape):
                        raise ValueError(
                            "Can not combine brain models defined in different volumes into a single Parcel axis")
            vertices = {}
            for name, _, bm_part in bm.iter_structures():
                if name in bm.nvertices.keys():
                    if name in nvertices.keys() and nvertices[name] != bm.nvertices[name]:
                        raise ValueError("Got multiple conflicting number of vertices for surface structure %s" % name)
                    nvertices[name] = bm.nvertices[name]
                    vertices[name] = bm_part.vertex
            arr[idx_parcel] = (parcel_name, voxels, vertices)
        return Parcels(arr, affine, volume_shape, nvertices)

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
        arr = np.zeros(nparcels, dtype=cls._use_dtype)
        volume_shape = None if mim.volume is None else mim.volume.volume_dimensions
        affine = None if mim.volume is None else mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        nvertices = {}
        for surface in mim.surfaces:
            nvertices[surface.brain_structure] = surface.surface_number_of_vertices
        for idx_parcel, parcel in enumerate(mim.parcels):
            nvoxels = 0 if parcel.voxel_indices_ijk is None else len(parcel.voxel_indices_ijk)
            voxels = np.zeros((nvoxels, 3), dtype='i4')
            if nvoxels != 0:
                voxels[()] = parcel.voxel_indices_ijk
            vertices = {}
            for vertex in parcel.vertices:
                name = vertex.brain_structure
                vertices[vertex.brain_structure] = np.array(vertex)
                if name not in nvertices.keys():
                    raise ValueError("Number of vertices for surface structure %s not defined" % name)
            arr[idx_parcel]['voxels'] = voxels
            arr[idx_parcel]['vertices'] = vertices
            arr[idx_parcel]['name'] = parcel.name
        return cls(arr, affine, volume_shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the get_parcels to a MatrixIndicesMap for storage in CIFTI format

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
        for name, voxels, vertices in self.arr:
            cifti_voxels = cifti2.Cifti2VoxelIndicesIJK(voxels)
            element = cifti2.Cifti2Parcel(name, cifti_voxels)
            for name, idx_vertices in vertices.items():
                element.vertices.append(cifti2.Cifti2Vertices(name, idx_vertices))
            mim.append(element)
        return mim

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

    @property
    def affine(self, ):
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value

    @property
    def volume_shape(self, ):
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError("Volume shape should be a tuple of length 3")
        self._volume_shape = value

    @property
    def name(self, ):
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = values

    @property
    def voxels(self, ):
        return self.arr['voxels']

    @voxels.setter
    def voxels(self, values):
        self.arr['voxels'] = values

    @property
    def vertices(self, ):
        return self.arr['vertices']

    @vertices.setter
    def vertices(self, values):
        self.arr['vertices'] = values

    def __getitem__(self, item):
        if isinstance(item, string_types):
            idx = np.where(self.name == item)[0]
            if len(idx) == 0:
                raise IndexError("Parcel %s not found" % item)
            if len(idx) > 1:
                raise IndexError("Multiple get_parcels with name %s found" % item)
            return self.voxels[idx[0]], self.vertices[idx[0]]
        if isinstance(item, int):
            return self.get_element(item)
        if isinstance(item, string_types):
            raise IndexError("Can not index an Axis with a string (except for Parcels)")
        return type(self)(self.arr[item], self.affine, self.volume_shape, self.nvertices)

    def __eq__(self, other):
        if (type(self) != type(other) or len(self) != len(other) or
                (self.name != other.name).all() or self.nvertices != other.nvertices or
                any((vox1 != vox2).any() for vox1, vox2 in zip(self.voxels, other.voxels))):
            return False
        if self.affine is not None:
            if (  other.affine is None or
                  abs(self.affine - other.affine).max() > 1e-8 or
                  self.volume_shape != other.volume_shape):
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
        other : Parcel
            parcel to be appended to the current one

        Returns
        -------
        Parcel
        """
        if type(self) == type(other):
            if self.affine is None:
                affine, shape = other.affine, other.volume_shape
            else:
                affine, shape = self.affine, self.volume_shape
                if other.affine is not None and ((other.affine != affine).all() or
                                                  other.volume_shape != shape):
                    raise ValueError("Trying to concatenate two Parcels defined in a different brain volume")
            nvertices = dict(self.nvertices)
            for name, value in other.nvertices.items():
                if name in nvertices.keys() and nvertices[name] != value:
                    raise ValueError("Trying to concatenate two Parcels with inconsistent number of vertices for %s"
                                     % name)
                nvertices[name] = value
            return type(self)(np.append(self.arr, other.arr), affine, shape, nvertices)
        return NotImplemented


class Scalar(Axis):
    """
    Along this axis of the CIFTI vector/matrix each row/column has been given a unique name and optionally metadata

    Attributes
    ----------
    name : np.ndarray
        (N, ) string array with the parcel names
    meta :  np.ndarray
        (N, ) array with a dictionary of metadata for each row/column
    """
    _use_dtype = np.dtype([('name', 'U60'), ('meta', 'object')])

    def __init__(self, arr):
        """
        Creates a new Scalar axis from (name, meta-data) pairs

        Parameters
        ----------
        arr : Iterable[Tuple[str, dict[str -> str]]
            iterates over all rows/columns assigning a name and a dictionary of metadata to each
        """
        super(Scalar, self).__init__(arr)

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new get_scalar axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Scalar
        """
        res = np.zeros(len(list(mim.named_maps)), dtype=cls._use_dtype)
        res['name'] = [nm.map_name for nm in mim.named_maps]
        res['meta'] = [{} if nm.metadata is None else dict(nm.metadata) for nm in mim.named_maps]
        return cls(res)

    @classmethod
    def from_names(cls, names):
        """
        Creates a new get_scalar axis with the given row/column names

        Parameters
        ----------
        names : List[str]
            gives a unique name to every row/column in the matrix

        Returns
        -------
        Scalar
        """
        res = np.zeros(len(names), dtype=cls._use_dtype)
        res['name'] = names
        res['meta'] = [{} for _ in names]
        return cls(res)

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
        for elem in self.arr:
            meta = None if len(elem['meta']) == 0 else elem['meta']
            named_map = cifti2.Cifti2NamedMap(elem['name'], cifti2.Cifti2MetaData(meta))
            mim.append(named_map)
        return mim

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
        - unicode name of the get_scalar
        - dictionary with the element metadata
        """
        return self.arr['name'][index], self.arr['meta'][index]

    def to_label(self, labels):
        """
        Creates a new Label axis based on the Scalar axis

        Parameters
        ----------
        labels : list[dict]
            mapping from integers to (name, (R, G, B, A)), where `name` is a string and R, G, B, and A are floats
            between 0 and 1 giving the colour and alpha (transparency)

        Returns
        -------
        Label
        """
        res = np.zeros(self.size, dtype=Label._use_dtype)
        res['name'] = self.arr['name']
        res['meta'] = self.arr['meta']
        res['get_label'] = labels
        return Label(res)

    @property
    def name(self, ):
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = values

    @property
    def meta(self, ):
        return self.arr['meta']

    @meta.setter
    def meta(self, values):
        self.arr['meta'] = values


class Label(Axis):
    """
    Along this axis of the CIFTI vector/matrix each row/column has been given a unique name,
    get_label table, and optionally metadata

    Attributes
    ----------
    name : np.ndarray
        (N, ) string array with the parcel names
    meta :  np.ndarray
        (N, ) array with a dictionary of metadata for each row/column
    get_label : sp.ndarray
        (N, ) array with dictionaries mapping integer values to get_label names and RGBA colors
    """
    _use_dtype = np.dtype([('name', 'U60'), ('get_label', 'object'), ('meta', 'object')])

    def __init__(self, arr):
        """
        Creates a new Scalar axis from (name, meta-data) pairs

        Parameters
        ----------
        arr : Iterable[Tuple[str, dict[int -> (str, (float, float, float, float)), dict(str->str)]]
            iterates over all rows/columns assigning a name, dictionary mapping integers to get_label names and rgba colours
            and a dictionary of metadata to each
        """
        super(Label, self).__init__(arr)

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new get_scalar axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Scalar
        """
        tables = [{key: (value.label, value.rgba) for key, value in nm.label_table.items()}
                  for nm in mim.named_maps]
        return Scalar.from_mapping(mim).to_label(tables)

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
        for elem in self.arr:
            label_table = cifti2.Cifti2LabelTable()
            for key, value in elem['get_label'].items():
                label_table[key] = (value[0],) + tuple(value[1])
            meta = None if len(elem['meta']) == 0 else elem['meta']
            named_map = cifti2.Cifti2NamedMap(elem['name'], cifti2.Cifti2MetaData(meta),
                                              label_table)
            mim.append(named_map)
        return mim

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
        - unicode name of the get_scalar
        - dictionary with the get_label table
        - dictionary with the element metadata
        """
        return self.arr['name'][index], self.arr['get_label'][index], self.arr['meta'][index]

    @property
    def name(self, ):
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = values

    @property
    def meta(self, ):
        return self.arr['meta']

    @meta.setter
    def meta(self, values):
        self.arr['meta'] = values

    @property
    def label(self, ):
        return self.arr['get_label']

    @label.setter
    def label(self, values):
        self.arr['get_label'] = values


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
    _unit = None

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
    def unit(self, ):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value.upper() not in ("SECOND", "HERTZ", "METER", "RADIAN"):
            raise ValueError("Series unit should be one of ('second', 'hertz', 'meter', or 'radian'")
        self._unit = value.upper()

    @property
    def arr(self, ):
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
        Converts the get_series to a MatrixIndicesMap for storage in CIFTI format

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

    def extend(self, other_axis):
        """
        Concatenates two get_series

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
            raise ValueError('Can only concatenate get_series with the same step size')
        if other_axis.unit != self.unit:
            raise ValueError('Can only concatenate get_series with the same unit')
        return Series(self.start, self.step, self.size + other_axis.size, self.unit)

    def __getitem__(self, item):
        if isinstance(item, slice):
            step = 1 if item.step is None else item.step
            idx_start = ((self.size - 1 if step < 0 else 0)
                         if item.start is None else
                         (item.start if item.start >= 0 else self.size + item.start))
            idx_end = ((-1 if step < 0 else self.size)
                       if item.stop is None else
                       (item.stop if item.stop >= 0 else self.size + item.stop))
            if idx_start > self.size:
                idx_start = self.size - 1
            if idx_end > self.size:
                idx_end = self.size
            nelements = (idx_end - idx_start) // step
            if nelements < 0:
                nelements = 0
            return Series(idx_start * self.step + self.start, self.step * step, nelements)
        elif isinstance(item, int):
            return self.get_element(item)
        raise IndexError('Series can only be indexed with integers or slices without breaking the regular structure')

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
        if index >= self.size:
            raise IndexError("index %i is out of range for get_series with size %i" % (index, self.size))
        return self.start + self.step * index

    def __add__(self, other):
        """
        Concatenates two Series

        Parameters
        ----------
        other : Series
            Time get_series to append at the end of the current time get_series.
            Note that the starting time of the other time get_series is ignored.

        Returns
        -------
        Series
            New time get_series with the concatenation of the two

        Raises
        ------
        ValueError
            raised if the repetition time of the two time get_series is different
        """
        if isinstance(other, Series):
            return self.extend(other)
        return NotImplemented

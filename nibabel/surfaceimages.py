from nibabel.affines import apply_affine
from nibabel.filebasedimages import FileBasedHeader
from nibabel.dataobj_images import DataobjImage


class Pointset:
    """ A collection of related sets of coordinates in 3D space.

    Coordinates are in RAS+ orientation, i.e., $(x, y, z)$ refers to
    a point $x\mathrm{mm}$ right, $y\mathrm{mm}$ anterior and $z\mathrm{mm}$
    superior of the origin.

    The order of coordinates should be assumed to be significant.
    """

    def get_coords(self, name=None):
        """ Get coordinate data in RAS+ space

        Parameters
        ----------
        name : str
            Name of one of a family of related coordinates.
            For example, ``"white"`` and ``"pial"`` surfaces

        Returns
        -------
        coords : (N, 3) array-like
            Coordinates in RAS+ space
        """
        raise NotImplementedError

    @property
    def n_coords(self):
        """ Number of coordinates

        The default implementation loads coordinates. Subclasses may
        override with more efficient implementations.
        """
        return self.get_coords().shape[0]


class TriangularMesh(Pointset):
    """ A triangular mesh is a description of a surface tesselated into triangles """

    def __init__(self, meshes=None):
        """ Triangular mesh objects have access to an internal
        ``_meshes`` dictionary that has keys that are mesh names.
        The values may be any structure that permits the class to
        provide coordinates and triangles on request.
        """
        if meshes is None:
            meshes = {}
        self._meshes = meshes
        super().__init__()

    @property
    def n_triangles(self):
        """ Number of triangles (faces)

        The default implementation loads triangles. Subclasses may
        override with more efficient implementations.
        """
        return self.get_triangles().shape[0]

    def get_triangles(self, name=None):
        """ Mx3 array of indices into coordinate table """
        raise NotImplementedError

    def get_mesh(self, name=None):
        return self.get_coords(name=name), self.get_triangles(name=name)

    def get_names(self):
        """ List of surface names that can be passed to
        ``get_{coords,triangles,mesh}``
        """
        return list(self._meshes.keys())

    def decimate(self, *, ncoords=None, ratio=None):
        """ Return a TriangularMesh with a smaller number of vertices that
        preserves the geometry of the original.

        Please contribute a generic decimation algorithm at
        https://github.com/nipy/nibabel
        """
        raise NotImplementedError

    def load_vertex_data(self, pathlike):
        """ Return a SurfaceImage with data corresponding to each vertex """
        raise NotImplementedError

    def load_face_data(self, pathlike):
        """ Return a SurfaceImage with data corresponding to each face """
        raise NotImplementedError


class SurfaceHeader(FileBasedHeader):
    """ Template class to implement SurfaceHeader protocol """
    def get_geometry(self):
        """ Generate ``TriangularMesh`` object from header object

        If no default geometry can be provided, returns ``None``.
        """
        return None


class SurfaceImage(DataobjImage):
    header_class = SurfaceHeader

    def __init__(self, dataobj, header=None, geometry=None, extra=None, file_map=None):
        """ Initialize image

        The image is a combination of 
        """
        super().__init__(dataobj, header=header, extra=extra, file_map=file_map)
        if geometry is None:
            geometry = self.header.get_geometry()
        self._geometry = geometry

    def load_geometry(self, pathlike):
        """ Specify a header to a data-only image """


class VolumeGeometry(Pointset):
    def __init__(self, affines, *, indices=None):
        try:
            self._affines = dict(affines)
        except TypeError:
            self._affines = {"affine": np.array(affines)}
        self._default = next(iter(self._affines))

        self._indices = indices

    def get_coords(self, name=None):
        if name is None:
            name = self._default
        return apply_affine(self._affines[name], self._indices)

    @property
    def n_coords(self):
        return self._indices.shape[0]

    def get_indices(self):
        return self._indices


class GeometryCollection:
    def __init__(self, structures=()):
        self._structures = dict(structures)

    def get_structure(self, name):
        return self._structures[name]

    @property
    def names(self):
        return list(self._structures)

    @classmethod
    def from_spec(klass, pathlike):
        """ Load a collection of geometries from a specification, broadly construed. """
        raise NotImplementedError


class PointsetSequence(GeometryCollection, Pointset):
    def __init__(self, structures=()):
        super().__init__(structures)
        self._indices = {}
        next_index = 0
        for name, struct in self._structures.items():
            end = next_index + struct.n_coords
            self._indices[name] = slice(next_index, end)
            next_index = end + 1

    def get_indices(self, *names):
        if len(names) == 1:
            return self._indices[name]
        return [self._indices[name] for name in names]

    # def get_structures(self, *, names=None, indices=None):
    #     """ We probably want some way to get a subset of structures """

    def get_coords(self, name=None):
        return np.vstack([struct.get_coords(name=name)
                          for struct in self._structures.values()])

    @property
    def n_coords(self):
        return sum(struct.n_coords for struct in self._structures.values())

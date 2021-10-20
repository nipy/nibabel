from nibabel.filebasedimages import FileBasedHeader
from nibabel.dataobj_images import DataobjImage


class SurfaceGeometry:
    def __init__(self, meshes=None):
        """ Surface header objects have access to an internal
        ``_meshes`` dictionary that has keys that are mesh names.
        The values may be any structure that permits the class to
        provide coordinates and triangles on request.
        """
        if meshes is None:
            meshes = {}
        self._meshes = meshes
        super().__init__(*args, **kwargs)

    @property
    def n_coords(self):
        """ Number of coordinates (vertices)

        The default implementation loads coordinates. Subclasses may
        override with more efficient implementations.
        """
        return self.get_coords().shape[0]

    @property
    def n_triangles(self):
        """ Number of triangles (faces)

        The default implementation loads triangles. Subclasses may
        override with more efficient implementations.
        """
        return self.get_triangles().shape[0]

    def get_coords(self, name=None):
        """ Nx3 array of coordinates in RAS+ space """
        raise NotImplementedError

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
        """ Return a SurfaceHeader with a smaller number of vertices that
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
        """ Generate ``SurfaceGeometry`` object from header object

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

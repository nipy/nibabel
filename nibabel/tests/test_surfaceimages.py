from nibabel.surfaceimages import SurfaceGeometry, SurfaceHeader, SurfaceImage
from nibabel.optpkg import optional_package

from nibabel.tests.test_filebasedimages import FBNumpyImage

import numpy as np
from pathlib import Path

h5, has_h5py, _ = optional_package('h5py')


class H5ArrayProxy:
    def __init__(self, file_like, dataset_name):
        self.file_like = file_like
        self.dataset_name = dataset_name
        with h5.File(file_like, "r") as h5f:
            arr = h5f[dataset_name]
            self._shape = arr.shape
            self._dtype = arr.dtype

    @property
    def is_proxy(self):
        return True

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype

    def __array__(self, dtype=None):
        with h5.File(self.file_like, "r") as h5f:
            return np.asanyarray(h5f[self.dataset_name], dtype)

    def __slicer__(self, slicer):
        with h5.File(self.file_like, "r") as h5f:
            return h5f[self.dataset_name][slicer]


class H5Geometry(SurfaceGeometry):
    """Simple Geometry file structure that combines a single topology
    with one or more coordinate sets
    """
    @classmethod
    def from_filename(klass, pathlike):
        meshes = {}
        with h5.File(pathlike, "r") as h5f:
            triangles = h5f['topology']
            for name, coords in h5f['coordinates'].items():
                meshes[name] = (coords, triangles)
        return klass(meshes)

    
    def to_filename(self, pathlike):
        topology = None
        coordinates = {}
        for name, mesh in self.meshes.items():
            coords, faces = mesh
            if topology is None:
                topology = faces
            elif not np.array_equal(faces, topology):
                raise ValueError("Inconsistent topology")
            coordinates[name] = coords

        with h5.File(pathlike, "w") as h5f:
            h5f.create_dataset("/topology", topology)
            for name, coord in coordinates.items():
                h5f.create_dataset(f"/coordinates/{name}", coord)


    def get_coords(self, name=None):
        if name is None:
            name = next(iter(self._meshes))
        coords, _ = self._meshes[name]
        return coords


    def get_triangles(self, name=None):
        if name is None:
            name = next(iter(self._meshes))
        _, triangles = self._meshes[name]
        return triangles


class NPSurfaceImage(SurfaceImage):
    valid_exts = ('.npy',)
    files_types = (('image', '.npy'),)

    @classmethod
    def from_file_map(klass, file_map):
        with file_map['image'].get_prepare_fileobj('rb') as fobj:
            arr = np.load(fobj)
        return klass(arr)

    def to_file_map(self, file_map=None):
        file_map = self.file_map if file_map is None else file_map
        with file_map['image'].get_prepare_fileobj('wb') as fobj:
            np.save(fobj, self.arr)

    def get_data_dtype(self):
        return self.dataobj.dtype

    def set_data_dtype(self, dtype):
        self.dataobj = self.dataobj.astype(dtype)

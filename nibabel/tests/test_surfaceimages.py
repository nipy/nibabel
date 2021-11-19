import os
from nibabel.surfaceimages import *
from nibabel.arrayproxy import ArrayProxy
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


class H5Geometry(TriangularMesh):
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


class FSGeometryProxy:
    def __init__(self, pathlike):
        self._file_like = str(Path(pathlike))
        self._offset = None
        self._vnum = None
        self._fnum = None

    def _peek(self):
        from nibabel.freesurfer.io import _fread3
        with open(self._file_like, "rb") as fobj:
            magic = _fread3(fobj)
            if magic != 16777214:
                raise NotImplementedError("Triangle files only!")
            fobj.readline()
            fobj.readline()
            self._vnum = np.fromfile(fobj, ">i4", 1)[0]
            self._fnum = np.fromfile(fobj, ">i4", 1)[0]
            self._offset = fobj.tell()

    @property
    def vnum(self):
        if self._vnum is None:
            self._peek()
        return self._vnum

    @property
    def fnum(self):
        if self._fnum is None:
            self._peek()
        return self._fnum

    @property
    def offset(self):
        if self._offset is None:
            self._peek()
        return self._offset

    @property
    def coords(self):
        ap = ArrayProxy(self._file_like, ((self.vnum, 3), ">f4", self.offset))
        ap.order = 'C'
        return ap

    @property
    def triangles(self):
        offset = self.offset + 12 * self.vnum
        ap = ArrayProxy(self._file_like, ((self.fnum, 3), ">i4", offset))
        ap.order = 'C'
        return ap


class FreeSurferHemisphere(TriangularMesh):
    @classmethod
    def from_filename(klass, pathlike):
        path = Path(pathlike)
        hemi, default = path.name.split(".")
        mesh_names = ('orig', 'white', 'smoothwm',
                      'pial', 'inflated', 'sphere',
                      'midthickness', 'graymid')  # Often created
        if default not in mesh_names:
            mesh_names.append(default)
        meshes = {}
        for mesh in mesh_names:
            fpath = path.parent / f"{hemi}.{mesh}"
            if fpath.exists():
                meshes[mesh] = FSGeometryProxy(fpath)
        hemi = klass(meshes)
        hemi._default = default
        return hemi

    def get_coords(self, name=None):
        if name is None:
            name = self._default
        return self._meshes[name].coords

    def get_triangles(self, name=None):
        if name is None:
            name = self._default
        return self._meshes[name].triangles

    @property
    def n_coords(self):
        return self.meshes[self._default].vnum

    @property
    def n_triangles(self):
        return self.meshes[self._default].fnum


class FreeSurferSubject(GeometryCollection):
    @classmethod
    def from_subject(klass, subject_id, subjects_dir=None):
        """ Load a FreeSurfer subject by ID """
        if subjects_dir is None:
            subjects_dir = os.environ["SUBJECTS_DIR"]
        return klass.from_directory(Path(subjects_dir) / subject_id)

    @classmethod
    def from_spec(klass, pathlike):
        """ Load a FreeSurfer subject from its directory structure """
        self._subject_dir = Path(pathlike)
        surfs = self._subject_dir / "surf"
        self._structures = {
            "lh": FreeSurferHemisphere.from_filename(surfs / "lh.white"),
            "rh": FreeSurferHemisphere.from_filename(surfs / "rh.white"),
            }
        super().__init__()

from collections import namedtuple
from math import prod
from pathlib import Path

import numpy as np
import pytest

from nibabel import pointset as ps
from nibabel.affines import apply_affine
from nibabel.fileslice import strided_scalar
from nibabel.optpkg import optional_package
from nibabel.spatialimages import SpatialImage
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data

h5, has_h5py, _ = optional_package('h5py')

FS_DATA = Path(get_nibabel_data()) / 'nitest-freesurfer'


class TestPointsets:
    rng = np.random.default_rng()

    @pytest.mark.parametrize('shape', [(5, 2), (5, 3), (5, 4)])
    @pytest.mark.parametrize('homogeneous', [True, False])
    def test_init(self, shape, homogeneous):
        coords = self.rng.random(shape)

        if homogeneous:
            coords = np.column_stack([coords, np.ones(shape[0])])

        points = ps.Pointset(coords, homogeneous=homogeneous)
        assert np.allclose(points.affine, np.eye(shape[1] + 1))
        assert points.homogeneous is homogeneous
        assert (points.n_coords, points.dim) == shape

        points = ps.Pointset(coords, affine=np.diag([2] * shape[1] + [1]), homogeneous=homogeneous)
        assert np.allclose(points.affine, np.diag([2] * shape[1] + [1]))
        assert points.homogeneous is homogeneous
        assert (points.n_coords, points.dim) == shape

        # Badly shaped affine
        with pytest.raises(ValueError):
            ps.Pointset(coords, affine=[0, 1])

        # Badly valued affine
        with pytest.raises(ValueError):
            ps.Pointset(coords, affine=np.ones((shape[1] + 1, shape[1] + 1)))

    @pytest.mark.parametrize('shape', [(5, 2), (5, 3), (5, 4)])
    @pytest.mark.parametrize('homogeneous', [True, False])
    def test_affines(self, shape, homogeneous):
        orig_coords = coords = self.rng.random(shape)

        if homogeneous:
            coords = np.column_stack([coords, np.ones(shape[0])])

        points = ps.Pointset(coords, homogeneous=homogeneous)
        assert np.allclose(points.get_coords(), orig_coords)

        # Apply affines
        scaler = np.diag([2] * shape[1] + [1])
        scaled = scaler @ points
        assert np.array_equal(scaled.coordinates, points.coordinates)
        assert np.array_equal(scaled.affine, scaler)
        assert np.allclose(scaled.get_coords(), 2 * orig_coords)

        flipper = np.eye(shape[1] + 1)
        # [[1, 0, 0], [0, 1, 0], [0, 0, 1]] becomes [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        flipper[:-1] = flipper[-2::-1]
        flipped = flipper @ points
        assert np.array_equal(flipped.coordinates, points.coordinates)
        assert np.array_equal(flipped.affine, flipper)
        assert np.allclose(flipped.get_coords(), orig_coords[:, ::-1])

        # Concatenate affines, with any associativity
        for doubledup in [(scaler @ flipper) @ points, scaler @ (flipper @ points)]:
            assert np.array_equal(doubledup.coordinates, points.coordinates)
            assert np.allclose(doubledup.affine, scaler @ flipper)
            assert np.allclose(doubledup.get_coords(), 2 * orig_coords[:, ::-1])

    def test_homogeneous_coordinates(self):
        ccoords = self.rng.random((5, 3))
        hcoords = np.column_stack([ccoords, np.ones(5)])

        cartesian = ps.Pointset(ccoords)
        homogeneous = ps.Pointset(hcoords, homogeneous=True)

        for points in (cartesian, homogeneous):
            assert np.array_equal(points.get_coords(), ccoords)
            assert np.array_equal(points.get_coords(as_homogeneous=True), hcoords)

        affine = np.diag([2, 3, 4, 1])
        cart2 = affine @ cartesian
        homo2 = affine @ homogeneous

        exp_c = apply_affine(affine, ccoords)
        exp_h = (affine @ hcoords.T).T
        for points in (cart2, homo2):
            assert np.array_equal(points.get_coords(), exp_c)
            assert np.array_equal(points.get_coords(as_homogeneous=True), exp_h)


def test_GridIndices():
    # 2D case
    shape = (2, 3)
    gi = ps.GridIndices(shape)

    assert gi.dtype == np.dtype('u1')
    assert gi.shape == (6, 2)
    assert repr(gi) == '<GridIndices(2, 3)>'

    gi_arr = np.asanyarray(gi)
    assert gi_arr.dtype == np.dtype('u1')
    assert gi_arr.shape == (6, 2)
    # Tractable to write out
    assert np.array_equal(gi_arr, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

    shape = (2, 3, 4)
    gi = ps.GridIndices(shape)

    assert gi.dtype == np.dtype('u1')
    assert gi.shape == (24, 3)
    assert repr(gi) == '<GridIndices(2, 3, 4)>'

    gi_arr = np.asanyarray(gi)
    assert gi_arr.dtype == np.dtype('u1')
    assert gi_arr.shape == (24, 3)
    # Separate implementation
    assert np.array_equal(gi_arr, np.mgrid[:2, :3, :4].reshape(3, -1).T)


class TestGrids(TestPointsets):
    @pytest.mark.parametrize('shape', [(5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5, 5)])
    def test_from_image(self, shape):
        # Check image is generates voxel coordinates
        affine = np.diag([2, 3, 4, 1])
        img = SpatialImage(strided_scalar(shape), affine)
        grid = ps.Grid.from_image(img)
        grid_coords = grid.get_coords()

        assert grid.n_coords == prod(shape[:3])
        assert grid.dim == 3
        assert np.allclose(grid.affine, affine)

        assert np.allclose(grid_coords[0], [0, 0, 0])
        # Final index is [4, 4, 4], scaled by affine
        assert np.allclose(grid_coords[-1], [8, 12, 16])

    def test_from_mask(self):
        affine = np.diag([2, 3, 4, 1])
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        img = SpatialImage(mask, affine)

        grid = ps.Grid.from_mask(img)
        grid_coords = grid.get_coords()

        assert grid.n_coords == 1
        assert grid.dim == 3
        assert np.array_equal(grid_coords, [[2, 3, 4]])

    def test_to_mask(self):
        coords = np.array([[1, 1, 1]])

        grid = ps.Grid(coords)

        mask_img = grid.to_mask()
        assert mask_img.shape == (2, 2, 2)
        assert np.array_equal(mask_img.get_fdata(), [[[0, 0], [0, 0]], [[0, 0], [0, 1]]])
        assert np.array_equal(mask_img.affine, np.eye(4))

        mask_img = grid.to_mask(shape=(3, 3, 3))
        assert mask_img.shape == (3, 3, 3)
        assert np.array_equal(
            mask_img.get_fdata(),
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        )
        assert np.array_equal(mask_img.affine, np.eye(4))


class TestTriangularMeshes:
    def test_api(self):
        # Tetrahedron
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        triangles = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [0, 1, 3],
                [1, 2, 3],
            ]
        )

        mesh = namedtuple('mesh', ('coordinates', 'triangles'))(coords, triangles)

        tm1 = ps.TriangularMesh(coords, triangles)
        tm2 = ps.TriangularMesh.from_tuple(mesh)
        tm3 = ps.TriangularMesh.from_object(mesh)

        assert np.allclose(tm1.affine, np.eye(4))
        assert np.allclose(tm2.affine, np.eye(4))
        assert np.allclose(tm3.affine, np.eye(4))

        assert tm1.homogeneous is False
        assert tm2.homogeneous is False
        assert tm3.homogeneous is False

        assert (tm1.n_coords, tm1.dim) == (4, 3)
        assert (tm2.n_coords, tm2.dim) == (4, 3)
        assert (tm3.n_coords, tm3.dim) == (4, 3)

        assert tm1.n_triangles == 4
        assert tm2.n_triangles == 4
        assert tm3.n_triangles == 4

        out_coords, out_tris = tm1.get_mesh()
        # Currently these are the exact arrays, but I don't think we should
        # bake that assumption into the tests
        assert np.allclose(out_coords, coords)
        assert np.allclose(out_tris, triangles)


class TestCoordinateFamilyMixin(TestPointsets):
    def test_names(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        cfm = ps.CoordinateFamilyMixin(coords)

        assert cfm.get_names() == ['original']
        assert np.allclose(cfm.with_name('original').coordinates, coords)

        cfm.add_coordinates('shifted', coords + 1)
        assert set(cfm.get_names()) == {'original', 'shifted'}
        shifted = cfm.with_name('shifted')
        assert np.allclose(shifted.coordinates, coords + 1)
        assert set(shifted.get_names()) == {'original', 'shifted'}
        original = shifted.with_name('original')
        assert np.allclose(original.coordinates, coords)

        # Avoid duplicating objects
        assert original.with_name('original') is original
        # But don't try too hard
        assert original.with_name('original') is not cfm

        # with_name() preserves the exact coordinate mapping of the source object.
        # Modifications of one are immediately available to all others.
        # This is currently an implementation detail, and the expectation is that
        # a family will be created once and then queried, but this behavior could
        # potentially become confusing or relied upon.
        # Change with care.
        shifted.add_coordinates('shifted-again', coords + 2)
        shift2 = shifted.with_name('shifted-again')
        shift3 = cfm.with_name('shifted-again')


class H5ArrayProxy:
    def __init__(self, file_like, dataset_name):
        self.file_like = file_like
        self.dataset_name = dataset_name
        with h5.File(file_like, 'r') as h5f:
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
        with h5.File(self.file_like, 'r') as h5f:
            return np.asanyarray(h5f[self.dataset_name], dtype)

    def __getitem__(self, slicer):
        with h5.File(self.file_like, 'r') as h5f:
            return h5f[self.dataset_name][slicer]


class H5Geometry(ps.CoordinateFamilyMixin, ps.TriangularMesh):
    """Simple Geometry file structure that combines a single topology
    with one or more coordinate sets
    """

    @classmethod
    def from_filename(klass, pathlike):
        coords = {}
        with h5.File(pathlike, 'r') as h5f:
            triangles = H5ArrayProxy(pathlike, '/topology')
            for name in h5f['coordinates']:
                coords[name] = H5ArrayProxy(pathlike, f'/coordinates/{name}')
        self = klass(next(iter(coords.values())), triangles, mapping=coords)
        return self

    def to_filename(self, pathlike):
        with h5.File(pathlike, 'w') as h5f:
            h5f.create_dataset('/topology', data=self.get_triangles())
            for name, coord in self._coords.items():
                h5f.create_dataset(f'/coordinates/{name}', data=coord)


class FSGeometryProxy:
    def __init__(self, pathlike):
        self._file_like = str(Path(pathlike))
        self._offset = None
        self._vnum = None
        self._fnum = None

    def _peek(self):
        from nibabel.freesurfer.io import _fread3

        with open(self._file_like, 'rb') as fobj:
            magic = _fread3(fobj)
            if magic != 16777214:
                raise NotImplementedError('Triangle files only!')
            fobj.readline()
            fobj.readline()
            self._vnum = np.fromfile(fobj, '>i4', 1)[0]
            self._fnum = np.fromfile(fobj, '>i4', 1)[0]
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

    @auto_attr
    def coordinates(self):
        return ArrayProxy(self._file_like, ((self.vnum, 3), '>f4', self.offset), order='C')

    @auto_attr
    def triangles(self):
        return ArrayProxy(
            self._file_like,
            ((self.fnum, 3), '>i4', self.offset + 12 * self.vnum),
            order='C',
        )


class FreeSurferHemisphere(ps.CoordinateFamilyMixin, ps.TriangularMesh):
    @classmethod
    def from_filename(klass, pathlike):
        path = Path(pathlike)
        hemi, default = path.name.split('.')
        self = klass.from_object(FSGeometryProxy(path), name=default)
        mesh_names = (
            'orig',
            'white',
            'smoothwm',
            'pial',
            'inflated',
            'sphere',
            'midthickness',
            'graymid',
        )  # Often created

        for mesh in mesh_names:
            if mesh != default:
                fpath = path.parent / f'{hemi}.{mesh}'
                if fpath.exists():
                    self.add_coordinates(mesh, FSGeometryProxy(fpath).coordinates)
        return self


@needs_nibabel_data('nitest-freesurfer')
def test_FreeSurferHemisphere():
    lh = FreeSurferHemisphere.from_filename(FS_DATA / 'fsaverage/surf/lh.white')
    assert lh.n_coords == 163842
    assert lh.n_triangles == 327680


@skipUnless(has_h5py, reason='Test requires h5py')
@needs_nibabel_data('nitest-freesurfer')
def test_make_H5Geometry(tmp_path):
    lh = FreeSurferHemisphere.from_filename(FS_DATA / 'fsaverage/surf/lh.white')
    h5geo = H5Geometry.from_object(lh)
    for name in ('white', 'pial'):
        h5geo.add_coordinates(name, lh.with_name(name).coordinates)
    h5geo.to_filename(tmp_path / 'geometry.h5')

    rt_h5geo = H5Geometry.from_filename(tmp_path / 'geometry.h5')
    assert set(h5geo._coords) == set(rt_h5geo._coords)
    assert np.array_equal(
        lh.with_name('white').get_coords(), rt_h5geo.with_name('white').get_coords()
    )
    assert np.array_equal(lh.get_triangles(), rt_h5geo.get_triangles())

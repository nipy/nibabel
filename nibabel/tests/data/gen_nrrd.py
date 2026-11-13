"""Generate the small NRRD test files used by ``nibabel/tests/test_nrrd.py``

Run from this directory with ``python gen_nrrd.py``. The script only needs numpy
and the standard library, so the files do not depend on the reader under test.

Files written:

* ``nrrd_raw_lps.nrrd``: 3 x 4 x 5 int16, little endian, raw, oblique, LPS
* ``nrrd_gzip_ras.nrrd``: same voxels and geometry as ``nrrd_raw_lps.nrrd``, in RAS,
  gzip encoded
* ``nrrd_detached.nhdr`` and ``nrrd_detached.raw``: same voxels and geometry, detached
  raw data with ``line skip`` and ``byte skip``
* ``nrrd_detached_be.nhdr`` and ``nrrd_detached_be.raw.gz``: same voxels and geometry,
  big endian, detached gzip data
* ``nrrd_list_vector.nrrd``: float32 with a vector axis first and a list axis last
  (on disk sizes 3 x 3 x 4 x 5 x 2), gzip encoded, with a measurement frame,
  spacings and key/value pairs
* ``nrrd_2d.nrrd``: 4 x 5 uint8 with ``space dimension: 2``
"""

import gzip

import numpy as np

# Voxel values shared by the 3D files, stored in Fortran order on disk
DATA_3D = np.arange(60, dtype=np.int16).reshape((3, 4, 5)) - 30

LPS_GEOMETRY = (
    'space: left-posterior-superior\n'
    'space directions: (1.2,0.9,0) (-1.2,1.6,0) (0,0,2.5)\n'
    'space origin: (10,-20,30)\n'
)
RAS_GEOMETRY = (
    'space: RAS\n'
    'space directions: (-1.2,-0.9,0) (1.2,-1.6,0) (0,0,2.5)\n'
    'space origin: (-10,20,30)\n'
)


def header(version, fields):
    return f'NRRD000{version}\n# Written by gen_nrrd.py\n{fields}'.encode('ascii')


def write(fname, contents):
    with open(fname, 'wb') as fobj:
        fobj.write(contents)


def main():
    raw_le = DATA_3D.astype('<i2').tobytes(order='F')
    raw_be = DATA_3D.astype('>i2').tobytes(order='F')
    fields_3d = 'type: short\ndimension: 3\nsizes: 3 4 5\nkinds: domain domain domain\n'

    write(
        'nrrd_raw_lps.nrrd',
        header(4, fields_3d + 'endian: little\nencoding: raw\n' + LPS_GEOMETRY + '\n') + raw_le,
    )
    write(
        'nrrd_gzip_ras.nrrd',
        header(4, fields_3d + 'endian: little\nencoding: gzip\n' + RAS_GEOMETRY + '\n')
        + gzip.compress(raw_le, mtime=0),
    )
    write(
        'nrrd_detached.nhdr',
        header(
            4,
            fields_3d
            + 'endian: little\nencoding: raw\n'
            + LPS_GEOMETRY
            + 'line skip: 1\nbyte skip: 4\ndata file: nrrd_detached.raw\n',
        ),
    )
    write('nrrd_detached.raw', b'skipped text line\n' + b'\xff' * 4 + raw_le)
    write(
        'nrrd_detached_be.nhdr',
        header(
            4,
            fields_3d
            + 'endian: big\nencoding: gzip\n'
            + LPS_GEOMETRY
            + 'data file: nrrd_detached_be.raw.gz\n',
        ),
    )
    write('nrrd_detached_be.raw.gz', gzip.compress(raw_be, mtime=0))

    # On disk axes: (component, x, y, z, time)
    disk_vector = np.arange(360, dtype='<f4').reshape((3, 3, 4, 5, 2), order='F')
    write(
        'nrrd_list_vector.nrrd',
        header(
            5,
            'type: float\n'
            'dimension: 5\n'
            'sizes: 3 3 4 5 2\n'
            'kinds: vector domain domain domain list\n'
            'endian: little\n'
            'encoding: gzip\n'
            'space: left-posterior-superior\n'
            'space directions: none (1.2,0.9,0) (-1.2,1.6,0) (0,0,2.5) none\n'
            'space origin: (10,-20,30)\n'
            'measurement frame: (0,1,0) (-1,0,0) (0,0,1)\n'
            'spacings: nan nan nan nan 0.5\n'
            'labels: "velocity" "" "" "" "time"\n'
            'description:=two\\nlines and one \\\\ backslash\n'
            'modality:=test\n'
            '\n',
        )
        + gzip.compress(disk_vector.tobytes(order='F'), mtime=0),
    )

    data_2d = np.arange(20, dtype=np.uint8).reshape((4, 5))
    write(
        'nrrd_2d.nrrd',
        header(
            5,
            'type: uchar\n'
            'dimension: 2\n'
            'sizes: 4 5\n'
            'kinds: domain domain\n'
            'encoding: raw\n'
            'space dimension: 2\n'
            'space directions: (0.5,0) (0,0.25)\n'
            'space origin: (1,2)\n'
            '\n',
        )
        + data_2d.tobytes(order='F'),
    )


if __name__ == '__main__':
    main()

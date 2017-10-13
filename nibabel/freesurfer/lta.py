import numpy as np
from ..wrapstruct import LabeledWrapStruct
from ..volumeutils import Recoder

transform_codes = Recoder((
    (0, 'LINEAR_VOX_TO_VOX'),
    (1, 'LINEAR_RAS_TO_RAS'),
    (2, 'LINEAR_PHYSVOX_TO_PHYSVOX'),
    (14, 'REGISTER_DAT'),
    (21, 'LINEAR_COR_TO_COR')),
    fields=('code', 'label'))


class StringBasedStruct(LabeledWrapStruct):
    def __init__(self,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        if binaryblock is not None and getattr(binaryblock, 'dtype',
                                               None) == self.dtype:
            self._structarr = binaryblock.copy()
            return
        super(StringBasedStruct, self).__init__(binaryblock, endianness, check)

    def __array__(self):
        return self._structarr


class VolumeGeometry(StringBasedStruct):
    template_dtype = np.dtype([
        ('valid', 'i4'),              # Valid values: 0, 1
        ('volume', 'i4', (3, 1)),     # width, height, depth
        ('voxelsize', 'f4', (3, 1)),  # xsize, ysize, zsize
        ('xras', 'f4', (3, 1)),       # x_r, x_a, x_s
        ('yras', 'f4', (3, 1)),       # y_r, y_a, y_s
        ('zras', 'f4', (3, 1)),       # z_r, z_a, z_s
        ('cras', 'f4', (3, 1)),       # c_r, c_a, c_s
        ('filename', 'U1024')])       # Not conformant (may be >1024 bytes)
    dtype = template_dtype

    def as_affine(self):
        affine = np.eye(4)
        sa = self.structarr
        A = np.hstack((sa['xras'], sa['yras'], sa['zras'])) * sa['voxelsize']
        b = sa['cras'] - A.dot(sa['volume']) / 2
        affine[:3, :3] = A
        affine[:3, [3]] = b
        return affine

    def to_string(self):
        sa = self.structarr
        lines = [
            'valid = {}  # volume info {:s}valid'.format(
                sa['valid'], '' if sa['valid'] else 'in'),
            'filename = {}'.format(sa['filename']),
            'volume = {:d} {:d} {:d}'.format(*sa['volume'].flatten()),
            'voxelsize = {:.15e} {:.15e} {:.15e}'.format(
                *sa['voxelsize'].flatten()),
            'xras   = {:.15e} {:.15e} {:.15e}'.format(*sa['xras'].flatten()),
            'yras   = {:.15e} {:.15e} {:.15e}'.format(*sa['yras'].flatten()),
            'zras   = {:.15e} {:.15e} {:.15e}'.format(*sa['zras'].flatten()),
            'cras   = {:.15e} {:.15e} {:.15e}'.format(*sa['cras'].flatten()),
        ]
        return '\n'.join(lines)

    @classmethod
    def from_image(klass, img):
        volgeom = klass()
        sa = volgeom.structarr
        sa['valid'] = 1
        sa['volume'][:, 0] = img.shape[:3]    # Assumes xyzt-ordered image
        sa['voxelsize'][:, 0] = img.header.get_zooms()[:3]
        A = img.affine[:3, :3]
        b = img.affine[:3, [3]]
        cols = A * (1 / sa['voxelsize'])
        sa['xras'] = cols[:, [0]]
        sa['yras'] = cols[:, [1]]
        sa['zras'] = cols[:, [2]]
        sa['cras'] = b + A.dot(sa['volume']) / 2
        try:
            sa['filename'] = img.file_map['image'].filename
        except:
            pass

        return volgeom

    @classmethod
    def from_string(klass, string):
        volgeom = klass()
        sa = volgeom.structarr
        lines = string.splitlines()
        for key in ('valid', 'filename', 'volume', 'voxelsize',
                    'xras', 'yras', 'zras', 'cras'):
            label, valstring = lines.pop(0).split(' = ')
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()],
                                dtype=klass.dtype[key])
            sa[key] = val.reshape(sa[key].shape) if val.size else ''

        return volgeom


class LinearTransform(StringBasedStruct):
    template_dtype = np.dtype([
        ('mean', 'f4', (3, 1)),       # x0, y0, z0
        ('sigma', 'f4'),
        ('m_L', 'f4', (4, 4)),
        ('m_dL', 'f4', (4, 4)),
        ('m_last_dL', 'f4', (4, 4)),
        ('src', VolumeGeometry),
        ('dst', VolumeGeometry),
        ('label', 'i4')])
    dtype = template_dtype

    def __getitem__(self, idx):
        val = super(LinearTransform, self).__getitem__(idx)
        if idx in ('src', 'dst'):
            val = VolumeGeometry(val)
        return val

    def to_string(self):
        sa = self.structarr
        lines = [
            'mean      = {:6.4f} {:6.4f} {:6.4f}'.format(
                *sa['mean'].flatten()),
            'sigma     = {:6.4f}'.format(float(sa['sigma'])),
            '1 4 4',
            ('{:18.15e} ' * 4).format(*sa['m_L'][0]),
            ('{:18.15e} ' * 4).format(*sa['m_L'][1]),
            ('{:18.15e} ' * 4).format(*sa['m_L'][2]),
            ('{:18.15e} ' * 4).format(*sa['m_L'][3]),
            'src volume info',
            self['src'].to_string(),
            'dst volume info',
            self['dst'].to_string(),
        ]
        return '\n'.join(lines)

    @classmethod
    def from_string(klass, string):
        lt = klass()
        sa = lt.structarr
        lines = string.splitlines()
        for key in ('mean', 'sigma'):
            label, valstring = lines.pop(0).split(' = ')
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()],
                                dtype=klass.dtype[key])
            sa[key] = val.reshape(sa[key].shape)
        assert lines.pop(0) == '1 4 4'
        val = np.genfromtxt([valstring.encode() for valstring in lines[:4]],
                            dtype='f4')
        sa['m_L'] = val
        lines = lines[4:]
        assert lines.pop(0) == 'src volume info'
        sa['src'] = np.asanyarray(VolumeGeometry.from_string('\n'.join(lines[:8])))
        lines = lines[8:]
        assert lines.pop(0) == 'dst volume info'
        sa['dst'] = np.asanyarray(VolumeGeometry.from_string('\n'.join(lines)))
        return lt


class LinearTransformArray(StringBasedStruct):
    template_dtype = np.dtype([
        ('type', 'i4'),
        ('nxforms', 'i4'),
        ('subject', 'U1024'),
        ('fscale', 'f4')])
    dtype = template_dtype
    _xforms = None

    def __init__(self,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        super(LinearTransformArray, self).__init__(binaryblock, endianness, check)
        self._xforms = [LinearTransform()
                        for _ in range(self.structarr['nxforms'])]

    def __getitem__(self, idx):
        if idx == 'xforms':
            return self._xforms
        if idx == 'nxforms':
            return len(self._xforms)
        return super(LinearTransformArray, self).__getitem__(idx)

    def to_string(self):
        code = int(self['type'])
        header = [
            'type      = {} # {}'.format(code, transform_codes.label[code]),
            'nxforms   = {}'.format(self['nxforms'])]
        xforms = [xfm.to_string() for xfm in self._xforms]
        footer = [
            'subject {}'.format(self['subject']),
            'fscale {:.6f}'.format(float(self['fscale']))]
        return '\n'.join(header + xforms + footer)

    @classmethod
    def from_string(klass, string):
        lta = klass()
        sa = lta.structarr
        lines = string.splitlines()
        for key in ('type', 'nxforms'):
            label, valstring = lines.pop(0).split(' = ')
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()],
                                dtype=klass.dtype[key])
            sa[key] = val.reshape(sa[key].shape) if val.size else ''
        for _ in range(sa['nxforms']):
            lta._xforms.append(
                LinearTransform.from_string('\n'.join(lines[:25])))
            lines = lines[25:]
        for key in ('subject', 'fscale'):
            # Optional keys
            if not lines[0].startswith(key):
                continue
            label, valstring = lines.pop(0).split(' ')
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()],
                                dtype=klass.dtype[key])
            sa[key] = val.reshape(sa[key].shape) if val.size else ''

        assert len(lta._xforms) == sa['nxforms']

        return lta

    @classmethod
    def from_fileobj(klass, fileobj, check=True):
        return klass.from_string(fileobj.read())

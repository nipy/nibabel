# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from __future__ import print_function, division

from copy import deepcopy
import os.path as op
import re

import numpy as np

from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .keywordonly import kw_only_meth
from .spatialimages import SpatialImage, SpatialHeader
from .volumeutils import Recoder

_attr_dic = {
    'string': str,
    'integer': int,
    'float': float
}

_endian_dict = {
    'LSB_FIRST': '<',
    'MSB_FIRST': '>',
}

_dtype_dict = {
    0: 'B',
    1: 'h',
    3: 'f',
    5: 'D',
}

_orient_dict = {
    0: 'R',
    1: 'L',
    2: 'P',
    3: 'A',
    4: 'I',
    5: 'S'
}

space_codes = Recoder((
    (0, 'unknown', ''),
    (1, 'scanner', 'ORIG'),
    (3, 'talairach', 'TLRC'),
    (4, 'mni', 'MNI')), fields=('code', 'label', 'space'))


class AFNIError(Exception):
    """ Error when reading AFNI files
    """


DATA_OFFSET = 0
TYPE_RE = re.compile('type\s*=\s*(string|integer|float)-attribute\s*\n')
NAME_RE = re.compile('name\s*=\s*(\w+)\s*\n')


def _unpack_var(var):
    """ Parses key, value pair from `var`

    Parameters
    ----------
    var : str
        Example: 'type = integer-attribute\nname = BRICK_TYPES\ncount = 1\n1\n'

    Returns
    -------
    (key, value)
        Example: ('BRICK_TYPES', [1])
    """
    # data type and key
    atype = TYPE_RE.findall(var)[0]
    aname = NAME_RE.findall(var)[0]
    atype = _attr_dic.get(atype, str)
    attr = ' '.join(var.strip().split('\n')[3:])
    if atype is not str:
        attr = [atype(f) for f in attr.split()]
        if len(attr) == 1:
            attr = attr[0]
    else:
        attr = attr.strip('\'~')

    return aname, attr


def _get_datatype(info):
    """ Gets datatype from `info` header information
    """
    bo = info['BYTEORDER_STRING']
    bt = info['BRICK_TYPES']
    if isinstance(bt, list):
        if len(np.unique(bt)) > 1:
            raise AFNIError('Can\'t load dataset with multiple data types.')
        else:
            bt = bt[0]
    bo = _endian_dict.get(bo, '=')
    bt = _dtype_dict.get(bt, None)
    if bt is None:
        raise AFNIError('Can\'t deduce image data type.')

    return np.dtype(bo + bt)


def parse_AFNI_header(fobj):
    """ Parses HEAD file for relevant information

    Parameters
    ----------
    fobj : file-object
        AFNI HEAD file object

    Returns
    -------
    all_info : dict
        Contains all the information from the HEAD file
    """
    head = fobj.read().split('\n\n')
    all_info = {key: value for key, value in map(_unpack_var, head)}

    return all_info


class AFNIArrayProxy(ArrayProxy):
    @kw_only_meth(2)
    def __init__(self, file_like, header, mmap=True, keep_file_open=None):
        """ Initialize AFNI array proxy

        Parameters
        ----------
        file_like : file-like object
            File-like object or filename. If file-like object, should implement
            at least ``read`` and ``seek``.
        header : AFNIHeader object
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading data.
            If False, do not try numpy ``memmap`` for data array.  If one of
            {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
            True gives the same behavior as ``mmap='c'``.  If `file_like`
            cannot be memory-mapped, ignore `mmap` value and read array from
            file.
        keep_file_open : { None, 'auto', True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed. If
            ``'auto'``, and the optional ``indexed_gzip`` dependency is
            present, a single file handle is created and persisted. If
            ``indexed_gzip`` is not available, behaviour is the same as if
            ``keep_file_open is False``. If ``file_like`` is an open file
            handle, this setting has no effect. The default value (``None``)
            will result in the value of ``KEEP_FILE_OPEN_DEFAULT`` being used.
        """
        super(AFNIArrayProxy, self).__init__(file_like,
                                             header,
                                             mmap=mmap,
                                             keep_file_open=keep_file_open)
        self._scaling = header.get_data_scaling()

    @property
    def scaling(self):
        return self._scaling

    def __array__(self):
        raw_data = self.get_unscaled()
        # apply volume specific scaling
        if self._scaling is not None:
            return raw_data * self._scaling.astype(self.dtype)

        return raw_data

    def __getitem__(self, slicer):
        raw_data = super(AFNIArrayProxy, self).__getitem__(slicer)
        # apply volume specific scaling
        if self._scaling is not None:
            scaling = self._scaling.copy()
            fake_data = strided_scalar(self._shape)
            _, scaling = np.broadcast_arrays(fake_data, scaling)
            raw_data = raw_data * scaling[slicer]

        return raw_data


class AFNIHeader(SpatialHeader):
    """ Class for AFNI header
    """
    def __init__(self, info):
        """
        Parameters
        ----------
        info : dict
            Information from AFNI HEAD file (as obtained by
            `parse_AFNI_header()`)
        """
        self.info = info
        dt = _get_datatype(self.info)
        super(AFNIHeader, self).__init__(data_dtype=dt,
                                         shape=self._calc_data_shape(),
                                         zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise AFNIError('Cannot create AFNIHeader from nothing.')
        if type(header) == klass:
            return header.copy()
        raise AFNIError('Cannot create AFNIHeader from non-AFNIHeader.')

    @classmethod
    def from_fileobj(klass, fileobj):
        info = parse_AFNI_header(fileobj)
        return klass(info)

    def copy(self):
        return AFNIHeader(deepcopy(self.info))

    def _calc_data_shape(self):
        """ Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        (x, y, z, t) : tuple of int
        """
        dset_rank = self.info['DATASET_RANK']
        shape = tuple(self.info['DATASET_DIMENSIONS'][:dset_rank[0]])
        n_vols = dset_rank[1]

        return shape + (n_vols,) if n_vols > 1 else shape

    def _calc_zooms(self):
        """ Get image zooms from header data

        Spatial axes are first three indices

        Returns
        -------
        zooms : tuple
        """
        xyz_step = tuple(np.abs(self.info['DELTA']))
        t_step = self.info.get('TAXIS_FLOATS', ())
        if len(t_step) > 0:
            t_step = (t_step[1],)

        return xyz_step + t_step

    def get_orient(self):
        """ Returns orientation of data

        Three letter string of {('L','R'),('P','A'),('I','S')} specifying
        data orientation

        Returns
        -------
        orient : str
        """
        orient = [_orient_dict[f][0] for f in self.info['ORIENT_SPECIFIC']]

        return ''.join(orient)

    def get_space(self):
        """ Returns space of dataset

        Returns
        -------
        space : str
        """
        listed_space = self.info.get('TEMPLATE_SPACE', 0)
        space = space_codes.label[listed_space]

        return space

    def get_affine(self):
        """ Returns affine of dataset
        """
        # AFNI default is RAI/DICOM order (i.e., RAI are - axis)
        # need to flip RA sign to align with nibabel RAS+ system
        affine = np.asarray(self.info['IJK_TO_DICOM_REAL']).reshape(3, 4)
        affine = np.row_stack((affine * [[-1], [-1], [1]],
                               [0, 0, 0, 1]))

        return affine

    def get_data_scaling(self):
        """ AFNI applies volume-specific data scaling
        """
        floatfacs = self.info.get('BRICK_FLOAT_FACS', None)
        if floatfacs is None or not np.any(floatfacs):
            return None
        scale = np.ones(self.info['DATASET_RANK'][1])
        floatfacs = np.asarray(floatfacs)
        scale[floatfacs.nonzero()] = floatfacs[floatfacs.nonzero()]

        return scale

    def get_slope_inter(self):
        """ Use `self.get_data_scaling()` instead
        """
        return None, None

    def get_data_offset(self):
        return DATA_OFFSET

    def get_volume_labels(self):
        """ Returns volume labels

        Returns
        -------
        labels : list of str
        """
        labels = self.info.get('BRICK_LABS', None)
        if labels is not None:
            labels = labels.split('~')

        return labels


class AFNIImage(SpatialImage):
    """ AFNI image file
    """

    header_class = AFNIHeader
    valid_exts = ('.brik', '.head')
    files_types = (('image', '.brik'), ('header', '.head'))
    _compressed_suffixes = ('.gz', '.bz2')
    makeable = False
    rw = False
    ImageArrayProxy = AFNIArrayProxy

    @classmethod
    @kw_only_meth(1)
    def from_file_map(klass, file_map, mmap=True):
        """
        Parameters
        ----------
        file_map : dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective REC and PAR files.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        """
        with file_map['header'].get_prepare_fileobj('rt') as hdr_fobj:
            hdr = klass.header_class.from_fileobj(
                hdr_fobj)
        brik_fobj = file_map['image'].get_prepare_fileobj()
        data = klass.ImageArrayProxy(brik_fobj, hdr,
                                     mmap=mmap)
        return klass(data, hdr.get_affine(), header=hdr, extra=None,
                     file_map=file_map)

    @classmethod
    @kw_only_meth(1)
    def from_filename(klass, filename, mmap=True):
        """
        Parameters
        ----------
        file_map : dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective REC and PAR files.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        """
        file_map = klass.filespec_to_file_map(filename)
        # only BRIK can be compressed, but `filespec_to_file_map` doesn't
        # handle that case; remove potential compression suffixes from HEAD
        head_fname = file_map['header'].filename
        if not op.exists(head_fname):
            for ext in klass._compressed_suffixes:
                head_fname = re.sub(ext, '', head_fname)
            file_map['header'].filename = head_fname
        # if HEAD is read in and BRIK is compressed, function won't detect the
        # compressed format; check for these cases
        if not op.exists(file_map['image'].filename):
            for ext in klass._compressed_suffixes:
                im_ext = file_map['image'].filename + ext
                if op.exists(im_ext):
                    file_map['image'].filename = im_ext
                    break
        return klass.from_file_map(file_map,
                                   mmap=mmap)

    load = from_filename


load = AFNIImage.load

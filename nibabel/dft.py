# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# Copyright (C) 2011 Christian Haselgrove
""" DICOM filesystem tools
"""


import os
from os.path import join as pjoin
import tempfile
import getpass
import logging
import warnings
import sqlite3

import numpy

from io import BytesIO

from .nifti1 import Nifti1Header

from .pydicom_compat import pydicom, read_file

logger = logging.getLogger('nibabel.dft')


class DFTError(Exception):
    "base class for DFT exceptions"


class CachingError(DFTError):
    "error while caching"


class VolumeError(DFTError):
    "unsupported volume parameter"


class InstanceStackError(DFTError):

    "bad series of instance numbers"

    def __init__(self, series, i, si):
        self.series = series
        self.i = i
        self.si = si
        return

    def __str__(self):
        fmt = 'expecting instance number %d, got %d'
        return fmt % (self.i + 1, self.si.instance_number)


class _Study(object):

    def __init__(self, d):
        self.uid = d['uid']
        self.date = d['date']
        self.time = d['time']
        self.comments = d['comments']
        self.patient_name = d['patient_name']
        self.patient_id = d['patient_id']
        self.patient_birth_date = d['patient_birth_date']
        self.patient_sex = d['patient_sex']
        self.series = None
        return

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'series' and val is None:
            val = []
            with _db_nochange() as c:
                c.execute("SELECT * FROM series WHERE study = ?", (self.uid, ))
                cols = [el[0] for el in c.description]
                for row in c:
                    d = dict(zip(cols, row))
                    val.append(_Series(d))
            self.series = val
        return val

    def patient_name_or_uid(self):
        if self.patient_name == '':
            return self.uid
        return self.patient_name


class _Series(object):

    def __init__(self, d):
        self.uid = d['uid']
        self.study = d['study']
        self.number = d['number']
        self.description = d['description']
        self.rows = d['rows']
        self.columns = d['columns']
        self.bits_allocated = d['bits_allocated']
        self.bits_stored = d['bits_stored']
        self.storage_instances = None
        return

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'storage_instances' and val is None:
            val = []
            with _db_nochange() as c:
                query = """SELECT *
                             FROM storage_instance
                            WHERE series = ?
                            ORDER BY instance_number"""
                c.execute(query, (self.uid, ))
                cols = [el[0] for el in c.description]
                for row in c:
                    d = dict(zip(cols, row))
                    val.append(_StorageInstance(d))
            self.storage_instances = val
        return val

    def as_png(self, index=None, scale_to_slice=True):
        import PIL.Image
        # For compatibility with older versions of PIL that did not
        # have `frombytes`:
        if hasattr(PIL.Image, 'frombytes'):
            frombytes = PIL.Image.frombytes
        else:
            frombytes = PIL.Image.fromstring

        if index is None:
            index = len(self.storage_instances) // 2
        d = self.storage_instances[index].dicom()
        data = d.pixel_array.copy()
        if self.bits_allocated != 16:
            raise VolumeError('unsupported bits allocated')
        if self.bits_stored != 12:
            raise VolumeError('unsupported bits stored')
        data = data / 16
        if scale_to_slice:
            min = data.min()
            max = data.max()
            data = data * 255 / (max - min)
        data = data.astype(numpy.uint8)
        im = frombytes('L', (self.rows, self.columns), data.tobytes())

        s = BytesIO()
        im.save(s, 'PNG')
        return s.getvalue()

    def png_size(self, index=None, scale_to_slice=True):
        return len(self.as_png(index=index, scale_to_slice=scale_to_slice))

    def as_nifti(self):
        if len(self.storage_instances) < 2:
            raise VolumeError('too few slices')
        d = self.storage_instances[0].dicom()
        if self.bits_allocated != 16:
            raise VolumeError('unsupported bits allocated')
        if self.bits_stored != 12:
            raise VolumeError('unsupported bits stored')
        data = numpy.ndarray((len(self.storage_instances), self.rows,
                              self.columns), dtype=numpy.int16)
        for (i, si) in enumerate(self.storage_instances):
            if i + 1 != si.instance_number:
                raise InstanceStackError(self, i, si)
            logger.info('reading %d/%d' % (i + 1, len(self.storage_instances)))
            d = self.storage_instances[i].dicom()
            data[i, :, :] = d.pixel_array

        d1 = self.storage_instances[0].dicom()
        dn = self.storage_instances[-1].dicom()

        pdi = d1.PixelSpacing[0]
        pdj = d1.PixelSpacing[0]
        pdk = d1.SpacingBetweenSlices

        cosi = d1.ImageOrientationPatient[0:3]
        cosi[0] = -1 * cosi[0]
        cosi[1] = -1 * cosi[1]
        cosj = d1.ImageOrientationPatient[3:6]
        cosj[0] = -1 * cosj[0]
        cosj[1] = -1 * cosj[1]

        pos_1 = numpy.array(d1.ImagePositionPatient)
        pos_1[0] = -1 * pos_1[0]
        pos_1[1] = -1 * pos_1[1]
        pos_n = numpy.array(dn.ImagePositionPatient)
        pos_n[0] = -1 * pos_n[0]
        pos_n[1] = -1 * pos_n[1]
        cosk = pos_n - pos_1
        cosk = cosk / numpy.linalg.norm(cosk)

        m = ((pdi * cosi[0], pdj * cosj[0], pdk * cosk[0], pos_1[0]),
             (pdi * cosi[1], pdj * cosj[1], pdk * cosk[1], pos_1[1]),
             (pdi * cosi[2], pdj * cosj[2], pdk * cosk[2], pos_1[2]),
             (0, 0, 0, 1))

        # Values are python Decimals in pydicom 0.9.7
        m = numpy.array(m, dtype=float)

        hdr = Nifti1Header(endianness='<')
        hdr.set_intent(0)
        hdr.set_qform(m, 1)
        hdr.set_xyzt_units(2, 8)
        hdr.set_data_dtype(numpy.int16)
        hdr.set_data_shape((self.columns, self.rows,
                            len(self.storage_instances)))

        s = BytesIO()
        hdr.write_to(s)

        return s.getvalue() + data.tobytes()

    def nifti_size(self):
        return 352 + 2 * len(self.storage_instances) * self.columns * self.rows


class _StorageInstance(object):

    def __init__(self, d):
        self.uid = d['uid']
        self.instance_number = d['instance_number']
        self.series = d['series']
        self.files = None
        return

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'files' and val is None:
            with _db_nochange() as c:
                query = """SELECT directory, name
                             FROM file
                            WHERE storage_instance = ?
                            ORDER BY directory, name"""
                c.execute(query, (self.uid, ))
                val = ['%s/%s' % tuple(row) for row in c]
            self.files = val
        return val

    def dicom(self):
        return read_file(self.files[0])


class _db_nochange:
    """context guard for read-only database access"""

    def __enter__(self):
        self.c = DB.cursor()
        return self.c

    def __exit__(self, type, value, traceback):
        if type is None:
            self.c.close()
        DB.rollback()
        return


class _db_change:
    """context guard for database access requiring a commit"""

    def __enter__(self):
        self.c = DB.cursor()
        return self.c

    def __exit__(self, type, value, traceback):
        if type is None:
            self.c.close()
            DB.commit()
        else:
            DB.rollback()
        return


def _get_subdirs(base_dir, files_dict=None, followlinks=False):
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(base_dir, followlinks=followlinks):
        abs_dir = os.path.realpath(dirpath)
        if abs_dir in dirs:
            raise CachingError(f'link cycle detected under {base_dir}')
        dirs.append(abs_dir)
        if files_dict is not None:
            files_dict[abs_dir] = filenames
    return dirs


def update_cache(base_dir, followlinks=False):
    mtimes = {}
    files_by_dir = {}
    dirs = _get_subdirs(base_dir, files_by_dir, followlinks)
    for d in dirs:
        os.stat(d)
        mtimes[d] = os.stat(d).st_mtime
    with _db_nochange() as c:
        c.execute("SELECT path, mtime FROM directory")
        db_mtimes = dict(c)
        c.execute("SELECT uid FROM study")
        studies = [row[0] for row in c]
        c.execute("SELECT uid FROM series")
        series = [row[0] for row in c]
        c.execute("SELECT uid FROM storage_instance")
        storage_instances = [row[0] for row in c]
    with _db_change() as c:
        for dir in sorted(mtimes.keys()):
            if dir in db_mtimes and mtimes[dir] <= db_mtimes[dir]:
                continue
            logger.debug(f'updating {dir}')
            _update_dir(c, dir, files_by_dir[dir], studies, series,
                        storage_instances)
            if dir in db_mtimes:
                query = "UPDATE directory SET mtime = ? WHERE path = ?"
                c.execute(query, (mtimes[dir], dir))
            else:
                query = "INSERT INTO directory (path, mtime) VALUES (?, ?)"
                c.execute(query, (dir, mtimes[dir]))
    return


def get_studies(base_dir=None, followlinks=False):
    if base_dir is not None:
        update_cache(base_dir, followlinks)
    if base_dir is None:
        with _db_nochange() as c:
            c.execute("SELECT * FROM study")
            studies = []
            cols = [el[0] for el in c.description]
            for row in c:
                d = dict(zip(cols, row))
                studies.append(_Study(d))
        return studies
    query = """SELECT study
                 FROM series
                WHERE uid IN (SELECT series
                                FROM storage_instance
                               WHERE uid IN (SELECT storage_instance
                                               FROM file
                                              WHERE directory = ?))"""
    with _db_nochange() as c:
        study_uids = {}
        for dir in _get_subdirs(base_dir, followlinks=followlinks):
            c.execute(query, (dir, ))
            for row in c:
                study_uids[row[0]] = None
        studies = []
        for uid in study_uids:
            c.execute("SELECT * FROM study WHERE uid = ?", (uid, ))
            cols = [el[0] for el in c.description]
            d = dict(zip(cols, c.fetchone()))
            studies.append(_Study(d))
    return studies


def _update_dir(c, dir, files, studies, series, storage_instances):
    logger.debug(f'Updating directory {dir}')
    c.execute("SELECT name, mtime FROM file WHERE directory = ?", (dir, ))
    db_mtimes = dict(c)
    for fname in db_mtimes:
        if fname not in files:
            logger.debug(f'    remove {fname}')
            c.execute("DELETE FROM file WHERE directory = ? AND name = ?",
                      (dir, fname))
    for fname in files:
        mtime = os.lstat(f'{dir}/{fname}').st_mtime
        if fname in db_mtimes and mtime <= db_mtimes[fname]:
            logger.debug(f'    okay {fname}')
        else:
            logger.debug(f'    update {fname}')
            si_uid = _update_file(c, dir, fname, studies, series,
                                  storage_instances)
            if fname not in db_mtimes:
                query = """INSERT INTO file (directory,
                                             name,
                                             mtime,
                                             storage_instance)
                           VALUES (?, ?, ?, ?)"""
                c.execute(query, (dir, fname, mtime, si_uid))
            else:
                query = """UPDATE file
                              SET mtime = ?, storage_instance = ?
                            WHERE directory = ? AND name = ?"""
                c.execute(query, (mtime, si_uid, dir, fname))
    return


def _update_file(c, path, fname, studies, series, storage_instances):
    try:
        do = read_file(f'{path}/{fname}')
    except pydicom.filereader.InvalidDicomError:
        logger.debug('        not a DICOM file')
        return None
    try:
        study_comments = do.StudyComments
    except AttributeError:
        study_comments = ''
    try:
        logger.debug(f'        storage instance {do.SOPInstanceUID}')
        if str(do.StudyInstanceUID) not in studies:
            query = """INSERT INTO study (uid,
                                          date,
                                          time,
                                          comments,
                                          patient_name,
                                          patient_id,
                                          patient_birth_date,
                                          patient_sex)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
            params = (str(do.StudyInstanceUID),
                      do.StudyDate,
                      do.StudyTime,
                      study_comments,
                      str(do.PatientName),
                      do.PatientID,
                      do.PatientBirthDate,
                      do.PatientSex)
            c.execute(query, params)
            studies.append(str(do.StudyInstanceUID))
        if str(do.SeriesInstanceUID) not in series:
            query = """INSERT INTO series (uid,
                                           study,
                                           number,
                                           description,
                                           rows,
                                           columns,
                                           bits_allocated,
                                           bits_stored)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
            params = (str(do.SeriesInstanceUID),
                      str(do.StudyInstanceUID),
                      do.SeriesNumber,
                      do.SeriesDescription,
                      do.Rows,
                      do.Columns,
                      do.BitsAllocated,
                      do.BitsStored)
            c.execute(query, params)
            series.append(str(do.SeriesInstanceUID))
        if str(do.SOPInstanceUID) not in storage_instances:
            query = """INSERT INTO storage_instance (uid, instance_number, series)
                       VALUES (?, ?, ?)"""
            params = (str(do.SOPInstanceUID), do.InstanceNumber,
                      str(do.SeriesInstanceUID))
            c.execute(query, params)
            storage_instances.append(str(do.SOPInstanceUID))
    except AttributeError as data:
        logger.debug(f'        {data}')
        return None
    return str(do.SOPInstanceUID)


def clear_cache():
    with _db_change() as c:
        c.execute("DELETE FROM file")
        c.execute("DELETE FROM directory")
        c.execute("DELETE FROM storage_instance")
        c.execute("DELETE FROM series")
        c.execute("DELETE FROM study")
    return


CREATE_QUERIES = (
    """CREATE TABLE study (uid TEXT NOT NULL PRIMARY KEY,
                           date TEXT NOT NULL,
                           time TEXT NOT NULL,
                           comments TEXT NOT NULL,
                           patient_name TEXT NOT NULL,
                           patient_id TEXT NOT NULL,
                           patient_birth_date TEXT NOT NULL,
                           patient_sex TEXT NOT NULL)""",
    """CREATE TABLE series (uid TEXT NOT NULL PRIMARY KEY,
                            study TEXT NOT NULL REFERENCES study,
                            number TEXT NOT NULL,
                            description TEXT NOT NULL,
                            rows INTEGER NOT NULL,
                            columns INTEGER NOT NULL,
                            bits_allocated INTEGER NOT NULL,
                            bits_stored INTEGER NOT NULL)""",
    """CREATE TABLE storage_instance (uid TEXT NOT NULL PRIMARY KEY,
                                      instance_number INTEGER NOT NULL,
                                      series TEXT NOT NULL references series)""",
    """CREATE TABLE directory (path TEXT NOT NULL PRIMARY KEY,
                               mtime INTEGER NOT NULL)""",
    """CREATE TABLE file (directory TEXT NOT NULL REFERENCES directory,
                          name TEXT NOT NULL,
                          mtime INTEGER NOT NULL,
                          storage_instance TEXT DEFAULT NULL REFERENCES storage_instance,
                          PRIMARY KEY (directory, name))""")
DB_FNAME = pjoin(tempfile.gettempdir(), f'dft.{getpass.getuser()}.sqlite')
DB = None


def _init_db(verbose=True):
    """ Initialize database """
    if verbose:
        logger.info('db filename: ' + DB_FNAME)
    global DB
    DB = sqlite3.connect(DB_FNAME, check_same_thread=False)
    with _db_change() as c:
        c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'")
        if c.fetchone()[0] == 0:
            logger.debug('create')
            for q in CREATE_QUERIES:
                c.execute(q)


if os.name == 'nt':
    warnings.warn('dft needs FUSE which is not available for windows')
else:
    _init_db()
# eof

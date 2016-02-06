# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Tests for data module '''
from __future__ import division, print_function, absolute_import
import os
from os.path import join as pjoin
from os import environ as env
import sys
import tempfile

from ..data import (get_data_path, find_data_dir,
                    DataError, _cfg_value, make_datasource,
                    Datasource, VersionedDatasource, Bomber,
                    datasource_or_bomber)

from ..tmpdirs import TemporaryDirectory

from .. import data as nibd

from nose import with_setup

from nose.tools import (assert_equal, assert_raises, raises, assert_false)

from .test_environment import (setup_environment,
                               teardown_environment,
                               DATA_KEY,
                               USER_KEY)

DATA_FUNCS = {}


def setup_data_env():
    setup_environment()
    global DATA_FUNCS
    DATA_FUNCS['home_dir_func'] = nibd.get_nipy_user_dir
    DATA_FUNCS['sys_dir_func'] = nibd.get_nipy_system_dir
    DATA_FUNCS['path_func'] = nibd.get_data_path


def teardown_data_env():
    teardown_environment()
    nibd.get_nipy_user_dir = DATA_FUNCS['home_dir_func']
    nibd.get_nipy_system_dir = DATA_FUNCS['sys_dir_func']
    nibd.get_data_path = DATA_FUNCS['path_func']


# decorator to use setup, teardown environment
with_environment = with_setup(setup_data_env, teardown_data_env)


def test_datasource():
    # Tests for DataSource
    pth = pjoin('some', 'path')
    ds = Datasource(pth)
    yield assert_equal, ds.get_filename('unlikeley'), pjoin(pth, 'unlikeley')
    yield (assert_equal, ds.get_filename('un', 'like', 'ley'),
           pjoin(pth, 'un', 'like', 'ley'))


def test_versioned():
    with TemporaryDirectory() as tmpdir:
        yield (assert_raises,
               DataError,
               VersionedDatasource,
               tmpdir)
        tmpfile = pjoin(tmpdir, 'config.ini')
        # ini file, but wrong section
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[SOMESECTION]\n')
            fobj.write('version = 0.1\n')
        yield (assert_raises,
               DataError,
               VersionedDatasource,
               tmpdir)
        # ini file, but right section, wrong key
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('somekey = 0.1\n')
        yield (assert_raises,
               DataError,
               VersionedDatasource,
               tmpdir)
        # ini file, right section and key
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1\n')
        vds = VersionedDatasource(tmpdir)
        yield assert_equal, vds.version, '0.1'
        yield assert_equal, vds.version_no, 0.1
        yield assert_equal, vds.major_version, 0
        yield assert_equal, vds.minor_version, 1
        yield assert_equal, vds.get_filename('config.ini'), tmpfile
        # ini file, right section and key, funny value
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1.2.dev\n')
        vds = VersionedDatasource(tmpdir)
        yield assert_equal, vds.version, '0.1.2.dev'
        yield assert_equal, vds.version_no, 0.1
        yield assert_equal, vds.major_version, 0
        yield assert_equal, vds.minor_version, 1


def test__cfg_value():
    # no file, return ''
    yield assert_equal, _cfg_value('/implausible_file'), ''
    # try files
    try:
        fd, tmpfile = tempfile.mkstemp()
        fobj = os.fdopen(fd, 'wt')
        # wrong section, right key
        fobj.write('[strange section]\n')
        fobj.write('path = /some/path\n')
        fobj.flush()
        yield assert_equal, _cfg_value(tmpfile), ''
        # right section, wrong key
        fobj.write('[DATA]\n')
        fobj.write('funnykey = /some/path\n')
        fobj.flush()
        yield assert_equal, _cfg_value(tmpfile), ''
        # right section, right key
        fobj.write('path = /some/path\n')
        fobj.flush()
        yield assert_equal, _cfg_value(tmpfile), '/some/path'
        fobj.close()
    finally:
        try:
            os.unlink(tmpfile)
        except:
            pass


@with_environment
def test_data_path():
    # wipe out any sources of data paths
    if DATA_KEY in env:
        del env[DATA_KEY]
    if USER_KEY in env:
        del os.environ[USER_KEY]
    fake_user_dir = '/user/path'
    nibd.get_nipy_system_dir = lambda: '/unlikely/path'
    nibd.get_nipy_user_dir = lambda: fake_user_dir
    # now we should only have anything pointed to in the user's dir
    old_pth = get_data_path()
    # We should have only sys.prefix and, iff sys.prefix == /usr,
    # '/usr/local'.  This last to is deal with Debian patching to
    # distutils.
    def_dirs = [pjoin(sys.prefix, 'share', 'nipy')]
    if sys.prefix == '/usr':
        def_dirs.append(pjoin('/usr/local', 'share', 'nipy'))
    assert_equal(old_pth, def_dirs + ['/user/path'])
    # then we'll try adding some of our own
    tst_pth = '/a/path' + os.path.pathsep + '/b/ path'
    tst_list = ['/a/path', '/b/ path']
    # First, an environment variable
    os.environ[DATA_KEY] = tst_list[0]
    assert_equal(get_data_path(), tst_list[:1] + old_pth)
    os.environ[DATA_KEY] = tst_pth
    assert_equal(get_data_path(), tst_list + old_pth)
    del os.environ[DATA_KEY]
    # Next, make a fake user directory, and put a file in there
    with TemporaryDirectory() as tmpdir:
        tmpfile = pjoin(tmpdir, 'config.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s' % tst_pth)
        nibd.get_nipy_user_dir = lambda: tmpdir
        assert_equal(get_data_path(), tst_list + def_dirs + [tmpdir])
    nibd.get_nipy_user_dir = lambda: fake_user_dir
    assert_equal(get_data_path(), old_pth)
    # with some trepidation, the system config files
    with TemporaryDirectory() as tmpdir:
        nibd.get_nipy_system_dir = lambda: tmpdir
        tmpfile = pjoin(tmpdir, 'an_example.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s\n' % tst_pth)
        tmpfile = pjoin(tmpdir, 'another_example.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s\n' % '/path/two')
        assert_equal(get_data_path(),
                     tst_list + ['/path/two'] + old_pth)


def test_find_data_dir():
    here, fname = os.path.split(__file__)
    # here == '<rootpath>/nipy/utils/tests'
    under_here, subhere = os.path.split(here)
    # under_here == '<rootpath>/nipy/utils'
    # subhere = 'tests'
    # fails with non-existant path
    yield (assert_raises,
           DataError,
           find_data_dir,
           [here],
           'implausible',
           'directory')
    # fails with file, when directory expected
    yield (assert_raises,
           DataError,
           find_data_dir,
           [here],
           fname)
    # passes with directory that exists
    dd = find_data_dir([under_here], subhere)
    yield assert_equal, dd, here
    # and when one path in path list does not work
    dud_dir = pjoin(under_here, 'implausible')
    dd = find_data_dir([dud_dir, under_here], subhere)
    yield assert_equal, dd, here


@with_environment
def test_make_datasource():
    pkg_def = dict(
        relpath='pkg')
    with TemporaryDirectory() as tmpdir:
        nibd.get_data_path = lambda: [tmpdir]
        yield (assert_raises,
               DataError,
               make_datasource,
               pkg_def)
        pkg_dir = pjoin(tmpdir, 'pkg')
        os.mkdir(pkg_dir)
        yield (assert_raises,
               DataError,
               make_datasource,
               pkg_def)
        tmpfile = pjoin(pkg_dir, 'config.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1\n')
        ds = make_datasource(pkg_def, data_path=[tmpdir])
        yield assert_equal, ds.version, '0.1'


@raises(DataError)
def test_bomber():
    b = Bomber('bomber example', 'a message')
    b.any_attribute  # no error


def test_bomber_inspect():
    b = Bomber('bomber example', 'a message')
    assert_false(hasattr(b, 'any_attribute'))


@with_environment
def test_datasource_or_bomber():
    pkg_def = dict(
        relpath='pkg')
    with TemporaryDirectory() as tmpdir:
        nibd.get_data_path = lambda: [tmpdir]
        ds = datasource_or_bomber(pkg_def)
        yield (assert_raises,
               DataError,
               getattr,
               ds,
               'get_filename')
        pkg_dir = pjoin(tmpdir, 'pkg')
        os.mkdir(pkg_dir)
        tmpfile = pjoin(pkg_dir, 'config.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.2\n')
        ds = datasource_or_bomber(pkg_def)
        ds.get_filename('some_file.txt')
        # check that versioning works
        pkg_def['min version'] = '0.2'
        ds = datasource_or_bomber(pkg_def)  # OK
        ds.get_filename('some_file.txt')
        pkg_def['min version'] = '0.3'
        ds = datasource_or_bomber(pkg_def)  # not OK
        yield (assert_raises,
               DataError,
               getattr,
               ds,
               'get_filename')

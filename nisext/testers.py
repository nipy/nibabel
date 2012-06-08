''' Test package information in various install settings

The routines here install the package from source directories, zips or eggs, and
check these installations by running tests, checking version information,
looking for files that were not copied over.

The typical use for this module is as a Makefile target.  For example, here are
the Makefile targets from nibabel::

    # Check for files not installed
    check-files:
        $(PYTHON) -c 'from nisext.testers import check_files; check_files("nibabel")'

    # Print out info for possible install methods
    check-version-info:
        $(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("nibabel")'

    # Run tests from installed code
    installed-tests:
        $(PYTHON) -c 'from nisext.testers import tests_installed; tests_installed("nibabel")'

    # Run tests from installed code
    sdist-tests:
        $(PYTHON) -c 'from nisext.testers import sdist_tests; sdist_tests("nibabel")'

    # Run tests from binary egg
    bdist-egg-tests:
        $(PYTHON) -c 'from nisext.testers import bdist_egg_tests; bdist_egg_tests("nibabel")'

'''

import os
from os.path import join as pjoin, abspath
from glob import glob
import shutil
import tempfile
import zipfile
import re
from subprocess import call
from functools import partial

my_call = partial(call, shell=True)

PY_LIB_SDIR = 'pylib'

def run_mod_cmd(mod_name, pkg_path, cmd):
    ''' Run command in own process in anonymous path

    Parameters
    ----------
    mod_name : str
        Name of module to import - e.g. 'nibabel'
    pkg_path : str
        directory containing `mod_name` package.  Typically that will be the
        directory containing the e.g. 'nibabel' directory.
    '''
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
        my_call('python -c "import sys; sys.path.insert(1,\'%s\'); '
                'import %s;'
                'print(%s.__file__);'
                '%s"' % (pkg_path,
                         mod_name,
                         mod_name,
                         cmd))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)


def zip_extract_all(fname, path=None):
    ''' Extract all members from zipfile

    Deals with situation where the directory is stored in the zipfile as a name,
    as well as files that have to go into this directory.
    '''
    zf = zipfile.ZipFile(fname)
    members = zf.namelist()
    # Remove members that are just bare directories
    members = [m for m in members if not m.endswith('/')]
    for zipinfo in members:
        zf.extract(zipinfo, path, None)


def install_from_to(from_dir, to_dir, py_lib_sdir=PY_LIB_SDIR, bin_sdir='bin'):
    """ Install package in `from_dir` to standard location in `to_dir`

    Parameters
    ----------
    from_dir : str
        path containing files to install with ``python setup.py ...``
    to_dir : str
        prefix path to which files will be installed, as in ``python setup.py
        install --prefix=to_dir``
    py_lib_sdir : str, optional
        subdirectory within `to_dir` to which library code will be installed
    bin_sdir : str, optional
        subdirectory within `to_dir` to which scripts will be installed
    """
    site_pkgs_path = os.path.join(to_dir, py_lib_sdir)
    py_lib_locs = ' --install-purelib=%s --install-platlib=%s' % (
        site_pkgs_path, site_pkgs_path)
    pwd = os.path.abspath(os.getcwd())
    try:
        os.chdir(from_dir)
        my_call('python setup.py --quiet install --prefix=%s %s' % (to_dir,
                                                                    py_lib_locs))
    finally:
        os.chdir(pwd)


def install_from_zip(zip_fname, install_path, pkg_finder=None,
                     py_lib_sdir=PY_LIB_SDIR,
                     script_sdir='bin'):
    """ Install package from zip file `zip_fname`

    Parameters
    ----------
    zip_fname : str
        filename of zip file containing package code
    install_path : str
        output prefix at which to install package
    pkg_finder : None or callable, optional
        If None, assume zip contains ``setup.py`` at the top level.  Otherwise,
        find directory containing ``setup.py`` with ``pth =
        pkg_finder(unzip_path)`` where ``unzip_path`` is the path to which we
        have unzipped the zip file contents.
    py_lib_sdir : str, optional
        subdirectory to which to write the library code from the package.  Thus
        if package called ``nibabel``, the written code will be in
        ``<install_path>/<py_lib_sdir>/nibabel
    script_sdir : str, optional
        subdirectory to which we write the installed scripts.  Thus scripts will
        be written to ``<install_path>/<script_sdir>
    """
    unzip_path = tempfile.mkdtemp()
    try:
        # Zip may unpack module into current directory
        zip_extract_all(zip_fname, unzip_path)
        if pkg_finder is None:
            from_path = unzip_path
        else:
            from_path = pkg_finder(unzip_path)
        install_from_to(from_path, install_path, py_lib_sdir, script_sdir)
    finally:
        shutil.rmtree(unzip_path)


def contexts_print_info(mod_name, repo_path, install_path):
    ''' Print result of get_info from different installation routes

    Runs installation from:

    * git archive zip file
    * with setup.py install from repository directory
    * just running code from repository directory

    and prints out result of get_info in each case.  There will be many files
    written into `install_path` that you may want to clean up somehow.

    Parameters
    ----------
    mod_name : str
       package name that will be installed, and tested
    repo_path : str
       path to location of git repository
    install_path : str
       path into which to install temporary installations
    '''
    site_pkgs_path = os.path.join(install_path, PY_LIB_SDIR)
    # first test archive
    pwd = os.path.abspath(os.getcwd())
    out_fname = pjoin(install_path, 'test.zip')
    try:
        os.chdir(repo_path)
        my_call('git archive --format zip -o %s HEAD' % out_fname)
    finally:
        os.chdir(pwd)
    install_from_zip(out_fname, install_path, None)
    cmd_str = 'print(%s.get_info())' % mod_name
    run_mod_cmd(mod_name, site_pkgs_path, cmd_str)
    # now test install into a directory from the repository
    install_from_to(repo_path, install_path, PY_LIB_SDIR)
    run_mod_cmd(mod_name, site_pkgs_path, cmd_str)
    # test from development tree
    run_mod_cmd(mod_name, repo_path, cmd_str)
    return


def info_from_here(mod_name):
    ''' Run info context checks starting in working directory

    Runs checks from current working directory, installing temporary
    installations into a new temporary directory

    Parameters
    ----------
    mod_name : str
       package name that will be installed, and tested
    '''
    repo_path = os.path.abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        contexts_print_info(mod_name, repo_path, install_path)
    finally:
        shutil.rmtree(install_path)


def tests_installed(mod_name, source_path=None):
    """ Install from `source_path` into temporary directory; run tests

    Parameters
    ----------
    mod_name : str
        name of module - e.g. 'nibabel'
    source_path : None or str
        Path from which to install.  If None, defaults to working directory
    """
    if source_path is None:
        source_path = os.path.abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    site_pkgs_path = pjoin(install_path, PY_LIB_SDIR)
    try:
        install_from_to(source_path, install_path, PY_LIB_SDIR)
        run_mod_cmd(mod_name, site_pkgs_path, mod_name + '.test()')
    finally:
        shutil.rmtree(install_path)

# Tell nose this is not a test
tests_installed.__test__ = False


def check_installed_files(repo_mod_path, install_mod_path):
    """ Check files in `repo_mod_path` are installed at `install_mod_path`

    At the moment, all this does is check that all the ``*.py`` files in
    `repo_mod_path` are installed at `install_mod_path`.

    Parameters
    ----------
    repo_mod_path : str
        repository path containing package files, e.g. <nibabel-repo>/nibabel>
    install_mod_path : str
        path at which package has been installed.  This is the path where the
        root package ``__init__.py`` lives.

    Return
    ------
    uninstalled : list
        list of files that should have been installed, but have not been
        installed
    """
    return missing_from(repo_mod_path, install_mod_path, filter=r"\.py$")


def missing_from(path0, path1, filter=None):
    """ Return filenames present in `path0` but not in `path1`

    Parameters
    ----------
    path0 : str
        path which contains all files of interest
    path1 : str
        path which should contain all files of interest
    filter : None or str or regexp, optional
        A successful result from ``filter.search(fname)`` means the file is of
        interest.  None means all files are of interest

    Returns
    -------
    path1_missing : list
        list of all files missing from `path1` that are in `path0` at the same
        relative path.
    """
    if not filter is None:
        filter = re.compile(filter)
    uninstalled = []
    # Walk directory tree to get py files
    for dirpath, dirnames, filenames in os.walk(path0):
        out_dirpath = dirpath.replace(path0, path1)
        for fname in filenames:
            if not filter is None and filter.search(fname) is None:
                continue
            equiv_fname = os.path.join(out_dirpath, fname)
            if not os.path.isfile(equiv_fname):
                uninstalled.append(pjoin(dirpath, fname))
    return uninstalled


def check_files(mod_name, repo_path=None, scripts_sdir='bin'):
    """ Print library and script files not picked up during install
    """
    if repo_path is None:
        repo_path = abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    repo_mod_path = pjoin(repo_path, mod_name)
    installed_mod_path = pjoin(install_path, PY_LIB_SDIR, mod_name)
    repo_bin = pjoin(repo_path, 'bin')
    installed_bin = pjoin(install_path, 'bin')
    try:
        zip_fname = make_dist(repo_path,
                              install_path,
                              'sdist --formats=zip',
                              '*.zip')
        pf = get_sdist_finder(mod_name)
        install_from_zip(zip_fname, install_path, pf, PY_LIB_SDIR, scripts_sdir)
        lib_misses = missing_from(repo_mod_path, installed_mod_path, r"\.py$")
        script_misses = missing_from(repo_bin, installed_bin)
    finally:
        shutil.rmtree(install_path)
    if lib_misses:
        print "Missed library files: ", ', '.join(lib_misses)
    else:
        print "You got all the library files"
    if script_misses:
        print "Missed script files: ", ', '.join(script_misses)
    else:
        print "You got all the script files"
    return len(lib_misses) > 0 or len(script_misses) > 0


def get_sdist_finder(mod_name):
    """ Return function finding sdist source directory for `mod_name`
    """
    def pf(pth):
        pkg_dirs = glob(pjoin(pth, mod_name + '-*'))
        if len(pkg_dirs) != 1:
            raise OSError('There must be one and only one package dir')
        return pkg_dirs[0]
    return pf


def sdist_tests(mod_name, repo_path=None):
    """ Make sdist zip, install from it, and run tests """
    if repo_path is None:
        repo_path = abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        zip_fname = make_dist(repo_path,
                              install_path,
                              'sdist --formats=zip',
                              '*.zip')
        site_pkgs_path = pjoin(install_path, PY_LIB_SDIR)
        pf = get_sdist_finder(mod_name)
        install_from_zip(zip_fname, install_path, pf, PY_LIB_SDIR)
        run_mod_cmd(mod_name, site_pkgs_path, mod_name + '.test()')
    finally:
        shutil.rmtree(install_path)

sdist_tests.__test__ = False


def bdist_egg_tests(mod_name, repo_path=None):
    """ Make bdist_egg, unzip it, and run tests from result """
    if repo_path is None:
        repo_path = abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        zip_fname = make_dist(repo_path,
                              install_path,
                              'bdist_egg',
                              '*.egg')
        zip_extract_all(zip_fname, install_path)
        run_mod_cmd(mod_name, install_path, mod_name + '.test()')
    finally:
        shutil.rmtree(install_path)

bdist_egg_tests.__test__ = False


def make_dist(repo_path, out_dir, setup_params, zipglob):
    """ Create distutils distribution file

    Parameters
    ----------
    repo_path : str
        path to repository containing code and ``setup.py``
    out_dir : str
        path to which to write new distribution file
    setup_params: str
        parameters to pass to ``setup.py`` to create distribution.
    zipglob : str
        glob identifying expected output file.

    Returns
    -------
    out_fname : str
        filename of generated distribution file

    Examples
    --------
    Make, return a zipped sdist::

      make_dist('/path/to/repo', '/tmp/path', 'sdist --formats=zip', '*.zip')

    Make, return a binary egg::

      make_dist('/path/to/repo', '/tmp/path', 'bdist_egg', '*.egg')
    """
    pwd = os.path.abspath(os.getcwd())
    try:
        os.chdir(repo_path)
        my_call('python setup.py %s --dist-dir=%s'
                % (setup_params, out_dir))
        zips = glob(pjoin(out_dir, zipglob))
        if len(zips) != 1:
            raise OSError('There must be one and only one %s file, '
                          'but I found "%s"' %
                          (zipglob, ': '.join(zips)))
    finally:
        os.chdir(pwd)
    return zips[0]

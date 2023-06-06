"""Distutils / setuptools helpers"""

import os
from configparser import ConfigParser
from distutils import log
from distutils.command.build_py import build_py
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext


def get_comrec_build(pkg_dir, build_cmd=build_py):
    """Return extended build command class for recording commit

    The extended command tries to run git to find the current commit, getting
    the empty string if it fails.  It then writes the commit hash into a file
    in the `pkg_dir` path, named ``COMMIT_INFO.txt``.

    In due course this information can be used by the package after it is
    installed, to tell you what commit it was installed from if known.

    To make use of this system, you need a package with a COMMIT_INFO.txt file -
    e.g. ``myproject/COMMIT_INFO.txt`` - that might well look like this::

        # This is an ini file that may contain information about the code state
        [commit hash]
        # The line below may contain a valid hash if it has been substituted during 'git archive'
        archive_subst_hash=$Format:%h$
        # This line may be modified by the install process
        install_hash=

    The COMMIT_INFO file above is also designed to be used with git substitution
    - so you probably also want a ``.gitattributes`` file in the root directory
    of your working tree that contains something like this::

       myproject/COMMIT_INFO.txt export-subst

    That will cause the ``COMMIT_INFO.txt`` file to get filled in by ``git
    archive`` - useful in case someone makes such an archive - for example with
    via the github 'download source' button.

    Although all the above will work as is, you might consider having something
    like a ``get_info()`` function in your package to display the commit
    information at the terminal.  See the ``pkg_info.py`` module in the nipy
    package for an example.
    """

    class MyBuildPy(build_cmd):
        """Subclass to write commit data into installation tree"""

        def run(self):
            build_cmd.run(self)
            import subprocess

            proc = subprocess.Popen(
                'git rev-parse --short HEAD',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            repo_commit, _ = proc.communicate()
            # Fix for python 3
            repo_commit = str(repo_commit)
            # We write the installation commit even if it's empty
            cfg_parser = ConfigParser()
            cfg_parser.read(pjoin(pkg_dir, 'COMMIT_INFO.txt'))
            cfg_parser.set('commit hash', 'install_hash', repo_commit)
            out_pth = pjoin(self.build_lib, pkg_dir, 'COMMIT_INFO.txt')
            cfg_parser.write(open(out_pth, 'wt'))

    return MyBuildPy


def _add_append_key(in_dict, key, value):
    """Helper for appending dependencies to setuptools args"""
    # If in_dict[key] does not exist, create it
    # If in_dict[key] is a string, make it len 1 list of strings
    # Append value to in_dict[key] list
    if key not in in_dict:
        in_dict[key] = []
    elif isinstance(in_dict[key], str):
        in_dict[key] = [in_dict[key]]
    in_dict[key].append(value)


# Dependency checks
def package_check(
    pkg_name,
    version=None,
    optional=False,
    checker=LooseVersion,
    version_getter=None,
    messages=None,
    setuptools_args=None,
):
    """Check if package `pkg_name` is present and has good enough version

    Has two modes of operation.  If `setuptools_args` is None (the default),
    raise an error for missing non-optional dependencies and log warnings for
    missing optional dependencies.  If `setuptools_args` is a dict, then fill
    ``install_requires`` key value with any missing non-optional dependencies,
    and the ``extras_requires`` key value with optional dependencies.

    This allows us to work with and without setuptools.  It also means we can
    check for packages that have not been installed with setuptools to avoid
    installing them again.

    Parameters
    ----------
    pkg_name : str
       name of package as imported into python
    version : {None, str}, optional
       minimum version of the package that we require. If None, we don't
       check the version.  Default is None
    optional : bool or str, optional
       If ``bool(optional)`` is False, raise error for absent package or wrong
       version; otherwise warn.  If ``setuptools_args`` is not None, and
       ``bool(optional)`` is not False, then `optional` should be a string
       giving the feature name for the ``extras_require`` argument to setup.
    checker : callable, optional
       callable with which to return comparable thing from version
       string.  Default is ``distutils.version.LooseVersion``
    version_getter : {None, callable}:
       Callable that takes `pkg_name` as argument, and returns the
       package version string - as in::

          ``version = version_getter(pkg_name)``

       If None, equivalent to::

          mod = __import__(pkg_name); version = mod.__version__``
    messages : None or dict, optional
       dictionary giving output messages
    setuptools_args : None or dict
       If None, raise errors / warnings for missing non-optional / optional
       dependencies.  If dict fill key values ``install_requires`` and
       ``extras_require`` for non-optional and optional dependencies.
    """
    setuptools_mode = not setuptools_args is None
    optional_tf = bool(optional)
    if version_getter is None:

        def version_getter(pkg_name):
            mod = __import__(pkg_name)
            return mod.__version__

    if messages is None:
        messages = {}
    msgs = {
        'missing': 'Cannot import package "%s" - is it installed?',
        'missing opt': 'Missing optional package "%s"',
        'opt suffix': '; you may get run-time errors',
        'version too old': 'You have version %s of package "%s" but we need version >= %s',
    }
    msgs.update(messages)
    status, have_version = _package_status(pkg_name, version, version_getter, checker)
    if status == 'satisfied':
        return
    if not setuptools_mode:
        if status == 'missing':
            if not optional_tf:
                raise RuntimeError(msgs['missing'] % pkg_name)
            log.warn(msgs['missing opt'] % pkg_name + msgs['opt suffix'])
            return
        elif status == 'no-version':
            raise RuntimeError(f'Cannot find version for {pkg_name}')
        assert status == 'low-version'
        if not optional_tf:
            raise RuntimeError(msgs['version too old'] % (have_version, pkg_name, version))
        log.warn(msgs['version too old'] % (have_version, pkg_name, version) + msgs['opt suffix'])
        return
    # setuptools mode
    if optional_tf and not isinstance(optional, str):
        raise RuntimeError('Not-False optional arg should be string')
    dependency = pkg_name
    if version:
        dependency += '>=' + version
    if optional_tf:
        if not 'extras_require' in setuptools_args:
            setuptools_args['extras_require'] = {}
        _add_append_key(setuptools_args['extras_require'], optional, dependency)
    else:
        _add_append_key(setuptools_args, 'install_requires', dependency)


def _package_status(pkg_name, version, version_getter, checker):
    try:
        __import__(pkg_name)
    except ImportError:
        return 'missing', None
    if not version:
        return 'satisfied', None
    try:
        have_version = version_getter(pkg_name)
    except AttributeError:
        return 'no-version', None
    if checker(have_version) < checker(version):
        return 'low-version', have_version
    return 'satisfied', have_version


BAT_TEMPLATE = r"""@echo off
REM wrapper to use shebang first line of {FNAME}
set mypath=%~dp0
set pyscript="%mypath%{FNAME}"
set /p line1=<%pyscript%
if "%line1:~0,2%" == "#!" (goto :goodstart)
echo First line of %pyscript% does not start with "#!"
exit /b 1
:goodstart
set py_exe=%line1:~2%
call "%py_exe%" %pyscript% %*
"""


class install_scripts_bat(install_scripts):
    """Make scripts executable on Windows

    Scripts are bare file names without extension on Unix, fitting (for example)
    Debian rules. They identify as python scripts with the usual ``#!`` first
    line. Unix recognizes and uses this first "shebang" line, but Windows does
    not. So, on Windows only we add a ``.bat`` wrapper of name
    ``bare_script_name.bat`` to call ``bare_script_name`` using the python
    interpreter from the #! first line of the script.

    Notes
    -----
    See discussion at
    https://matthew-brett.github.io/pydagogue/installing_scripts.html and
    example at git://github.com/matthew-brett/myscripter.git for more
    background.
    """

    def run(self):
        install_scripts.run(self)
        if not os.name == 'nt':
            return
        for filepath in self.get_outputs():
            # If we can find an executable name in the #! top line of the script
            # file, make .bat wrapper for script.
            with open(filepath, 'rt') as fobj:
                first_line = fobj.readline()
            if not (first_line.startswith('#!') and 'python' in first_line.lower()):
                log.info('No #!python executable found, skipping .bat wrapper')
                continue
            pth, fname = psplit(filepath)
            froot, ext = splitext(fname)
            bat_file = pjoin(pth, froot + '.bat')
            bat_contents = BAT_TEMPLATE.replace('{FNAME}', fname)
            log.info(f'Making {bat_file} wrapper for {filepath}')
            if self.dry_run:
                continue
            with open(bat_file, 'wt') as fobj:
                fobj.write(bat_contents)


class Bunch:
    def __init__(self, vars):
        for key, name in vars.items():
            if key.startswith('__'):
                continue
            self.__dict__[key] = name


def read_vars_from(ver_file):
    """Read variables from Python text file

    Parameters
    ----------
    ver_file : str
        Filename of file to read

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `ver_file` appear as
        attributes
    """
    # Use exec for compabibility with Python 3
    ns = {}
    with open(ver_file, 'rt') as fobj:
        exec(fobj.read(), ns)
    return Bunch(ns)

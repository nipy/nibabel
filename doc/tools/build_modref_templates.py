#!/usr/bin/env python
"""Script to auto-generate our API docs"""
import contextlib
import re
import sys
from distutils.version import LooseVersion
from pathlib import Path

from apigen import ApiDocWriter

VERSION_PATTERN: str = '^_version_(major|minor|micro|extra)'
SKIP_PATTERNS = [
    r'\.fixes$',
    r'\.fixes.*$',
    r'\.externals$',
    r'\.externals.*$',
    r'.*test.*$',
    r'\.info.*$',
    r'\.pkg_info.*$',
    r'\.py3k.*$',
    r'\._version.*$',
]


def abort(error):
    print(f'*WARNING* API documentation not generated: {error}')
    exit(1)


if __name__ == '__main__':
    package = sys.argv[1]
    outdir = sys.argv[2]
    try:
        other_defines = sys.argv[3]
    except IndexError:
        other_defines = True
    else:
        other_defines = other_defines in {'True', 'true', '1'}

    # Check that the package is available. If not, the API documentation is not
    # (re)generated and existing API documentation sources will be used.

    try:
        __import__(package)
    except ImportError as e:
        abort(f'Failed to import {package}!')

    module = sys.modules[package]

    # Check that the source version is equal to the installed
    # version. If the versions mismatch the API documentation sources
    # are not (re)generated. This avoids automatic generation of documentation
    # for older or newer versions if such versions are installed on the system.

    installed_version = LooseVersion(module.__version__)

    package_root = Path(__file__).parent.parent.parent
    version_file = package_root / package / '_version.py'

    source_version = None
    if version_file.exists():
        # Versioneer
        from runpy import run_path

        with contextlib.suppress(FileNotFoundError, KeyError):
            source_version = run_path(version_file)['version']
        if source_version == '0+unknown':
            source_version = None
    if source_version is None:
        # Legacy fall-back
        info_file = package_root / package / 'info.py'
        info_lines = open(info_file).readlines()
        source_version = '.'.join(
            [v.split('=')[1].strip(" '\n.") for v in info_lines if re.match(VERSION_PATTERN, v)]
        )
    print('***', source_version)

    if source_version != installed_version:
        abort('Installed version does not match source version')

    docwriter = ApiDocWriter(package, rst_extension='.rst', other_defines=other_defines)
    docwriter.package_skip_patterns += SKIP_PATTERNS
    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'index', relative_to=outdir)

    n_written = len(docwriter.written_modules)
    print(f'{n_written} files written')

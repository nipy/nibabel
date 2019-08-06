import sys
import re
from distutils.version import StrictVersion
from . import _version

__version__ = _version.get_versions()['version']


def _parse_version(version_str):
    """ Parse version string `version_str` in our format
    """
    match = re.match(r'([0-9.]*\d)(.*)', version_str)
    if match is None:
        raise ValueError('Invalid version ' + version_str)
    return match.groups()


def _cmp(a, b):
    """ Implementation of ``cmp`` for Python 3
    """
    return (a > b) - (a < b)


def cmp_pkg_version(version_str, pkg_version_str=__version__):
    """ Compare `version_str` to current package version

    To be valid, a version must have a numerical major version followed by a
    dot, followed by a numerical minor version.  It may optionally be followed
    by a dot and a numerical micro version, and / or by an "extra" string.
    *Any* extra string labels the version as pre-release, so `1.2.0somestring`
    compares as prior to (pre-release for) `1.2.0`, where `somestring` can be
    any string.

    Parameters
    ----------
    version_str : str
        Version string to compare to current package version
    pkg_version_str : str, optional
        Version of our package.  Optional, set fom ``__version__`` by default.

    Returns
    -------
    version_cmp : int
        1 if `version_str` is a later version than `pkg_version_str`, 0 if
        same, -1 if earlier.

    Examples
    --------
    >>> cmp_pkg_version('1.2.1', '1.2.0')
    1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0')
    -1
    """
    version, extra = _parse_version(version_str)
    pkg_version, pkg_extra = _parse_version(pkg_version_str)
    if version != pkg_version:
        return _cmp(StrictVersion(version), StrictVersion(pkg_version))
    return (0 if extra == pkg_extra
            else 1 if extra == ''
            else -1 if pkg_extra == ''
            else _cmp(extra, pkg_extra))


def pkg_commit_hash(pkg_path=None):
    ''' Get short form of commit hash

    Versioneer placed a ``_version.py`` file in the package directory. This file
    gets updated on installation or ``git archive``.
    We inspect the contents of ``_version`` to detect whether we are in a
    repository, an archive of the repository, or an installed package.

    If detection fails, we return a not-found placeholder tuple

    Parameters
    ----------
    pkg_path : str
       directory containing package

    Returns
    -------
    hash_from : str
       Where we got the hash from - description
    hash_str : str
       short form of hash
    '''
    versions = _version.get_versions()
    hash_str = versions['full-revisionid'][:7]
    if hasattr(_version, 'version_json'):
        hash_from = 'installation'
    elif not _version.get_keywords()['full'].startswith('$Format:'):
        hash_from = 'archive substitution'
    elif versions['version'] == '0+unknown':
        hash_from, hash_str = '(none found)', '<not found>'
    else:
        hash_from = 'repository'
    return hash_from, hash_str


def get_pkg_info(pkg_path):
    ''' Return dict describing the context of this package

    Parameters
    ----------
    pkg_path : str
       path containing __init__.py for package

    Returns
    -------
    context : dict
       with named parameters of interest
    '''
    src, hsh = pkg_commit_hash()
    import numpy
    return dict(
        pkg_path=pkg_path,
        commit_source=src,
        commit_hash=hsh,
        sys_version=sys.version,
        sys_executable=sys.executable,
        sys_platform=sys.platform,
        np_version=numpy.__version__)

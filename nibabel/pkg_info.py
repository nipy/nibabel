import sys
from . import _version


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

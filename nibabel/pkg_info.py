import sys
from packaging.version import Version
from . import _version

__version__ = _version.get_versions()['version']


def _cmp(a, b):
    """ Implementation of ``cmp`` for Python 3
    """
    return (a > b) - (a < b)


def cmp_pkg_version(version_str, pkg_version_str=__version__):
    """ Compare ``version_str`` to current package version

    This comparator follows `PEP-440`_ conventions for determining version
    ordering.

    To be valid, a version must have a numerical major version. It may be
    optionally followed by a dot and a numerical minor version, which may,
    in turn, optionally be followed by a dot and a numerical micro version,
    and / or by an "extra" string.
    The extra string may further contain a "+". Any value to the left of a "+"
    labels the version as pre-release, while values to the right indicate a
    post-release relative to the values to the left. That is,
    ``1.2.0+1`` is post-release for ``1.2.0``, while ``1.2.0rc1+1`` is
    post-release for ``1.2.0rc1`` and pre-release for ``1.2.0``.

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
    >>> cmp_pkg_version('1.2.0dev', '1.2.0rc1')
    -1
    >>> cmp_pkg_version('1.2.0rc1', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0rc1+1', '1.2.0rc1')
    1
    >>> cmp_pkg_version('1.2.0rc1+1', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0.post1', '1.2.0')
    1

    .. _`PEP-440`: https://www.python.org/dev/peps/pep-0440/
    """
    return _cmp(Version(version_str), Version(pkg_version_str))


def pkg_commit_hash(pkg_path=None):
    """ Get short form of commit hash

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
    """
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
    """ Return dict describing the context of this package

    Parameters
    ----------
    pkg_path : str
       path containing __init__.py for package

    Returns
    -------
    context : dict
       with named parameters of interest
    """
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

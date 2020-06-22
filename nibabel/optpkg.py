""" Routines to support optional packages """
from distutils.version import LooseVersion
from .tripwire import TripWire


def _check_pkg_version(pkg, min_version):
    # Default version checking function
    if isinstance(min_version, str):
        min_version = LooseVersion(min_version)
    try:
        return min_version <= pkg.__version__
    except AttributeError:
        return False


def optional_package(name, trip_msg=None, min_version=None):
    """ Return package-like thing and module setup for package `name`

    Parameters
    ----------
    name : str
        package name
    trip_msg : None or str
        message to give when someone tries to use the return package, but we
        could not import it at an acceptable version, and have returned a
        TripWire object instead. Default message if None.
    min_version : None or str or LooseVersion or callable
        If None, do not specify a minimum version.  If str, convert to a
        `distutils.version.LooseVersion`.  If str or LooseVersion` compare to
        version of package `name` with ``min_version <= pkg.__version__``.   If
        callable, accepts imported ``pkg`` as argument, and returns value of
        callable is True for acceptable package versions, False otherwise.

    Returns
    -------
    pkg_like : module or ``TripWire`` instance
        If we can import the package, return it.  Otherwise return an object
        raising an error when accessed
    have_pkg : bool
        True if import for package was successful, false otherwise
    module_setup : function
        callable usually set as ``setup_module`` in calling namespace, to allow
        skipping tests.

    Examples
    --------
    Typical use would be something like this at the top of a module using an
    optional package:

    >>> from nibabel.optpkg import optional_package
    >>> pkg, have_pkg, setup_module = optional_package('not_a_package')

    Of course in this case the package doesn't exist, and so, in the module:

    >>> have_pkg
    False

    and

    >>> pkg.some_function() #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TripWireError: We need package not_a_package for these functions,
        but ``import not_a_package`` raised an ImportError

    If the module does exist - we get the module

    >>> pkg, _, _ = optional_package('os')
    >>> hasattr(pkg, 'path')
    True

    Or a submodule if that's what we asked for

    >>> subpkg, _, _ = optional_package('os.path')
    >>> hasattr(subpkg, 'dirname')
    True
    """
    if callable(min_version):
        check_version = min_version
    elif min_version is None:
        check_version = lambda pkg: True
    else:
        check_version = lambda pkg: _check_pkg_version(pkg, min_version)
    # fromlist=[''] results in submodule being returned, rather than the top
    # level module.  See help(__import__)
    fromlist = [''] if '.' in name else []
    exc = None
    try:
        pkg = __import__(name, fromlist=fromlist)
    except Exception as exc_:
        # Could fail due to some ImportError or for some other reason
        # e.g. h5py might have been checking file system to support UTF-8
        # etc.  We should not blow if they blow
        exc = exc_  # So it is accessible outside of the code block
    else:  # import worked
        # top level module
        if check_version(pkg):
            return pkg, True, lambda: None
        # Failed version check
        if trip_msg is None:
            if callable(min_version):
                trip_msg = f'Package {min_version} fails version check'
            else:
                trip_msg = f'These functions need {name} version >= {min_version}'
    if trip_msg is None:
        trip_msg = (f'We need package {name} for these functions, '
                    f'but ``import {name}`` raised {exc}')
    pkg = TripWire(trip_msg)

    def setup_module():
        import unittest
        raise unittest.SkipTest(f'No {name} for these tests')

    return pkg, False, setup_module

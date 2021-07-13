""" Module to help with deprecating objects and classes
"""

import warnings

from .deprecator import Deprecator
from .pkg_info import cmp_pkg_version


class ModuleProxy(object):
    """ Proxy for module that may not yet have been imported

    Parameters
    ----------
    module_name : str
        Full module name e.g. ``nibabel.minc``

    Examples
    --------

    ::
        arr = np.arange(24).reshape((2, 3, 4))
        nifti1 = ModuleProxy('nibabel.nifti1')
        nifti1_image = nifti1.Nifti1Image(arr, np.eye(4))

    So, the ``nifti1`` object is a proxy that will import the required module
    when you do attribute access and return the attributes of the imported
    module.
    """

    def __init__(self, module_name):
        self._module_name = module_name

    def __getattr__(self, key):
        mod = __import__(self._module_name, fromlist=[''])
        return getattr(mod, key)

    def __repr__(self):
        return f"<module proxy for {self._module_name}>"


class FutureWarningMixin(object):
    """ Insert FutureWarning for object creation

    Examples
    --------
    >>> class C(object): pass
    >>> class D(FutureWarningMixin, C):
    ...     warn_message = "Please, don't use this class"

    Record the warning

    >>> with warnings.catch_warnings(record=True) as warns:
    ...     d = D()
    ...     warns[0].message.args[0]
    "Please, don't use this class"
    """
    warn_message = 'This class will be removed in future versions'

    def __init__(self, *args, **kwargs):
        warnings.warn(self.warn_message,
                      FutureWarning,
                      stacklevel=2)
        super(FutureWarningMixin, self).__init__(*args, **kwargs)


class VisibleDeprecationWarning(UserWarning):
    """ Deprecation warning that will be shown by default

    Python >= 2.7 does not show standard DeprecationWarnings by default:

    http://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x

    Use this class for cases where we do want to show deprecations by default.
    """
    pass


deprecate_with_version = Deprecator(cmp_pkg_version)

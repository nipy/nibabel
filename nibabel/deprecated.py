""" Module to help with deprecating classes and modules
"""

import warnings

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
        minc = ModuleProxy('nibabel.minc')
        minc_image = minc.Minc1Image(arr, np.eye(4))

    So, the ``minc`` object is a proxy that will import the required module when
    you do attribute access and return the attributes of the imported module.
    """
    def __init__(self, module_name):
        self._module_name = module_name

    def __hasattr__(self, key):
        mod = __import__(self._module_name, fromlist=[''])
        return hasattr(mod, key)

    def __getattr__(self, key):
        mod = __import__(self._module_name, fromlist=[''])
        return getattr(mod, key)

    def __repr__(self):
        return "<module proxy for {0}>".format(self._module_name)


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
    ...     warns[0].message
    FutureWarning("Please, don't use this class",)
    """
    warn_message = 'This class will be removed in future versions'
    def __init__(self, *args, **kwargs):
        warnings.warn(self.warn_message,
                      FutureWarning,
                      stacklevel=2)
        super(FutureWarningMixin, self).__init__(*args, **kwargs)

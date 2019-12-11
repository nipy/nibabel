""" Deprecated MINC1 module """

import warnings

warnings.warn("We will remove this module from nibabel 3.0; "
              "Please use the 'minc1' module instead",
              DeprecationWarning,
              stacklevel=2)

from .minc1 import *  # noqa

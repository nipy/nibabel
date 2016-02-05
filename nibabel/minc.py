""" Deprecated MINC1 module """

import warnings

warnings.warn("We will remove this module from nibabel soon; "
              "Please use the 'minc1' module instead",
              FutureWarning,
              stacklevel=2)

from .minc1 import *  # noqa

import warnings

warnings.warn("We no longer carry a copy of the 'py3k' module in nibabel; "
              "Please import from the 'numpy.compat.py3k' module directly. "
              "Full removal scheduled for nibabel 4.0.",
              FutureWarning,
              stacklevel=2)

from numpy.compat.py3k import *  # noqa

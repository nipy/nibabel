""" Shim allowing some grace time for removal of six.py copy """
# Remove around version 4.0
from __future__ import absolute_import

import warnings

warnings.warn("We no longer carry a copy of the 'six' package in nibabel; "
              "Please import the 'six' package directly",
              FutureWarning,
              stacklevel=2)

from six import *  # noqa

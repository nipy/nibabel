# init for externals package
try:
    from collections import OrderedDict
except ImportError:  # < Python 2.7
    from .ordereddict import OrderedDict

""" Decorator for labeling keyword arguments as keyword only
"""

from functools import wraps
import warnings

warnings.warn("We will remove the 'keywordonly' module from nibabel 5.0. "
              "Please use the built-in Python `*` argument to ensure "
              "keyword-only parameters (see PEP 3102).",
              DeprecationWarning,
              stacklevel=2)


def kw_only_func(n):
    """ Return function decorator enforcing maximum of `n` positional arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > n:
                raise TypeError(
                    f"{func.__name__} takes at most {n} positional argument{'s' if n > 1 else ''}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def kw_only_meth(n):
    """ Return method decorator enforcing maximum of `n` positional arguments

    The method has at least one positional argument ``self`` or ``cls``; allow
    for that.
    """
    return kw_only_func(n + 1)

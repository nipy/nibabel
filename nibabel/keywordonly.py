""" Decorator for labeling keyword arguments as keyword only
"""

from functools import wraps


def kw_only_func(n):
    """ Return function decorator enforcing maximum of `n` positional arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > n:
                raise TypeError(
                    '{0} takes at most {1} positional argument{2}'.format(
                        func.__name__, n, 's' if n > 1 else ''))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def kw_only_meth(n):
    """ Return method decorator enforcing maximum of `n` positional arguments

    The method has at least one positional argument ``self`` or ``cls``; allow
    for that.
    """
    return kw_only_func(n + 1)

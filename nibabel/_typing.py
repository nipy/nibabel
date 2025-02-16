"""Helpers for typing compatibility across Python versions"""

import sys

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 13):
    from typing_extensions import TypeVar
else:
    from typing import TypeVar


__all__ = [
    'ParamSpec',
    'Self',
    'TypeVar',
]

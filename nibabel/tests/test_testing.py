""" Test testing utilties
"""

import sys

from warnings import warn, simplefilter

from ..testing import catch_warn_reset

from nose.tools import assert_equal


def test_catch_warn_reset():
    # Initial state of module, no warnings
    my_mod = sys.modules[__name__]
    assert_equal(getattr(my_mod, '__warningregistry__', None), None)
    with catch_warn_reset(modules=[my_mod]):
        simplefilter('ignore')
        warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})
    with catch_warn_reset():
        simplefilter('ignore')
        warn('Some warning')
    assert_equal(len(my_mod.__warningregistry__), 1)
    with catch_warn_reset(modules=[my_mod]):
        simplefilter('ignore')
        warn('Another warning')
    assert_equal(len(my_mod.__warningregistry__), 1)
    with catch_warn_reset():
        simplefilter('ignore')
        warn('Another warning')
    assert_equal(len(my_mod.__warningregistry__), 2)

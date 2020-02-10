""" Testing tripwire module
"""

from ..tripwire import TripWire, is_tripwire, TripWireError

import pytest

def test_is_tripwire():
    assert not is_tripwire(object())
    assert is_tripwire(TripWire('some message'))


def test_tripwire():
    # Test tripwire object
    silly_module_name = TripWire('We do not have silly_module_name')
    with pytest.raises(TripWireError):
        getattr(silly_module_name, 'do_silly_thing')
    # Check AttributeError can be checked too
    try:
        silly_module_name.__wrapped__
    except TripWireError as err:
        assert isinstance(err, AttributeError)
    else:
        raise RuntimeError("No error raised, but expected")

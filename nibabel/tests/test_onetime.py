import pytest

from nibabel.onetime import auto_attr, setattr_on_read
from nibabel.testing import expires


@expires('5.0.0')
def test_setattr_on_read():
    with pytest.deprecated_call():

        class MagicProp:
            @setattr_on_read
            def a(self):
                return object()

    x = MagicProp()
    assert 'a' not in x.__dict__
    obj = x.a
    assert 'a' in x.__dict__
    # Each call to object() produces a unique object. Verify we get the same one every time.
    assert x.a is obj


def test_auto_attr():
    class MagicProp:
        @auto_attr
        def a(self):
            return object()

    x = MagicProp()
    assert 'a' not in x.__dict__
    obj = x.a
    assert 'a' in x.__dict__
    # Each call to object() produces a unique object. Verify we get the same one every time.
    assert x.a is obj

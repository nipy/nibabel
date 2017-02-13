""" Test we are deprecating externals.six import
"""

import warnings
import types

from nose.tools import assert_true, assert_equal

from nibabel.deprecated import ModuleProxy


def test_old_namespace():
    with warnings.catch_warnings(record=True) as warns:
        # Top level import.
        # This import does not trigger an import of the six.py module, because
        # it's the proxy object.
        from nibabel.externals import six
        assert_equal(warns, [])
        # If there was a previous import it will be module, otherwise it will be
        # a proxy.
        previous_import = isinstance(six, types.ModuleType)
        if not previous_import:
            assert_true(isinstance(six, ModuleProxy))
        shim_BytesIO = six.BytesIO  # just to check it works
        # There may or may not be a warning raised on accessing the proxy,
        # depending on whether the externals.six.py module is already imported
        # in this test run.
        if not previous_import:
            assert_equal(warns.pop(0).category, FutureWarning)
        from six import BytesIO
        assert_equal(warns, [])
        # The import from old module is the same as that from new
        assert_true(shim_BytesIO is BytesIO)

# Test ordered dict loading

from os.path import dirname, relpath, realpath
import sys
import inspect

from .. import OrderedDict

from nose.tools import assert_equal, assert_true

MY_PATH = dirname(realpath(__file__))

def test_right_version():
    # If Python < 2.7, use our own copy, else use system copy
    class_path = inspect.getfile(OrderedDict)
    rel_dir = dirname(relpath(realpath(class_path), MY_PATH))
    if sys.version_info[:2] < (2, 7):
        assert_equal(rel_dir, '..')
    else:
        import collections
        assert_true(collections.OrderedDict is OrderedDict)

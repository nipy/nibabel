""" Testing environment settings
"""

import os
from os import environ as env
from os.path import join as pjoin, abspath


from .. import environment as nibe

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_equal

from nose import with_setup

GIVEN_ENV = {}
DATA_KEY = 'NIPY_DATA_PATH'
USER_KEY = 'NIPY_USER_DIR'


def setup_environment():
    """Setup test environment for some functions that are tested
    in this module. In particular this functions stores attributes
    and other things that we need to stub in some test functions.
    This needs to be done on a function level and not module level because
    each testfunction needs a pristine environment.
    """
    global GIVEN_ENV
    GIVEN_ENV['env'] = env.copy()


def teardown_environment():
    """Restore things that were remembered by the setup_environment function
    """
    orig_env = GIVEN_ENV['env']
    # Pull keys out into list to avoid altering dictionary during iteration,
    # causing python 3 error
    for key in list(env.keys()):
        if key not in orig_env:
            del env[key]
    env.update(orig_env)


# decorator to use setup, teardown environment
with_environment = with_setup(setup_environment, teardown_environment)


def test_nipy_home():
    # Test logic for nipy home directory
    assert_equal(nibe.get_home_dir(), os.path.expanduser('~'))


@with_environment
def test_user_dir():
    if USER_KEY in env:
        del env[USER_KEY]
    home_dir = nibe.get_home_dir()
    if os.name == "posix":
        exp = pjoin(home_dir, '.nipy')
    else:
        exp = pjoin(home_dir, '_nipy')
    assert_equal(exp, nibe.get_nipy_user_dir())
    env[USER_KEY] = '/a/path'
    assert_equal(abspath('/a/path'), nibe.get_nipy_user_dir())


def test_sys_dir():
    sys_dir = nibe.get_nipy_system_dir()
    if os.name == 'nt':
        assert_equal(sys_dir, r'C:\etc\nipy')
    elif os.name == 'posix':
        assert_equal(sys_dir, r'/etc/nipy')
    else:
        assert_equal(sys_dir, None)

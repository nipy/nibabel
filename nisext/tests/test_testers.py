""" Tests for testers
"""

import os
from os.path import dirname, pathsep

from ..testers import run_mod_cmd

from nose.tools import assert_true, assert_false, assert_equal, assert_raises


def test_run_mod_cmd():
    mod = 'os'
    mod_dir = dirname(os.__file__)
    assert_equal(run_mod_cmd(mod, mod_dir, "print('Hello')", None, False),
                 ("Hello", ""))
    sout, serr = run_mod_cmd(mod, mod_dir, "print('Hello again')")
    assert_equal(serr, '')
    mod_file, out_str = [s.strip() for s in sout.split('\n')]
    assert_true(mod_file.startswith(mod_dir))
    assert_equal(out_str, 'Hello again')
    sout, serr = run_mod_cmd(mod,
                             mod_dir,
                             "print(os.environ['PATH'])",
                             None,
                             False)
    assert_equal(serr, '')
    sout2, serr = run_mod_cmd(mod,
                              mod_dir,
                              "print(os.environ['PATH'])",
                              'pth2',
                              False)
    assert_equal(serr, '')
    assert_equal(sout2, 'pth2' + pathsep + sout)

""" Tests for testers
"""
from __future__ import division, print_function

import os
from os.path import dirname, pathsep

from ..testers import back_tick, run_mod_cmd, PYTHON

from nose.tools import assert_true, assert_equal, assert_raises


def test_back_tick():
    cmd = '{0} -c "print(\'Hello\')"'.format(PYTHON)
    assert_equal(back_tick(cmd), "Hello")
    assert_equal(back_tick(cmd, ret_err=True), ("Hello", ""))
    assert_equal(back_tick(cmd, True, False), (b"Hello", b""))
    cmd = '{0} -c "raise ValueError()"'.format(PYTHON)
    assert_raises(RuntimeError, back_tick, cmd)


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
    assert_equal(sout2, '"pth2"' + pathsep + sout)

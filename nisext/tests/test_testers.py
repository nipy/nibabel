""" Tests for testers
"""

import os
from os.path import dirname, pathsep

from ..testers import back_tick, run_mod_cmd, PYTHON

import pytest


def test_back_tick():
    cmd = f'{PYTHON} -c "print(\'Hello\')"'
    assert back_tick(cmd) == "Hello"
    assert back_tick(cmd, ret_err=True) == ("Hello", "")
    assert back_tick(cmd, True, False) == (b"Hello", b"")
    cmd = f'{PYTHON} -c "raise ValueError()"'
    with pytest.raises(RuntimeError):
        back_tick(cmd)


def test_run_mod_cmd():
    mod = 'os'
    mod_dir = dirname(os.__file__)
    assert run_mod_cmd(mod, mod_dir, "print('Hello')", None, False) == ("Hello", "")
    sout, serr = run_mod_cmd(mod, mod_dir, "print('Hello again')")
    assert serr == ''
    mod_file, out_str = [s.strip() for s in sout.split('\n')]
    assert mod_file.startswith(mod_dir)
    assert out_str == 'Hello again'
    sout, serr = run_mod_cmd(mod, mod_dir, "print(os.environ['PATH'])", None, False)
    assert serr == ''
    sout2, serr = run_mod_cmd(mod, mod_dir, "print(os.environ['PATH'])", 'pth2', False)
    assert serr == ''
    assert sout2 == '"pth2"' + pathsep + sout

#!/usr/bin/env python
''' Test nibabel version options

Fairly unix specific because of use of 'tar'
'''

import os
import sys
import shutil
import tempfile
from subprocess import call
from functools import partial

my_call = partial(call, shell=True)

py_lib_sdir = 'pylib'

def test_print(mod_name, pkg_path):
    os.chdir(os.path.expanduser('~'))
    my_call('python -c "import sys; sys.path.insert(0,\'%s\'); '
            'import %s; print %s.get_info()"' % (pkg_path,
                                                 mod_name,
                                                 mod_name))


def run_tests(mod_name, repo_path, install_path):
    site_pkgs_path = os.path.join(install_path, py_lib_sdir)
    py_lib_locs = ' --install-purelib=%s --install-platlib=%s' % (
        site_pkgs_path, site_pkgs_path)
    # first test archive
    os.chdir(repo_path)
    my_call('git archive --format tar -o %s/test.tar master' % install_path)
    os.chdir(install_path)
    my_call('tar xf test.tar')
    my_call('python setup.py --quiet install --prefix=%s %s' % (install_path,
                                                                py_lib_locs))
    test_print(mod_name, site_pkgs_path)

    # remove installation
    shutil.rmtree(site_pkgs_path)
    # now test install into a directory from the repository
    os.chdir(repo_path)
    my_call('python setup.py --quiet install --prefix=%s %s' % (install_path,
                                                                py_lib_locs))
    test_print(mod_name, site_pkgs_path)

    # test from development tree
    test_print(mod_name, repo_path)
    return


if __name__ == '__main__':
    try:
        mod_name = sys.argv[1]
    except IndexError:
        raise OSError("Need module name")
    try:
        repo_path = sys.argv[2]
    except IndexError:
        repo_path = os.path.abspath(os.getcwd())
    if not os.path.isfile(os.path.join(repo_path, 'setup.py')):
        raise OSError('Need setup.py in repo path %s' % repo_path)
    if not os.path.isdir(os.path.join(repo_path, mod_name)):
        raise OSError('Need package % in repo path %s' % (mod_name, repo_path))
    os.chdir(repo_path)
    install_path = tempfile.mkdtemp()
    try:
        run_tests(mod_name, repo_path, install_path)
    finally:
        shutil.rmtree(install_path)

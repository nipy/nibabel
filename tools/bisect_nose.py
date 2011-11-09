#!/usr/bin/env python
""" Utility for git-bisecting nose failures
"""
DESCRIP = 'Check nose output for given text, set sys exit for git bisect'
EPILOG = \
"""
Imagine you've just detected a nose test failure.  The failure is in a
particular test or test module - here 'test_analyze.py'.  The failure *is* in
git branch ``main-master`` but it *is not* in tag ``v1.6.1``. Then you can
bisect with something like::

    git co main-master
    git bisect start HEAD v1.6.1 --
    git bisect run /path/to/bisect_nose.py nibabel/tests/test_analyze.py:TestAnalyzeImage.test_str

You might well want to test that::

    nosetests nibabel/tests/test_analyze.py:TestAnalyzeImage.test_str

works as you expect first.

Let's say instead that you prefer to recognize the failure with an output
string.  Maybe this is because there are lots of errors but you are only
interested in one of them, or because you are looking for a Segmentation fault
instead of a test failure. Then::

    git co main-master
    git bisect start HEAD v1.6.1 --
    git bisect run /path/to/bisect_nose.py --error-txt='HeaderDataError: data dtype "int64" not recognized'  nibabel/tests/test_analyze.py

where ``error-txt`` is in fact a regular expression.

You will need 'argparse' installed somewhere. This is in the system libraries
for python 2.7 and python 3.2 onwards.

We run the tests in a temporary directory, so the code you are testing must be
on the python path.
"""
import os
import sys
import shutil
import tempfile
import re
from functools import partial
from subprocess import check_call, Popen, PIPE, CalledProcessError

from argparse import ArgumentParser, RawDescriptionHelpFormatter

caller = partial(check_call, shell=True)
popener = partial(Popen, stdout=PIPE, stderr=PIPE, shell=True)

# git bisect exit codes
UNTESTABLE = 125
GOOD = 0
BAD = 1

def call_or_untestable(cmd):
    try:
        caller(cmd)
    except CalledProcessError:
        sys.exit(UNTESTABLE)


def main():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('test_path',  type=str,
                        help='Path to test')
    parser.add_argument('--error-txt', type=str,
                        help='regular expression for error of interest')
    parser.add_argument('--clean', action='store_true',
                        help='Clean git tree before running tests')
    parser.add_argument('--build', action='store_true',
                        help='Build git tree before running tests')
    # parse the command line
    args = parser.parse_args()
    path = os.path.abspath(args.test_path)
    if args.clean:
        print "Cleaning"
        call_or_untestable('git clean -fxd')
    if args.build:
        print "Building"
        call_or_untestable('python setup.py build_ext -i')
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
        print "Testing"
        proc = popener('nosetests ' + path)
        stdout, stderr = proc.communicate()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)
    if args.error_txt:
        regex = re.compile(args.error_txt)
        if regex.search(stderr):
            sys.exit(BAD)
        sys.exit(GOOD)
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()

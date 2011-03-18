""" distutils utilities for porting to python 3 within 2-compatible tree """

from __future__ import with_statement
import re

try:
    from distutils.command.build_py import build_py_2to3
except ImportError:
    # 2.x - no parsing of code
    from distutils.command.build_py import build_py
else: # Python 3
    # Command to also apply 2to3 to doctests
    from distutils import log
    class build_py(build_py_2to3):
        def run_2to3(self, files):
            # Add doctest parsing; this stuff copied from distutils.utils in
            # python 3.2 source
            if not files:
                return
            fixer_names, options, explicit = (self.fixer_names,
                                              self.options,
                                              self.explicit)
            # Make this class local, to delay import of 2to3
            from lib2to3.refactor import RefactoringTool, get_fixers_from_package
            class DistutilsRefactoringTool(RefactoringTool):
                def log_error(self, msg, *args, **kw):
                    log.error(msg, *args)

                def log_message(self, msg, *args):
                    log.info(msg, *args)

                def log_debug(self, msg, *args):
                    log.debug(msg, *args)

            if fixer_names is None:
                fixer_names = get_fixers_from_package('lib2to3.fixes')
            r = DistutilsRefactoringTool(fixer_names, options=options)
            r.refactor(files, write=True)
            # Then doctests
            r.refactor(files, write=True, doctests_only=True)
            # Then custom doctests markup
            doctest_markup_files(files)


def doctest_markup_files(fnames):
    """ Process simple doctest comment markup on sequence of filenames

    Parameters
    ----------
    fnames : seq
        sequence of filenames

    Returns
    -------
    None
    """
    for fname in fnames:
        with open(fname, 'rt') as fobj:
            res = list(fobj)
        out = doctest_markup(res)
        with open(fname, 'wt') as fobj:
            fobj.write(''.join(out))


REGGIES = (
     (re.compile('from\s+io\s+import\s+StringIO\s+as\s+BytesIO\s*?(?=$)'),
     'from io import BytesIO'),
)

MARK_COMMENT = re.compile('#23:\s*(.*?)\s*$')
FIRST_NW = re.compile('(\s*)(\S.*)', re.DOTALL)

def doctest_markup(lines):
    """ Process doctest comment markup on sequence of strings

    The markup is very crude.  The algorithm looks for lines starting with
    ``>>>``.  All other lines are passed through unchanged.

    Next it looks for known simple search replaces and does those.

    * ``from StringIO import StringIO as BytesIO`` replaced by ``from io import
      BytesIO``.
    * ``from StringIO import StringIO`` replaced by ``from io import StringIO``.

    Next it looks to see if the line ends with a comment starting with ``#23:``.

    * ``bytes`` : prepend 'b' to next output line; for when you want to show
      output of bytes type in python 3

    Parameters
    ----------
    lines : sequence of str

    Returns
    -------
    newlines : sequence of str
    """
    newlines = []
    lines = iter(lines)
    while(True):
        try:
            line = next(lines)
        except StopIteration:
            break
        if not line.lstrip().startswith('>>>'):
            newlines.append(line)
            continue
        # Check simple regexps (no markup)
        for reg, substr in REGGIES:
            if reg.search(line):
                line = reg.sub(substr, line)
                break
        # Check for specific markup
        mark_match = MARK_COMMENT.search(line)
        if mark_match is None:
            newlines.append(line)
            continue
        markup = mark_match.groups()[0]
        # Actually we don't know what to do with markup yet
        if markup == 'bytes':
            newlines.append(line)
            try:
                line = next(lines)
            except StopIteration:
                break
            match = FIRST_NW.match(line)
            if match:
                line = '%sb%s' % match.groups()
            newlines.append(line)
        else:
            newlines.append(line)
    return newlines


"""distutils utilities for porting to python 3 within 2-compatible tree"""


try:
    from distutils.command.build_py import build_py_2to3
except ImportError:
    # 2.x - no parsing of code
    from distutils.command.build_py import build_py
else:   # Python 3
    # Command to also apply 2to3 to doctests
    from distutils import log

    class build_py(build_py_2to3):
        def run_2to3(self, files):
            # Add doctest parsing; this stuff copied from distutils.utils in
            # python 3.2 source
            if not files:
                return
            fixer_names, options, explicit = (self.fixer_names, self.options, self.explicit)
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

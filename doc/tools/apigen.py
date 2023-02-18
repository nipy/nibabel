"""
Attempt to generate templates for module reference with Sphinx

To include extension modules, first identify them as valid in the
``_uri2path`` method, then handle them in the ``_parse_module_with_import``
script.

Notes
-----
This parsing is based on import and introspection of modules.
Previously functions and classes were found by parsing the text of .py files.

Extension modules should be discovered and included as well.

This is a modified version of a script originally shipped with the PyMVPA
project, then adapted for use first in NIPY and then in skimage. PyMVPA
is an MIT-licensed project.
"""
from __future__ import annotations

import contextlib
import os
import re
from inspect import getmodule
from io import TextIOWrapper
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Literal, Sequence

# suppress print statements (warnings for empty files)
DEBUG = True

MatchType = Literal['package', 'module']


class ApiDocWriter:
    """Class for automatic detection and parsing of API docs
    to Sphinx-parsable reST format"""

    # only separating first two levels
    rst_section_levels = ['*', '=', '-', '~', '^']

    AUTO_GENERATED_MESSAGE: str = '.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n'
    REFERENCE_TITLE: str = 'API Reference'

    def __init__(
        self,
        package_name: str,
        rst_extension: str = '.rst',
        package_skip_patterns: Sequence[str] | None = None,
        module_skip_patterns: Sequence[str] | None = None,
        other_defines: bool = True,
    ) -> None:
        r"""Initialize package for parsing

        Parameters
        ----------
        package_name : str
            Name of the top-level package.  *package_name* must be the
            name of an importable package
        rst_extension : str
            Extension for reST files, default '.rst'
        package_skip_patterns : Sequence[str] | None
            Sequence of strings giving URIs of packages to be excluded
            Operates on the package path, starting at (including) the
            first dot in the package path, after *package_name* - so,
            if *package_name* is ``sphinx``, then ``sphinx.util`` will
            result in ``.util`` being passed for searching by these
            regexps.  If is None, gives default. Default is:
            ['\.tests$']
        module_skip_patterns : Sequence[str] | None
            Sequence of strings giving URIs of modules to be excluded
            Operates on the module name including preceding URI path,
            back to the first dot after *package_name*.  For example
            ``sphinx.util.console`` results in the string to search of
            ``.util.console``
            If is None, gives default. Default is:
            ['\.setup$', '\._']
        other_defines : bool
            Whether to include classes and functions that are imported in a
            particular module but not defined there.
        """
        if package_skip_patterns is None:
            package_skip_patterns = ['\\.tests$']
        if module_skip_patterns is None:
            module_skip_patterns = ['\\.setup$', '\\._']
        self.package_name = package_name
        self.rst_extension = rst_extension
        self.package_skip_patterns = package_skip_patterns
        self.module_skip_patterns = module_skip_patterns
        self.other_defines = other_defines

    @property
    def package_name(self) -> str:
        return self._package_name

    @package_name.setter
    def package_name(self, package_name: str) -> None:
        """Set package_name

        >>> docwriter = ApiDocWriter('sphinx')
        >>> import sphinx
        >>> docwriter.root_path == sphinx.__path__[0]
        True
        >>> docwriter.package_name = 'docutils'
        >>> import docutils
        >>> docwriter.root_path == docutils.__path__[0]
        True
        """
        # It's also possible to imagine caching the module parsing here
        self._package_name = package_name
        root_module = self._import(package_name)
        self.root_path = Path(root_module.__path__[-1])
        self.written_modules = None

    def _import(self, name: str) -> ModuleType:
        """Import namespace package"""
        mod = __import__(name)
        components = name.split('.')
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def _get_object_name(self, line: str) -> str:
        """Get second token in line
        >>> docwriter = ApiDocWriter('sphinx')
        >>> docwriter._get_object_name("  def func():  ")
        'func'
        >>> docwriter._get_object_name("  class Klass:  ")
        'Klass'
        >>> docwriter._get_object_name("  class Klass:  ")
        'Klass'
        """
        name = line.split()[1].split('(')[0].strip()
        # in case we have classes which are not derived from object
        # ie. old style classes
        return name.rstrip(':')

    def _uri2path(self, uri: str) -> str | None:
        """Convert uri to absolute filepath

        Parameters
        ----------
        uri : str
            URI of python module to return path for

        Returns
        -------
        path : str | None
            Returns None if there is no valid path for this URI
            Otherwise returns absolute file system path for URI

        Examples
        --------
        >>> docwriter = ApiDocWriter('sphinx')
        >>> import sphinx
        >>> modpath = sphinx.__path__[0]
        >>> res = docwriter._uri2path('sphinx.builder')
        >>> res == os.path.join(modpath, 'builder.py')
        True
        >>> res = docwriter._uri2path('sphinx')
        >>> res == os.path.join(modpath, '__init__.py')
        True
        >>> docwriter._uri2path('sphinx.does_not_exist')

        """
        if uri == self.package_name:
            return self.root_path / '__init__.py'
        path = uri.replace(f'{self.package_name}.', '')
        path = path.replace('.', os.path.sep)
        path = self.root_path / path
        # XXX maybe check for extensions as well?
        if path.with_suffix('.py').exists():
            return path.with_suffix('.py')
        elif (path / '__init__.py').exists():
            return path / '__init__.py'

    def _path2uri(self, dirpath: str) -> str:
        """Convert directory path to uri"""
        package_dir = self.package_name.replace('.', os.path.sep)
        relpath = str(dirpath).replace(str(self.root_path), package_dir)
        if relpath.startswith(os.path.sep):
            relpath = relpath[1:]
        return relpath.replace(os.path.sep, '.')

    def _parse_module(self, uri: str) -> tuple[list[str], list[str]]:
        """Parse module defined in *uri*"""
        filename = self._uri2path(uri)
        if filename is None:
            print(filename, 'erk')
            # nothing that we could handle here.
            return ([], [])

        with open(filename, 'rt') as f:
            functions, classes = self._parse_lines(f)
        return functions, classes

    def _parse_module_with_import(self, uri: str) -> tuple[list[str], list[str]]:
        """Look for functions and classes in an importable module.

        Parameters
        ----------
        uri : str
            The name of the module to be parsed. This module needs to be
            importable.

        Returns
        -------
        functions : list of str
            A list of (public) function names in the module.
        classes : list of str
            A list of (public) class names in the module.
        """
        mod = __import__(uri, fromlist=[uri.split('.')[-1]])
        # find all public objects in the module.
        obj_strs = [obj for obj in dir(mod) if not obj.startswith('_')]
        functions = []
        classes = []
        for obj_str in obj_strs:
            # find the actual object from its string representation
            if obj_str not in mod.__dict__:
                continue
            obj = mod.__dict__[obj_str]
            # Check if function / class defined in module
            if not self.other_defines and getmodule(obj) != mod:
                continue
            # figure out if obj is a function or class
            if isinstance(obj, (FunctionType, BuiltinFunctionType)):
                functions.append(obj_str)
            else:
                with contextlib.suppress(TypeError):
                    issubclass(obj, object)
                    classes.append(obj_str)
        return functions, classes

    def _parse_lines(self, linesource: str) -> tuple[list[str], list[str]]:
        """Parse lines of text for functions and classes"""
        functions = []
        classes = []
        for line in linesource:
            if line.startswith('def ') and line.count('('):
                # exclude private stuff
                name = self._get_object_name(line)
                if not name.startswith('_'):
                    functions.append(name)
            elif line.startswith('class '):
                # exclude private stuff
                name = self._get_object_name(line)
                if not name.startswith('_'):
                    classes.append(name)
        functions.sort()
        classes.sort()
        return functions, classes

    def generate_api_doc(self, uri: str) -> tuple[str, str]:
        """Make autodoc documentation template string for a module

        Parameters
        ----------
        uri : string
            python location of module - e.g 'sphinx.builder'

        Returns
        -------
        head : string
            Module name, table of contents.
        body : string
            Function and class docstrings.
        """
        # get the names of all classes and functions
        functions, classes = self._parse_module_with_import(uri)
        if not len(functions) and not len(classes) and DEBUG:
            print('WARNING: Empty -', uri)  # dbg

        # Make a shorter version of the uri that omits the package name for
        # titles
        uri_short = re.sub(rf'^{self.package_name}\.', '', uri)

        head = self.AUTO_GENERATED_MESSAGE
        body = ''

        # Set the chapter title to read 'module' for all modules except for the
        # main packages
        if '.' in uri_short:
            title = f'Module: :mod:`{uri_short}`'
            head += title + '\n' + self.rst_section_levels[2] * len(title)
        else:
            title = f':mod:`{uri_short}`'
            head += title + '\n' + self.rst_section_levels[1] * len(title)

        head += '\n.. automodule:: ' + uri + '\n'
        head += '\n.. currentmodule:: ' + uri + '\n'
        body += '\n.. currentmodule:: ' + uri + '\n\n'
        for c in classes:
            body += '\n:class:`' + c + '`\n' + self.rst_section_levels[3] * (len(c) + 9) + '\n\n'
            body += '\n.. autoclass:: ' + c + '\n'
            # must NOT exclude from index to keep cross-refs working
            body += (
                '  :members:\n'
                '  :undoc-members:\n'
                '  :show-inheritance:\n'
                '\n'
                '  .. automethod:: __init__\n\n'
            )
        head += '.. autosummary::\n\n'
        for f in classes + functions:
            head += f'   {f}\n'
        head += '\n'

        for f in functions:
            # must NOT exclude from index to keep cross-refs working
            body += f + '\n'
            body += self.rst_section_levels[3] * len(f) + '\n'
            body += f'\n.. autofunction:: {f}\n\n'

        return head, body

    def _survives_exclude(self, matchstr: str, match_type: MatchType) -> bool:
        """Returns True if *matchstr* does not match patterns

        ``self.package_name`` removed from front of string if present

        Examples
        --------
        >>> dw = ApiDocWriter('sphinx')
        >>> dw._survives_exclude('sphinx.okpkg', 'package')
        True
        >>> dw.package_skip_patterns.append('^\\.badpkg$')
        >>> dw._survives_exclude('sphinx.badpkg', 'package')
        False
        >>> dw._survives_exclude('sphinx.badpkg', 'module')
        True
        >>> dw._survives_exclude('sphinx.badmod', 'module')
        True
        >>> dw.module_skip_patterns.append('^\\.badmod$')
        >>> dw._survives_exclude('sphinx.badmod', 'module')
        False
        """
        # Select skip patterns
        if match_type == 'module':
            patterns = self.module_skip_patterns
        elif match_type == 'package':
            patterns = self.package_skip_patterns
        else:
            raise ValueError(f'Cannot interpret match type "{match_type}"')

        # Match to URI without package name
        L = len(self.package_name)
        if matchstr[:L] == self.package_name:
            matchstr = matchstr[L:]
        for pattern in patterns:
            try:
                pattern.search
            except AttributeError:
                pattern = re.compile(pattern)
            if pattern.search(matchstr):
                return False

        return True

    def discover_modules(self) -> list[str]:
        r"""Return module sequence discovered from ``self.package_name``

        Parameters
        ----------
        None

        Returns
        -------
        mods : sequence
            Sequence of module names within ``self.package_name``

        Examples
        --------
        >>> dw = ApiDocWriter('sphinx')
        >>> mods = dw.discover_modules()
        >>> 'sphinx.util' in mods
        True
        >>> dw.package_skip_patterns.append('\.util$')
        >>> 'sphinx.util' in dw.discover_modules()
        False
        >>>
        """
        modules = [self.package_name]
        # raw directory parsing
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            # Check directory names for packages
            root_uri = self._path2uri(str(self.root_path / dirpath))

            # Normally, we'd only iterate over dirnames, but since
            # dipy does not import a whole bunch of modules we'll
            # include those here as well (the *.py filenames).
            filenames = [
                f[:-3] for f in filenames if f.endswith('.py') and not f.startswith('__init__')
            ]
            for filename in filenames:
                package_uri = '/'.join((dirpath, filename))

            for subpkg_name in dirnames + filenames:
                package_uri = '.'.join((root_uri, subpkg_name))
                package_path = self._uri2path(package_uri)
                if package_path and self._survives_exclude(package_uri, 'package'):
                    modules.append(package_uri)

        return sorted(modules)

    def write_modules_api(self, modules: list[str], destination: Path) -> None:
        # upper-level modules
        ulms = [
            '.'.join(m.split('.')[:2]) if m.count('.') >= 1 else m.split('.')[0] for m in modules
        ]

        from collections import OrderedDict

        module_by_ulm = OrderedDict()

        for v, k in zip(modules, ulms):
            if k in module_by_ulm:
                module_by_ulm[k].append(v)
            else:
                module_by_ulm[k] = [v]

        written_modules = []

        for ulm, mods in module_by_ulm.items():
            print(f'Generating docs for {ulm}:')
            document_head = []
            document_body = []

            for m in mods:
                print(f'  -> {m}')
                head, body = self.generate_api_doc(m)

                document_head.append(head)
                document_body.append(body)

            out_module = ulm + self.rst_extension
            outfile = destination / out_module
            with open(outfile, 'wt') as fileobj:
                fileobj.writelines(document_head + document_body)
            written_modules.append(out_module)

        self.written_modules = written_modules

    def write_api_docs(self, destination: os.PathLike | str) -> None:
        """Generate API reST files

        Parameters
        ----------
        destination : os.PathLike | str
            Directory name in which to store files
            We create automatic filenames for each module

        Returns
        -------
        None

        Notes
        -----
        Sets self.written_modules to list of written modules
        """
        destination = Path(destination)
        if not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)
        # compose list of modules
        modules = self.discover_modules()
        self.write_modules_api(modules, destination)

    def write_index(
        self, destination: os.PathLike | str, froot: str = 'gen', relative_to: str | None = None
    ) -> None:
        """Make a reST API index file from written files

        Parameters
        ----------
        path : os.PathLike | str
            Filename to write index to
        destination : string
            Directory to which to write generated index file
        froot : str
            root (filename without extension) of filename to write to
            Defaults to 'gen'.  We add ``self.rst_extension``.
        relative_to : str | None
            path to which written filenames are relative.  This
            component of the written file path will be removed from
            destination, in the generated index.  Default is None, meaning,
            leave path as it is.
        """
        if self.written_modules is None:
            raise ValueError('No modules written')
        # Get full filename path
        destination = Path(destination)
        path = (destination / froot).with_suffix(self.rst_extension)
        # Path written into index is relative to rootpath
        if relative_to is not None:
            relative_path = Path(
                (str(destination) + os.path.sep).replace(relative_to + os.path.sep, '')
            )
        else:
            relative_path = destination
        with open(path, 'wt') as index_file:
            self._write_index_contents(index_file, relative_path)

    def _write_index_contents(self, index_file: TextIOWrapper, relative_path: Path):
        w = index_file.write
        w(self.AUTO_GENERATED_MESSAGE)
        w(self.REFERENCE_TITLE + '\n')
        w('=' * len(self.REFERENCE_TITLE) + '\n\n')
        w('.. toctree::\n\n')
        for f in self.written_modules:
            module_rst = relative_path / f
            w(f'   {module_rst}\n')

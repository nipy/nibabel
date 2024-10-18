#!/usr/bin/env python
import re
import sys
from collections import defaultdict
from functools import cache
from operator import call
from pathlib import Path

from sphinx.ext.intersphinx import fetch_inventory

CHANGELOG = Path(__file__).parent.parent / 'Changelog'

# Match release lines like "5.2.0 (Monday 11 December 2023)"
RELEASE_REGEX = re.compile(r"""((?:\d+)\.(?:\d+)\.(?:\d+)) \(\w+ \d{1,2} \w+ \d{4}\)$""")


class MockConfig:
    intersphinx_timeout: int | None = None
    tls_verify = False
    tls_cacerts: str | dict[str, str] | None = None
    user_agent: str = ''


@call
class MockApp:
    srcdir = ''
    config = MockConfig()


fetch_inv = cache(fetch_inventory)


def get_intersphinx(obj):
    module = obj.split('.', 1)[0]

    registry = defaultdict(lambda: 'https://docs.python.org/3')
    registry.update(
        numpy='https://numpy.org/doc/stable',
    )

    base_url = registry[module]

    inventory = fetch_inv(MockApp, '', f'{base_url}/objects.inv')
    # Check py: first, then whatever
    for objclass in sorted(inventory, key=lambda x: not x.startswith('py:')):
        if obj in inventory[objclass]:
            return f'{base_url}/{inventory[objclass][obj][2]}'
    raise ValueError("Couldn't lookup {obj}")


def main():
    version = sys.argv[1]
    output = sys.argv[2]
    if output == '-':
        output = sys.stdout
    else:
        output = open(output, 'w')

    release_notes = []
    in_release_notes = False

    with open(CHANGELOG) as f:
        for line in f:
            match = RELEASE_REGEX.match(line)
            if match:
                if in_release_notes:
                    break
                in_release_notes = match.group(1) == version
                next(f)  # Skip the underline
                continue

            if in_release_notes:
                release_notes.append(line)

    # Drop empty lines at start and end
    while release_notes and not release_notes[0].strip():
        release_notes.pop(0)
    while release_notes and not release_notes[-1].strip():
        release_notes.pop()

    # Join lines
    release_notes = ''.join(release_notes)

    # Remove line breaks when they are followed by a space
    release_notes = re.sub(r'\n +', ' ', release_notes)

    # Replace pr/<number> with #<number> for GitHub
    release_notes = re.sub(r'pr/(\d+)', r'#\1', release_notes)

    # Replace :mod:`package.X` with [package.X](...)
    release_notes = re.sub(
        r':mod:`nibabel\.(.*)`',
        r'[nibabel.\1](https://nipy.org/nibabel/reference/nibabel.\1.html)',
        release_notes,
    )
    # Replace :class/func/attr:`package.module.X` with [package.module.X](...)
    release_notes = re.sub(
        r':(?:class|func|attr):`(nibabel\.\w*)(\.[\w.]*)?\.(\w+)`',
        r'[\1\2.\3](https://nipy.org/nibabel/reference/\1.html#\1\2.\3)',
        release_notes,
    )
    release_notes = re.sub(
        r':(?:class|func|attr):`~(nibabel\.\w*)(\.[\w.]*)?\.(\w+)`',
        r'[\3](https://nipy.org/nibabel/reference/\1.html#\1\2.\3)',
        release_notes,
    )
    # Replace :meth:`package.module.class.X` with [package.module.class.X](...)
    release_notes = re.sub(
        r':meth:`(nibabel\.[\w.]*)\.(\w+)\.(\w+)`',
        r'[\1.\2.\3](https://nipy.org/nibabel/reference/\1.html#\1.\2.\3)',
        release_notes,
    )
    release_notes = re.sub(
        r':meth:`~(nibabel\.[\w.]*)\.(\w+)\.(\w+)`',
        r'[\3](https://nipy.org/nibabel/reference/\1.html#\1.\2.\3)',
        release_notes,
    )
    # Replace :<any>:`<ref>` with intersphinx lookup
    for ref in re.findall(r'(:[^:]*:`~?\w[\w.]+\w`)', release_notes):
        objclass, tilde, module, obj = re.match(r':([^:]*):`(~?)([\w.]+)\.(\w+)`', ref).groups()
        url = get_intersphinx(f'{module}.{obj}')
        mdlink = f'[{"" if tilde else module}{obj}]({url})'
        release_notes = release_notes.replace(ref, mdlink)
    # Replace RST links with Markdown links
    release_notes = re.sub(r'`([^<`]*) <([^>]*)>`_+', r'[\1](\2)', release_notes)

    def python_doc(match):
        module = match.group(1)
        name = match.group(2)
        return f'[{name}](https://docs.python.org/3/library/{module.lower()}.html#{module}.{name})'

    release_notes = re.sub(r':meth:`~([\w.]+)\.(\w+)`', python_doc, release_notes)

    with output:
        output.write('## Release notes\n\n')
        output.write(release_notes)


if __name__ == '__main__':
    main()

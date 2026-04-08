#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gitpython>=3.1.46",
#     "ruamel.yaml>=0.18",
# ]
# ///
from pathlib import Path
from subprocess import PIPE, run

import git
from ruamel.yaml import YAML

skip = {'nibotmi', 'dependabot[bot]', 'pre-commit-ci[bot]'}


def build_author_map(authors):
    """Build a lookup from display name to existing author entry.

    Entries with given-names/family-names are keyed as "Given Family".
    Entries with only alias are keyed by the alias string.
    """
    author_map = {}
    for author in authors:
        given = author.get('given-names')
        family = author.get('family-names')
        if given and family:
            author_map[f'{given} {family}'] = author
        elif 'alias' in author:
            author_map[author['alias']] = author
    return author_map


def strip_trailing_whitespace(s: str) -> str:
    return s.replace(' \n', '\n')


git_root = Path(git.Repo('.', search_parent_directories=True).working_dir)

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 100
yaml.indent(sequence=4, offset=2)

citation_file = git_root / 'CITATION.cff'
citation = yaml.load(citation_file)

author_map = build_author_map(citation.get('authors', []))
shortlog = run(['git', 'shortlog', '-ns', 'HEAD'], stdout=PIPE)
commit_counts = dict(
    line.split('\t', 1)[::-1] for line in shortlog.stdout.decode().split('\n') if line
)

citation['authors'] = [
    author_map.get(committer, {'alias': committer})
    # Sort by commit count descending, then alphabetically
    for committer, _ in sorted(commit_counts.items(), key=lambda x: (-int(x[1]), x[0]))
    if committer not in skip
]


yaml.dump(citation, citation_file, transform=strip_trailing_whitespace)

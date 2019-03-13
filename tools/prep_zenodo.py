#!/usr/bin/env python3
import git
import json
from subprocess import run, PIPE
from pathlib import Path

skip = {'nibotmi'}


def decommify(name):
    return ' '.join(name.split(', ')[::-1])


git_root = Path(git.Repo('.', search_parent_directories=True).working_dir)
zenodo_file = git_root / '.zenodo.json'

zenodo = json.loads(zenodo_file.read_text()) if zenodo_file.exists() else {}

orig_creators = zenodo.get('creators', [])
creator_map = {decommify(creator['name']): creator
               for creator in orig_creators}

shortlog = run(['git', 'shortlog', '-ns'], stdout=PIPE)
counts = [line.split('\t', 1)[::-1]
          for line in shortlog.stdout.decode().split('\n') if line]

commit_counts = {}
for committer, commits in counts:
    commit_counts[committer] = commit_counts.get(committer, 0) + int(commits)

# Stable sort:
# Number of commits in reverse order
# Ties broken by alphabetical order of first name
committers = [committer
              for committer, _ in sorted(commit_counts.items(),
                                         key=lambda x: (-x[1], x[0]))]

creators = [
    creator_map.get(committer, {'name': committer})
    for committer in committers
    if committer not in skip
    ]

zenodo['creators'] = creators
zenodo_file.write_text(json.dumps(zenodo, indent=2, sort_keys=True) + '\n')

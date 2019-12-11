""" Benchmarking utilities
"""

from .. import get_info


def print_git_title(title):
    """ Prints title string with git hash if possible, and underline
    """
    title = '{0} for git revision {1}'.format(
        title,
        get_info()['commit_hash'])
    print(title)
    print('-' * len(title))

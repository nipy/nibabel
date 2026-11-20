import pytest

from contextlib import chdir as _chdir


@pytest.fixture(autouse=True)
def chdir_tmpdir(request, tmp_path):
    if request.node.__class__.__name__ == 'DoctestItem':
        with _chdir(tmp_path):
            yield
    else:
        yield

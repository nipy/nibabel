#####################
Nibabel documentation
#####################

To build the documentation, change to the root directory (containing
``pyproject.toml``) and run::

    uv sync --extra doc
    source .venv/bin/activate
    make -C doc html

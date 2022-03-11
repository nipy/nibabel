#!/bin/bash

echo "Building archive"

source tools/ci/activate.sh

set -eu

# Required dependencies
echo "INSTALL_TYPE = $INSTALL_TYPE"

set -x

if [ "$INSTALL_TYPE" = "sdist" -o "$INSTALL_TYPE" = "wheel" ]; then
    python -m build
elif [ "$INSTALL_TYPE" = "archive" ]; then
    ARCHIVE="/tmp/package.tar.gz"
    git archive -o $ARCHIVE HEAD
fi

if [ "$INSTALL_TYPE" = "sdist" ]; then
    ARCHIVE=$( ls $PWD/dist/*.tar.gz )
elif [ "$INSTALL_TYPE" = "wheel" ]; then
    ARCHIVE=$( ls $PWD/dist/*.whl )
elif [ "$INSTALL_TYPE" = "pip" ]; then
    ARCHIVE="$PWD"
fi

export ARCHIVE

set +eux

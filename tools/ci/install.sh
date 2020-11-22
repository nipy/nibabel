#!/bin/bash

echo Installing nibabel

source tools/ci/activate.sh

set -eu

# Required variables
echo INSTALL_TYPE = $INSTALL_TYPE
echo CHECK_TYPE = $CHECK_TYPE
echo EXTRA_PIP_FLAGS = $EXTRA_PIP_FLAGS

set -x

if [ "$INSTALL_TYPE" == "setup" ]; then
    python setup.py install
else
    pip install $EXTRA_PIP_FLAGS $ARCHIVE
fi

# Basic import check
python -c 'import nibabel; print(nibabel.__version__)'

if [ "$CHECK_TYPE" == "skiptests" ]; then
    exit 0
fi

pip install $EXTRA_PIP_FLAGS "nibabel[$CHECK_TYPE]"

set +eux

echo Done installing nibabel

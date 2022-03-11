#!/bin/bash

echo Installing nibabel

source tools/ci/activate.sh
source tools/ci/env.sh

set -eu

# Required variables
echo INSTALL_TYPE = $INSTALL_TYPE
echo CHECK_TYPE = $CHECK_TYPE
echo EXTRA_PIP_FLAGS = $EXTRA_PIP_FLAGS

set -x

if [ -n "$EXTRA_PIP_FLAGS" ]; then
    EXTRA_PIP_FLAGS=${!EXTRA_PIP_FLAGS}
fi

(
  # Ensure installation does not depend on being in source tree
  mkdir ../unversioned_install_dir
  cd ../unversioned_install_dir
  pip install $EXTRA_PIP_FLAGS $ARCHIVE

  # Basic import check
  python -c 'import nibabel; print(nibabel.__version__)'
)

if [ "$CHECK_TYPE" == "skiptests" ]; then
    exit 0
fi

pip install $EXTRA_PIP_FLAGS "nibabel[$CHECK_TYPE]"

set +eux

echo Done installing nibabel

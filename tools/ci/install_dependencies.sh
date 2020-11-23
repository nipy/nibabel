#!/bin/bash

echo Installing dependencies

source tools/ci/activate.sh

set -eu

# Required variables
echo EXTRA_PIP_FLAGS = $EXTRA_PIP_FLAGS
echo DEPENDS = $DEPENDS

set -x

if [ -n "$EXTRA_PIP_FLAGS" ]; then
    EXTRA_PIP_FLAGS=${!EXTRA_PIP_FLAGS}
fi

if [ -n "$DEPENDS" ]; then
    pip install ${EXTRA_PIP_FLAGS} ${!DEPENDS}
    pip install ${EXTRA_PIP_FLAGS} $OPTIONAL_DEPENDS || true
fi

set +eux

echo Done installing dependencies

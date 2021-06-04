SETUP_REQUIRES="pip setuptools>=30.3.0 wheel"

# Minimum requirements
REQUIREMENTS="-r requirements.txt"
# Minimum versions of minimum requirements
MIN_REQUIREMENTS="-r min-requirements.txt"

DEFAULT_OPT_DEPENDS="scipy matplotlib pillow pydicom h5py indexed_gzip pyzstd"
# pydicom has skipped some important pre-releases, so enable a check against master
PYDICOM_MASTER="git+https://github.com/pydicom/pydicom.git@master"
# Minimum versions of optional requirements
MIN_OPT_DEPENDS="matplotlib==1.5.3 pydicom==0.9.9 pillow==2.6"

# Numpy and scipy upload nightly/weekly/intermittent wheels
NIGHTLY_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
STAGING_WHEELS="https://pypi.anaconda.org/multibuild-wheels-staging/simple"
PRE_PIP_FLAGS="--pre --extra-index-url $NIGHTLY_WHEELS --extra-index-url $STAGING_WHEELS"

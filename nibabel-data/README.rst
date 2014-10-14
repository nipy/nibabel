############
Nibabel data
############

This subdirectory contains data repositories for testing.

The data repositories should not be included in source or binary
distributions.

A some point we might remove this directory from the source distribution and
make the data packages available with a more formal data package format.

For the moment the tests can find this data path by:

* Using the contents of the ``NIBABEL_DATA_DIR`` environment variable;
* Looking for this ``nibabel-data`` directory in the directory above (closer
  to the root directory) the directory containing the ``nibabel`` package.

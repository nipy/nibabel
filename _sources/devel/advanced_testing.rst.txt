.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the NiBabel package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _advanced_testing:

****************
Advanced Testing
****************

Setup
-----

Before running advanced tests, please update all submodules of nibabel, by
running ``git submodule update --init``

Long-running tests
------------------

Long-running tests are not enabled by default, and can be resource-intensive. To run these tests:

* Set environment variable ``NIPY_EXTRA_TESTS=slow``;
* Run ``pytest nibabel``.

Note that some tests may require a machine with >4GB of RAM.

.. include:: ../links_names.txt

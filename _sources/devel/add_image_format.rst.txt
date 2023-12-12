########################################
How to add a new image format to nibabel
########################################

These are some work-in-progress notes in the hope that they will help adding a
new image format to NiBabel.

**********
Philosophy
**********

As usual, the general idea is to make your image as explicit and transparent
as possible.

From the Zen of Python (``import this``), these guys spring to mind:

* Explicit is better than implicit.
* Errors should never pass silently.
* In the face of ambiguity, refuse the temptation to guess.
* Now is better than never.
* If the implementation is hard to explain, it's a bad idea.

So far we have tried to make the nibabel version of the image as close as
possible to the way the user of the particular format is expecting to see it.

For example, the NIfTI format documents describe the image with the first
dimension of the image data array being the fastest varying in memory (and on
disk).  Numpy defaults to having the last dimension of the array being the
fastest varying in memory.  We chose to have the first dimension vary fastest
in memory to match the conventions in the NIfTI specification.

******************************
Helping us to review your code
******************************

You are likely to know the image format much much better than the rest of us
do, but to help you with the code, we will need to learn.  The following will
really help us get up to speed:

#. Links in the code or in the docs to the information on the file format.
   For example, you'll see the canonical links for the NIfTI 2 format at the
   top of the :mod:`.nifti2` file, in the module docstring;
#. Example files in the format; see :doc:`add_test_data`;
#. Good test coverage.  The tests help us see how you are expecting the code
   and the format to be used.  We recommend writing the tests first; the tests
   do an excellent job in helping us and you see how the API is going to work.

***************************
The format can be read-only
***************************

Read-only access to a format is better than no access to a format, and often
much better.  For example, we can read but not write PAR / REC and MINC files.
Having the code to read the files makes it easier to work with these files in
Python, and easier for someone else to add the ability to write the format
later.

*************
The image API
*************

An image should conform to the image API.  See the module docstring for
:mod:`.spatialimages` for a description of the API.

You should test whether your image does conform to the API by adding a test
class for your image in :mod:`nibabel.tests.test_image_api`.  For example, the
API test for the PAR / REC image format looks like::

    class TestPARRECAPI(LoadImageAPI):
        def loader(self, fname):
            return parrec.load(fname)

        example_images = PARREC_EXAMPLE_IMAGES

where your work is to define the ``EXAMPLE_IMAGES`` list |--| see the
:mod:`nibabel.tests.test_parrec` file for the PAR / REC example images
definition.

****************************
Where to start with the code
****************************

There is no API requirement that a new image format inherit from the general
:class:`.SpatialImage` class, but in fact all our image
formats do inherit from this class.  We strongly suggest you do the same, to
get many simple methods implemented for free.  You can always override the
ones you don't want.

There is also a generic header class you might consider building on to contain
your image metadata |--| :class:`.Header`.  See that
class for the header API.

The API does not require it, but if it is possible, it may be good to
implement the image data as loaded from disk as an array proxy.  See the
docstring of :mod:`.arrayproxy` for a description of the API, and see the
module code for an implementation of the API.  You may be able to use the
unmodified :class:`.ArrayProxy` class for your image type.

If you write a new array proxy class, add tests for the API of the class in
:mod:`nibabel.tests.test_proxy_api`.  See
:class:`.TestPARRECAPI` for an example.

A nibabel image is the association of:

#. The image array data (as implemented by an array proxy or a numpy array);
#. An affine relating the image array coordinates to an RAS+ world (see
   :doc:`../coordinate_systems`);
#. Image metadata in the form of a header.

Your new image constructor may well be the default from
:class:`.SpatialImage`, which looks like this::

    def __init__(self, dataobj, affine, header=None,
                 extra=None, file_map=None):

Your job when loading a file is to create:

#. ``dataobj`` - an array or array proxy;
#. ``affine`` - 4 by 4 array relating array coordinates to world coordinates;
#. ``header`` - a metadata container implementing at least ``get_data_dtype``,
   ``get_data_shape``.

You will likely implement this logic in the ``from_file_map`` method of the
image class.  See :class:`.PARRECImage` for an example.

***************************************
A recipe for writing a new image format
***************************************

#. Find one or more examples images;
#. Put them in ``nibabel/tests/data`` or a data submodule (see
   :doc:`add_test_data`);
#. Create a file ``nibabel/tests/test_my_format_name_here.py``;
#. Use some program that can read the format correctly to fill out the needed
   fields for an ``EXAMPLE_IMAGES`` list (see
   :mod:`nibabel.tests.test_parrec.py` for example);
#. Add a test class using your ``EXAMPLE_IMAGES`` to
   :mod:`nibabel.tests.test_image_api`, using the PARREC image test class as
   an example. Now you have some failing tests |--| good job!;
#. If you can, extract the metadata information from the test file, so it is
   small enough to fit as a small test file into ``nibabel/tests/data`` (don't
   forget the license);
#. Write small maybe private functions to extract the header metadata from
   your new test file, testing these functions in
   ``test_my_format_name_here.py``.  See :mod:`.parrec` for examples;
#. When that is working, try sub-classing :class:`.Header`, and working out how
   to make the ``__init__`` and ``from_fileboj`` methods for that class.  Test
   in ``test_my_format_name_here.py``;
#. When that is working, try sub-classing :class:`.SpatialImage` and working
   out how to load the file with the ``from_file_map`` class;
#. Now try seeing if you can get your ``test_image_api.py`` tests to pass;
#. Consider adding more test data files, maybe to a test data repository
   submodule (:doc:`add_test_data`).  Check you can read these files correctly
   (see :mod:`nibabel.tests.test_parrec_data` for an example).
#. Ask for advice as early and as often as you can, either with a
   work-in-progress pull request (the easiest way for us to review) or on
   the mailing list or via github issues.

.. include:: ../links_names.txt

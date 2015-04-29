################
Adding test data
################

#. We really, really like test images, but
#. We are rather conservative about the size of our code repository.

So, we have two different ways of adding test data.

#. Small, open licensed files can go in the ``nibabel/tests/data`` directory
   (see below);
#. Larger files or files with extra licensing terms can go in their own git
   repositories and be added as submodules to the ``nibabel-data`` directory.

***********
Small files
***********

Small files are around 50K or less when compressed.  By "compressed", we mean,
compressed with zlib, which is what git uses when storing the file in the
repository.  You can check the exact length directly with Python and a script
like::

    import sys
    import zlib

    for fname in sys.argv[1:]:
        with open(fname, 'rb') as fobj:
            contents = fobj.read()
        compressed = zlib.compress(contents)
        print(fname, len(compressed) / 1024.)

One way of making files smaller when compressed is to set uninteresting values
to zero or some other number so that the compression algorithm can be more
effective.

Please don't compress the file yourself before committing to a git repo unless
there's a really good reason; git will do this for you when adding to the
repository, and it's a shame to make git compress a compressed file.

************************
Files with open licenses
************************

We very much prefer files with completely open licenses such as the `PDDL
1.0`_ or the CC0_ license.

The files in the ``nibabel/tests/data`` will get distributed with the nibabel
source code, and this can easily get installed without the user having an
opportunity to review the full license.  We don't think this is compatible
with extra license terms like agreeing to cite the people who provided the
data or agreeing not to try and work out the identity of the person who has
been scanned, because it would be too easy to miss these requirements when
using nibabel.  It is fine to use files with these kind of licenses, but they
should go in their own repository to be used as a submodule, so they do not
need to be distributed with nibabel.

*****************************************
Adding the file to ``nibabel/tests/data``
*****************************************

If the file is less then about 50K compressed, and the license is open, then
you might want to commit the file under ``nibabel/tests/data``.

Put the license for any new files in the COPYING file at the top level of the
nibabel repo.  You'll see some examples in that file already.

*****************************************
Adding as a submodule to ``nibabel-data``
*****************************************

Make a new git repository with the data.

There are example repos at

* https://github.com/yarikoptic/nitest-balls1
* https://github.com/matthew-brett/nitest-minc2

Despite the fact that both the examples are on github, Bitbucket_ is good for
repos like this because they don't enforce repository size limits.

Don't forget to include a LICENSE and README file in the repo.

When all is done, and the repository is safely on the internet and accessible,
add the repo as a submodule to the ``nitests-data`` directory, with something
like this::

    git submodule add https://bitbucket.org/nipy/rosetta-samples.git nitests-data/rosetta-samples

You should now have a checked out copy of the ``rosetta-samples`` repository
in the ``nibabel-data/rosetta-samples`` directory.  Commit the submodule that
is now in your git staging area.

If you are writing tests using files from this repository, you should use the
``needs_nibabel_data`` decorator to skip the tests if the data has not been
checked out into the submodules.  See ``nibabel/tests/test_parrec_data.py``
for an example.  For our example repository above it might look something
like::

    from .nibabel_data import get_nibabel_data, needs_nibabel_data

    ROSETTA_DATA = pjoin(get_nibabel_data(), 'rosetta-samples')

    @needs_nibabel_data('rosetta-samples')
    def test_something():
        # Some test using the data

Using submodules for tests
==========================

Tests run via `nibabel on travis`_ start with an automatic checkout of all
submodules in the project, so all test data submodules get checked out by
default.

If you are running the tests locally, you may well want to do::

    git submodule update --init

from the root nibabel directory.  This will checkout all the test data
repositories.

How much data should go in a single submodule?
==============================================

The limiting factor is how long it takes travis-ci_ to checkout the data for
the tests.  Up to a hundred megabytes in one repository should be OK.  The joy
of submodules is we can always drop a submodule, split the repository into two
and add only one back, so you aren't committing us to anything awful if you
accidentally put some very large files into your own data repository.

If in doubt
===========

If you are not sure, try us with a pull request to `nibabel github`_, or on the
`nipy mailing list`_, we will try to help.

.. include:: ../links_names.txt

.. -*- mode: rst -*-
.. vim:ft=rst

.. _changelog:

#############################
NiBabel Development Changelog
#############################

NiBabel is the successor to the much-loved PyNifti package. Here we list the
releases for both packages.

The full VCS changelog is available here:

  http://github.com/nipy/nibabel/commits/master

****************
Nibabel releases
****************

Most work on NiBabel so far has been by Matthew Brett (MB), Chris Markiewicz
(CM), Michael Hanke (MH), Marc-Alexandre Côté (MC), Ben Cipollini (BC), Paul
McCarthy (PM), Chris Cheng (CC), Yaroslav Halchenko (YOH), Satra Ghosh (SG),
Eric Larson (EL), Demian Wassermann, Stephan Gerhard and Ross Markello (RM).

References like "pr/298" refer to github pull request numbers.

3.2.0 (Tuesday 20 October 2020)
==============================

New feature release in the 3.2.x series.

New features
------------
* ``nib-stats`` CLI tool to expose new ``nibabel.imagestats`` API. Initial
  implementation of volume calculations, a la ``fslstats -V``. (Julian Klug,
  reviewed by CM and GitHub user 0rC0)
* ``nib-roi`` CLI tool to crop images and/or flip axes (pr/947) (CM, reviewed
  by Chris Cheng and Mathias Goncalves)
* Parser for Siemens "ASCCONV" text format (pr/896) (Brendan Moloney and MB,
  reviewed by CM)

Enhancements
------------
* Drop confusing mention of ``img.to_filename()`` in getting started guide
  (pr/946) (Fernando Pérez-Garcia, reviewed by MB, CM)
* Implement ``to_bytes()``/``from_bytes()`` methods for ``Cifti2Image``
  (pr/938) (CM, reviewed by Mathias Goncalves)
* Clean up of DICOM documentation (pr/910) (Jonathan Daniel, reviewed by MB)

Bug fixes
---------
* Use canvas manager API to set title in ``OrthoSlicer3D`` (pr/958) (EL,
  reviewed by CM)
* Record units as seconds parrec2nii; previously set TR to seconds but
  retained msec units (pr/931) (CM, reviewed by MB)
* Reflect on-disk dimensions in NIfTI-2 view of CIFTI-2 images (pr/930)
  (Mathias Goncalves and CM)
* Fix outdated Python 2 and Sympy code in DICOM derivations (pr/911) (MB,
  reviewed by CM)
* Change string with invalid escape to raw string (pr/909) (EL, reviewed
  by MB)

Maintenance
-----------
* Fix typo in docs (pr/955) (Carl Gauthier, reviewed by CM)
* Purge nose from nisext tests (pr/934) (Markéta Calábková, reviewed by CM)
* Suppress expected warnings in tests (pr/949) (CM, reviewed by Dorota
  Jarecka)
* Various cleanups and modernizations (pr/916, pr/917, pr/918, pr/919)
  (Jonathan Daniel, reviewed by CM)
* SVG logo for improved appearance in with zooming (pr/914) (Jonathan Daniel,
  reviewed by CM)

API changes and deprecations
----------------------------
* Drop support for Numpy < 1.13 (pr/922) (CM)
* Warn on use of ``onetime.setattr_on_read``, which has been a deprecated
  alias of ``auto_attr`` (pr/948) (CM, reviewed by Ariel Rokem)


3.1.1 (Friday 26 June 2020)
===========================

Bug-fix release in the 3.1.x series.

These are small compatibility fixes that support ARM64 architecture and
``indexed_gzip>=1.3.0``.

Bug fixes
---------
* Detect ``IndexedGzipFile`` as compressed file type (pr/925) (PM, reviewed by
  CM)
* Correctly cast ``nan`` when testing ``array_to_file``, fixing ARM64 builds
  (pr/862) (CM, reviewed by MB)


3.1.0 (Monday 20 April 2020)
============================

New feature release in the 3.1.x series.

New features
------------
* Conformation function (``processing.conform``) and CLI tool
  (``nib-conform``) to apply shape, orientation and zooms (pr/853) (Jakub
  Kaczmarzyk, reviewed by CM, YOH)
* Affine rescaling function (``affines.rescale_affine``) to update
  dimensions and voxel sizes (pr/853) (CM, reviewed by Jakub Kaczmarzyk)

Bug fixes
---------
* Delay import of h5py until neded (pr/889) (YOH, reviewed by CM)

Maintenance
-----------
* Fix typo in documentation (pr/893) (Zvi Baratz, reviewed by CM)
* Tests converted from nose to pytest (pr/865 + many sub-PRs)
  (Dorota Jarecka, Krzyzstof Gorgolewski, Roberto Guidotti, Anibal Solon,
  and Or Duek)

API changes and deprecations
----------------------------
* ``kw_only_meth``/``kw_only_func`` decorators are deprecated (pr/848)
  (RM, reviewed by CM)


2.5.2 (Wednesday 8 April 2020)
==============================

Bug-fix release in the 2.5.x series. This is an extended-support series,
providing bug fixes for Python 2.7 and 3.4.

This and all future releases in the 2.5.x series will be incompatible with
Python 3.9. The last compatible series of numpy and scipy are 1.16.x and
1.2.x, respectively.

If you are able to upgrade to Python 3, it is recommended to upgrade to
NiBabel 3.

Bug fixes
---------
* Change strings with invalid escapes to raw strings (pr/827) (EL, reviewed
  by CM)
* Re-import externals/netcdf.py from scipy to resolve numpy deprecation
  (pr/821) (CM)

Maintenance
-----------
* Set maximum numpy to 1.16.x, maximum scipy to 1.2.x (pr/901) (CM)


3.0.2 (Monday 9 March 2020)
===========================

Bug fixes
---------
* Attempt to find versioneer version when building docs (pr/894) (CM)
* Delay import of h5py until neded (backport of pr/889) (YOH, reviewed by CM)

Maintenance
-----------
* Fix typo in documentation (backport of pr/893) (Zvi Baratz, reviewed by CM)
* Set minimum matplotlib to 1.5.3 to ensure wheels are available on all
  supported Python versions. (backport of pr/887) (CM)
* Remove ``pyproject.toml`` for now. (issue/859) (CM)


3.0.1 (Monday 27 January 2020)
==============================

Bug fixes
---------
* Test failed by using array method on tuple. (pr/860) (Ben Darwin, reviewed by
  CM)
* Validate ``ExpiredDeprecationError``\s, promoted by 3.0 release from
  ``DeprecationWarning``\s. (pr/857) (CM)

Maintenance
-----------
* Remove logic accommodating numpy without float16 types. (pr/866) (CM)
* Accommodate new numpy dtype strings. (pr/858) (CM)


3.0.0 (Wednesday 18 December 2019)
==================================

New features
------------
* ArrayProxy ``__array__()`` now accepts a ``dtype`` parameter, allowing
  ``numpy.array(dataobj, dtype=...)`` calls, as well as casting directly
  with a dtype (for example, ``numpy.float32(dataobj)``) to control the
  output type. Scale factors (slope, intercept) are applied, but may be
  cast to narrower types, to control memory usage. This is now the basis
  of ``img.get_fdata()``, which will scale data in single precision if
  the output type is ``float32``. (pr/844) (CM, reviewed by Alejandro
  de la Vega, Ross Markello)
* GiftiImage method ``agg_data()`` to return usable data arrays (pr/793)
  (Hao-Ting Wang, reviewed by CM)
* Accept ``os.PathLike`` objects in place of filenames (pr/610) (Cameron
  Riddell, reviewed by MB, CM)
* Function to calculate obliquity of affines (pr/815) (Oscar Esteban,
  reviewed by MB)

Enhancements
------------
* Improve testing of data scaling in ArrayProxy API (pr/847) (CM, reviewed
  by Alejandro de la Vega)
* Document ``SpatialImage.slicer`` interface (pr/846) (CM)
* ``get_fdata(dtype=np.float32)`` will attempt to avoid casting data to
  ``np.float64`` when scaling parameters would otherwise promote the data
  type unnecessarily. (pr/833) (CM, reviewed by Ross Markello)
* ``ArraySequence`` now supports a large set of Python operators to combine
  or update in-place. (pr/811) (MC, reviewed by Serge Koudoro, Philippe Poulin,
  CM, MB)
* Warn, rather than fail, on DICOMs with unreadable Siemens CSA tags (pr/818)
  (Henry Braun, reviewed by CM)
* Improve clarity of coordinate system tutorial (pr/823) (Egor Panfilov,
  reviewed by MB)

Bug fixes
---------
* Sliced ``Tractogram``\s no longer ``apply_affine`` to the original
  ``Tractogram``'s streamlines. (pr/811) (MC, reviewed by Serge Koudoro,
  Philippe Poulin, CM, MB)
* Change strings with invalid escapes to raw strings (pr/827) (EL, reviewed
  by CM)
* Re-import externals/netcdf.py from scipy to resolve numpy deprecation
  (pr/821) (CM)

Maintenance
-----------
* Remove replicated metadata for packaged data from MANIFEST.in (pr/845) (CM)
* Support Python >=3.5.1, including Python 3.8.0 (pr/787) (CM)
* Manage versioning with slightly customized Versioneer (pr/786) (CM)
* Reference Nipy Community Code and Nibabel Developer Guidelines in
  GitHub community documents (pr/778) (CM, reviewed by MB)

API changes and deprecations
----------------------------
* Fully remove deprecated ``checkwarns`` and ``minc`` modules. (pr/852) (CM)
* The ``keep_file_open`` argument to file load operations and ``ArrayProxy``\s
  no longer acccepts the value ``"auto"``, raising a ``ValueError``. (pr/852)
  (CM)
* Deprecate ``ArraySequence.data`` in favor of ``ArraySequence.get_data()``,
  which will return a copy. ``ArraySequence.data`` now returns a read-only
  view. (pr/811) (MC, reviewed by Serge Koudoro, Philippe Poulin, CM, MB)
* Deprecate ``DataobjImage.get_data()`` API, to be removed in nibabel 5.0
  (pr/794, pr/809) (CM, reviewed by MB)


2.5.1 (Monday 23 September 2019)
================================

Enhancements
------------
* Ignore endianness in ``nib-diff`` if values match (pr/799) (YOH, reviewed
  by CM)

Bug fixes
---------
* Correctly handle Philips DICOMs w/ derived volume (pr/795) (Mathias
  Goncalves, reviewed by CM)
* Raise CSA tag limit to 1000, parametrize for future relaxing (pr/798,
  backported to 2.5.x in pr/800) (Henry Braun, reviewed by CM, MB)
* Coerce data types to match NIfTI intent codes when writing GIFTI data
  arrays (pr/806) (CM, reported by Tom Holroyd)

Maintenance
-----------
* Require h5py 2.10 for Windows + Python < 3.6 to resolve unexpected dtypes
  in Minc2 data (pr/804) (CM, reviewed by YOH)

API changes and deprecations
----------------------------
* Deprecate ``nicom.dicomwrappers.Wrapper.get_affine()`` in favor of ``affine``
  property; final removal in nibabel 4.0 (pr/796) (YOH, reviewed by CM)

2.5.0 (Sunday 4 August 2019)
============================

The 2.5.x series is the last with support for either Python 2 or Python 3.4.
Extended support for this series 2.5 will last through December 2020.

Thanks for the test ECAT file and fix provided by Andrew Crabb.

Enhancements
------------
* Add SerializableImage class with to/from_bytes methods (pr/644) (CM,
  reviewed by MB)
* Check CIFTI-2 data shape matches shape described by header (pr/774)
  (Michiel Cottaar, reviewed by CM)

Bug fixes
---------
* Handle stricter numpy casting rules in tests (pr/768) (CM)
  reviewed by PM)
* TRK header fields flipped in files written on big-endian systems
  (pr/782) (CM, reviewed by YOH, MB)
* Load multiframe ECAT images with Python 3 (CM and Andrew Crabb)

Maintenance
-----------
* Fix CodeCov paths on Appveyor for more accurate coverage (pr/769) (CM)
* Move to setuptools and reduce use ``nisext`` functions (pr/764) (CM,
  reviewed by YOH)
* Better handle test setup/teardown (pr/785) (CM, reviewed by YOH)

API changes and deprecations
----------------------------
* Effect threatened warnings and set some deprecation timelines (pr/755) (CM)
  * Trackvis methods now default to v2 formats
  * ``nibabel.trackvis`` scheduled for removal in nibabel 4.0
  * ``nibabel.minc`` and ``nibabel.MincImage`` will be removed in nibabel 3.0

2.4.1 (Monday 27 May 2019)
==========================

Contributions from Egor Pafilov, Jath Palasubramaniam, Richard Nemec, and
Dave Allured.

Enhancements
------------
* Enable ``mmap``, ``keep_file_open`` options when loading any
  ``DataobjImage`` (pr/759) (CM, reviewed by PM)

Bug fixes
---------
* Ensure loaded GIFTI files expose writable data arrays (pr/750) (CM,
  reviewed by PM)
* Safer warning registry manipulation when checking for overflows (pr/753)
  (CM, reviewed by MB)
* Correctly write .annot files with duplicate lables (pr/763) (Richard Nemec
  with CM)

Maintenance
-----------
* Fix typo in coordinate systems doc (pr/751) (Egor Panfilov, reviewed by
  CM)
* Replace invalid MINC1 test file with fixed file (pr/754) (Dave Allured
  with CM)
* Update Sphinx config to support recent Sphinx/numpydoc (pr/749) (CM,
  reviewed by PM)
* Pacify ``FutureWarning`` and ``DeprecationWarning`` from h5py, numpy
  (pr/760) (CM)
* Accommodate Python 3.8 deprecation of collections.MutableMapping
  (pr/762) (Jath Palasubramaniam, reviewed by CM)

API changes and deprecations
----------------------------
* Deprecate ``keep_file_open == 'auto'`` (pr/761) (CM, reviewed by PM)

2.4.0 (Monday 1 April 2019)
============================

New features
------------
* Alternative ``Axis``-based interface for manipulating CIFTI-2 headers
  (pr/641) (Michiel Cottaar, reviewed by Demian Wassermann, CM, SG)

Enhancements
------------
* Accept TCK files produced by tools with other delimiter/EOF defaults
  (pr/720) (Soichi Hayashi, reviewed by CM, MB, MC)
* Allow BrainModels or Parcels to contain a single vertex in CIFTI
  (pr/739) (Michiel Cottaar, reviewed by CM)
* Support for ``NIFTI_XFORM_TEMPLATE_OTHER`` xform code (pr/743) (CM)

Bug fixes
---------
* Skip refcheck in ArraySequence construction/extension (pr/719) (Ariel
  Rokem, reviewed by CM, MC)
* Use safe resizing for ArraySequence extension (pr/724) (CM, reviewed
  by MC)
* Fix typo in error message (pr/726) (Jon Haitz Legarreta Gorroño,
  reviewed by CM)
* Support DICOM slice sorting in Python 3 (pr/728) (Samir Reddigari,
  reviewed by CM)
* Correctly reorient dim_info when reorienting NIfTI images
  (Konstantinos Raktivan, CM, reviewed by CM)

Maintenance
-----------
* Import updates to reduce upstream deprecation warnings (pr/711,
  pr/705, pr/738) (EL, YOH, reviewed by CM)
* Delay import of ``nibabel.testing``, ``nose`` and ``mock`` to speed up
  import (pr/699) (CM)
* Increase coverage testing, drop coveralls (pr/722, pr/732) (CM)
* Add Zenodo metadata, sorted by commits (pr/732) (CM + others)
* Update author listing and copyrights (pr/742) (MB, reviewed by CM)

2.3.3 (Wednesday 16 January 2019)
=================================

Maintenance
-----------
* Restore ``six`` dependency (pr/714) (CM, reviewed by Gael Varoquaux, MB)

2.3.2 (Wednesday 2 January 2019)
================================

Enhancements
------------
* Enable toggling crosshair with ``Ctrl-x`` in ``OrthoSlicer3D`` viewer (pr/701)
  (Miguel Estevan Moreno, reviewed by CM)

Bug fixes
---------
* Read .PAR files corresponding to ADC maps (pr/685) (Gregory R. Lee, reviewed
  by CM)
* Increase maximum number of items read from Siemens CSA format (Igor Solovey,
  reviewed by CM, MB)
* Check boolean dtypes with ``numpy.issubdtype(..., np.bool_)`` (pr/707)
  (Jon Haitz Legarreta Gorroño, reviewed by CM)

Maintenance
-----------
* Fix small typos in parrec2nii help text (pr/682) (Thomas Roos, reviewed by
  MB)
* Remove deprecated calls to ``numpy.asscalar`` (pr/686) (CM, reviewed by
  Gregory R. Lee)
* Update QA directives to accommodate Flake8 3.6 (pr/695) (CM)
* Update DOI links to use ``https://doi.org`` (pr/703) (Katrin Leinweber,
  reviewed by CM)
* Remove deprecated calls to ``numpy.fromstring`` (pr/700) (Ariel Rokem,
  reviewed by CM, MB)
* Drop ``distutils`` support, require ``bz2file`` for Python 2.7 (pr/700)
  (CM, reviewed by MB)
* Replace mutable ``bytes`` hack, disabled in numpy pre-release, with
  ``bytearray``/``readinto`` strategy (pr/700) (Ariel Rokem, CM, reviewed by
  CM, MB)

API changes and deprecations
----------------------------
* Add ``Opener.readinto`` method to read file contents into pre-allocated buffers
  (pr/700) (Ariel Rokem, reviewed by CM, MB)

2.3.1 (Tuesday 16 October 2018)
===============================

New features
------------
* ``nib-diff`` command line tool for comparing image files (pr/617, pr/672,
  pr/678) (CC, reviewed by YOH, Pradeep Raamana and CM)

Enhancements
------------
* Speed up reading of numeric arrays in CIFTI2 (pr/655) (Michiel Cottaar,
  reviewed by CM)
* Add ``ndim`` property to ``ArrayProxy`` and ``DataobjImage`` (pr/674) (CM,
  reviewed by MB)

Bug fixes
---------
* Deterministic deduction of slice ordering in degenerate cases (pr/647)
  (YOH, reviewed by CM)
* Allow 0ms TR in MGH files (pr/653) (EL, reviewed by CM)
* Allow for PPC64 little-endian long doubles (pr/658) (MB, reviewed by CM)
* Correct construction of FreeSurfer annotation labels (pr/666) (CM, reviewed
  by EL, Paul D. McCarthy)
* Fix logic for persisting filehandles with indexed-gzip (pr/679) (Paul D.
  McCarthy, reviewed by CM)

Maintenance
-----------
* Fix semantic error in coordinate systems documentation (pr/646) (Ariel
  Rokem, reviewed by CM, MB)
* Test on Python 3.7, minor associated fixes (pr/651) (CM, reviewed by Gregory
  R. Lee, MB)

2.3 (Tuesday 12 June 2018)
==========================

New features
------------
* TRK <=> TCK streamlines conversion CLI tools (pr/606) (MC, reviewed by CM)
* Image slicing for SpatialImages (pr/550) (CM)

Enhancements
------------
* Simplfiy MGHImage and add footer fields (pr/569) (CM, reviewed by MB)
* Force sform/qform codes to be ints, rather than numpy types (pr/575) (Paul
  McCarthy, reviewed by MB, CM)
* Auto-fill color table in FreeSurfer annotation file (pr/592) (PM,
  reviewed by CM, MB)
* Set default intent code for CIFTI2 images (pr/604) (Mathias Goncalves,
  reviewed by CM, SG, MB, Tim Coalson)
* Raise informative error on empty files (pr/611) (Pradeep Raamana, reviewed
  by CM, MB)
* Accept degenerate filenames such as ``.nii`` (pr/621) (Dimitri
  Papadopoulos-Orfanos, reviewed by Yaroslav Halchenko)
* Take advantage of ``IndexedGzipFile`` ``drop_handles`` flag to release
  filehandles by default (pr/614) (PM, reviewed by CM, MB)

Bug fixes
---------
* Preserve first point of `LazyTractogram` (pr/588) (MC, reviewed by Nil
  Goyette, CM, MB)
* Stop adding extraneous metadata padding (pr/593) (Jon Stutters, reviewed by
  CM, MB)
* Accept lower-case orientation codes in TRK files (pr/600) (Kesshi Jordan,
  MB, reviewed by MB, MC, CM)
* Annotation file reading (pr/592) (PM, reviewed by CM, MB)
* Fix buffer size calculation in ArraySequence (pr/597) (Serge Koudoro,
  reviewed by MC, MB, Eleftherios Garyfallidis, CM)
* Resolve ``UnboundLocalError`` in Python 3 (pr/607) (Jakub Kaczmarzyk,
  reviewed by MB, CM)
* Do not crash on non-``ImportError`` failures in optional imports (pr/618)
  (Yaroslav Halchenko, reviewed by CM)
* Return original array from ``get_fdata`` for array image, if no cast
  required (pr/638, MB, reviewed by CM)

Maintenance
-----------
* Use SSH address to use key-based auth (pr/587) (CM, reviewed by MB)
* Fix doctests for numpy 1.14 array printing (pr/591) (MB, reviewed by CM)
* Refactor for pydicom 1.0 API changes (pr/599) (MB, reviewed by CM)
* Increase test coverage, remove unreachable code (pr/602) (CM, reviewed by 
  Yaroslav Halchenko, MB)
* Move ``nib-ls`` and other programs to a new cmdline module (pr/601, pr/615)
  (Chris Cheng, reviewed by MB, Yaroslav Halchenko)
* Remove deprecated numpy indexing (EL, reviewed by CM)
* Update documentation to encourage ``get_fdata`` over ``get_data`` (pr/637,
  MB, reviewed by CM)

API changes and deprecations
----------------------------
* Support for ``keep_file_open = 'auto'`` as a parameter to ``Opener()`` will
  be deprecated in 2.4, for removal in 3.0. Accordingly, support for
  ``openers.KEEP_FILE_OPEN_DEFAULT = 'auto'`` will be dropped on the same
  schedule.
* Drop-in support for ``indexed_gzip < 0.7`` has been removed.


2.2.1 (Wednesday 22 November 2017)
==================================

Bug fixes
---------

* Set L/R labels in orthoview correctly (pr/564) (CM)
* Defer use of ufunc / memmap test - allows "freezing" (pr/572) (MB, reviewed
  by SG)
* Fix doctest failures with pre-release numpy (pr/582) (MB, reviewed by CM)

Maintenance
-----------

* Update documentation around NIfTI qform/sform codes (pr/576) (PM,
  reviewed by MB, CM) + (pr/580) (Bennet Fauber, reviewed by PM)
* Skip precision test on macOS, newer numpy (pr/583) (MB, reviewed by CM)
* Simplify AppVeyor script, removing conda (pr/584) (MB, reviewed by CM)

2.2 (Friday 13 October 2017)
============================

New features
------------

* CIFTI support (pr/249) (SG, Michiel Cottaar, BC, CM, Demian Wassermann, MB)
* Support for MRtrix TCK streamlines file format (pr/486) (MC, reviewed by
  MB, Arnaud Bore, J-Donald Tournier, Jean-Christophe Houde)
* Added ``get_fdata()`` as default method to retrieve scaled floating point
  data from ``DataobjImage``\s (pr/551) (MB, reviewed by CM, SG)

Enhancements
------------

* Support for alternative header field name variants in .PAR files
  (pr/507) (Gregory R. Lee)
* Various enhancements to streamlines API by MC: support for reading TRK
  version 1 (pr/512); concatenation of tractograms using `+`/`+=` operators
  (pr/495); function to concatenate multiple ArraySequence objects (pr/494)
* Support for numpy 1.12 (pr/500, pr/502) (MC, MB)
* Allow dtype specifiers as fileslice input (pr/485) (MB)
* Support "headerless" ArrayProxy specification, enabling memory-efficient
  ArrayProxy reshaping (pr/521) (CM)
* Allow unknown NIfTI intent codes, add FSL codes (pr/528) (PM)
* Improve error handling for ``img.__getitem__`` (pr/533) (Ariel Rokem)
* Delegate reorientation to SpatialImage classes (pr/544) (Mark Hymers, CM,
  reviewed by MB)
* Enable using ``indexed_gzip`` to reduce memory usage when reading from
  gzipped NIfTI and MGH files (pr/552) (PM, reviewed by MB, CM)

Bug fixes
---------

* Miscellaneous MINC reader fixes (pr/493) (Robert D. Vincent, reviewed by CM,
  MB)
* Fix corner case in ``wrapstruct.get`` (pr/516) (PM, reviewed by
  CM, MB)

Maintenance
-----------

* Fix documentation errors (pr/517, pr/536) (Fernando Perez, Venky Reddy)
* Documentation update (pr/514) (Ivan Gonzalez)
* Update testing to use pre-release builds of dependencies (pr/509) (MB)
* Better warnings when nibabel not on path (pr/503) (MB)

API changes and deprecations
----------------------------

* ``header`` argument to ``ArrayProxy.__init__`` is renamed to ``spec``
* Deprecation of ``header`` property of ``ArrayProxy`` object, for removal in
  3.0
* ``wrapstruct.get`` now returns entries evaluating ``False``, instead of ``None``
* ``DataobjImage.get_data`` to be deprecated April 2018, scheduled for removal
  April 2020


2.1 (Monday 22 August 2016)
===========================

New features
------------

* New API for managing streamlines and their different file formats. This
  adds a new module ``nibabel.streamlines`` that will eventually deprecate
  the current trackvis reader found in ``nibabel.trackvis`` (pr/391) (MC,
  reviewed by Jean-Christophe Houde, Bago Amirbekian, Eleftherios
  Garyfallidis, Samuel St-Jean, MB);
* A prototype image viewer using matplotlib (pr/404) (EL, based on a
  proto-prototype by Paul Ivanov) (Reviewed by Gregory R. Lee, MB);
* Functions for image resampling and smoothing using scipy ndimage (pr/255)
  (MB, reviewed by EL, BC);
* Add ability to write FreeSurfer morphology data (pr/414) (CM, BC, reviewed
  by BC);
* Read and write support for DICOM tags in NIfTI Extended Header using
  pydicom (pr/296) (Eric Kastman).

Enhancements
------------

* Extensions to FreeSurfer module to fix reading and writing of FreeSurfer
  geometry data (pr/460) (Alexandre Gramfort, Jaakko Leppäkangas, reviewed
  by EL, CM, MB);
* Various improvements to PAR / REC handling by Gregory R. Lee: supporting
  multiple TR values (pr/429); output of volume labels (pr/427); fix for
  some diffusion files (pr/426); option for more sophisticated sorting of
  volumes (pr/409);
* Original trackvis reader will now allow final streamline to have fewer
  points than the number declared in the header, with ``strict=False``
  argument to ``read`` function;
* Helper function to return voxel sizes from an affine matrix (pr/413);
* Fixes to DICOM multiframe reading to avoid assumptions on the position of
  the multiframe index (pr/439) (Eric M. Baker);
* More robust handling of "CSA" private information in DICOM files (pr/393)
  (Brendan Moloney);
* More explicit error when trying to read image from non-existent file
  (pr/455) (Ariel Rokem);
* Extension to `nib-ls` command to show image statistics (pr/437) and other
  header files (pr/348) (Yarik Halchenko).

Bug fixes
---------

* Fixes to rotation order to generate affine matrices of PAR / REC files (MB,
  Gregory R Lee).

Maintenance
-----------

* Dropped support for Pythons 2.6 and 3.2;
* Comprehensive refactor and generalization of surface / GIFTI file support
  with improved API and extended tests (pr/352-355, pr/360, pr/365, pr/403)
  (BC, reviewed by CM, MB);
* Refactor of image classes (pr/328, pr/329) (BC, reviewed by CM);
* Better Appveyor testing on new Python versions (pr/446) (Ariel Rokem);
* Fix shebang lines in scripts for correct install into virtualenvs via pip
  (pr/434);
* Various fixes for numpy, matplotlib, and PIL / Pillow compatibility (CM,
  Ariel Rokem, MB);
* Improved test framework for warnings (pr/345) (BC, reviewed by CM, MB);
* New decorator to specify start and end versions for deprecation warnings
  (MB, reviewed by CM);
* Write qform affine matrix to NIfTI images output by ``parrec2nii`` (pr/478)
  (Jasper J.F. van den Bosch, reviewed by Gregory R. Lee, MB).

API changes and deprecations
----------------------------

* Minor API breakage in original (rather than new) trackvis reader. We are now
  raising a ``DataError`` if there are too few streamlines in the file,
  instead of a ``HeaderError``.  We are raising a ``DataError`` if the track
  is truncated when ``strict=True`` (the default), rather than a ``TypeError``
  when trying to create the points array.
* Change sform code that ``parrec2nii`` script writes to NIfTI images; change
  from 2 ("aligned") to 1 ("scanner");
* Deprecation of ``get_header``, ``get_affine`` method of image objects for
  removal in version 4.0;
* Removed broken ``from_filespec`` method from image objects, and deprecated
  ``from_filespec`` method of ECAT image objects for removal in 4.0;
* Deprecation of ``class_map`` instance in ``imageclasses`` module in favor of
  new image class attributes, for removal in 4.0;
* Deprecation of ``ext_map`` instance in ``imageclasses`` module in favor of
  new image loading API, for removal in 4.0;
* Deprecation of ``Header`` class in favor of ``SpatialHeader``, for removal
  in 4.0;
* Deprecation of ``BinOpener`` class in favor of more generic ``Opener``
  class, for removal in 4.0;
* Deprecation of ``GiftiMetadata`` methods ``get_metadata`` and ``get_rgba``;
  ``GiftiDataArray`` methods ``get_metadata``, ``get_labeltable``,
  ``set_labeltable``; ``GiftiImage`` methods ``get_meta``, ``set_meta``.  All
  these deprecated in favor of corresponding properties, for removal in 4.0;
* Deprecation of ``giftiio`` ``read`` and ``write`` functions in favor of
  nibabel ``load`` and ``save`` functions, for removal in 4.0;
* Deprecation of ``gifti.data_tag`` function, for removal in 4.0;
* Deprecation of write-access to ``GiftiDataArray.num_dim``, and new error
  when trying to set invalid values for ``num_dim``.  We will remove
  write-access in 4.0;
* Deprecation of ``GiftiDataArray.from_array`` in favor of ``GiftiDataArray``
  constructor, for removal in 4.0;
* Deprecation of ``GiftiDataArray`` ``to_xml_open, to_xml_close`` methods in
  favor of ``to_xml`` method, for removal in 4.0;
* Deprecation of ``parse_gifti_fast.Outputter`` class in favor of
  ``GiftiImageParser``, for removal in 4.0;
* Deprecation of ``parse_gifti_fast.parse_gifti_file`` function in favor of
  ``GiftiImageParser.parse`` method, for removal in 4.0;
* Deprecation of ``loadsave`` functions ``guessed_image_type`` and
  ``which_analyze_type``, in favor of new API where each image class tests the
  file for compatibility during load, for removal in 4.0.

2.0.2 (Monday 23 November 2015)
===============================

* Fix for integer overflow on large images (pr/325) (MB);
* Fix for Freesurfer nifti files with unusual dimensions (pr/332) (Chris
  Markiewicz);
* Fix typos on benchmarks and tests (pr/336, pr/340, pr/347) (Chris
  Markiewicz);
* Fix Windows install script (pr/339) (MB);
* Support for Python 3.5 (pr/363) (MB) and numpy 1.10 (pr/358) (Chris
  Markiewicz);
* Update pydicom imports to permit version 1.0 (pr/379) (Chris Markiewicz);
* Workaround for Python 3.5.0 gzip regression (pr/383) (Ben Cipollini).
* tripwire.TripWire object now raises subclass of AttributeError when trying
  to get an attribute, rather than a direct subclass of Exception.  This
  prevents Python 3.5 triggering the tripwire when doing inspection prior to
  running doctests.
* Minor API change for tripwire.TripWire object; code that checked for
  AttributeError will now also catch TripWireError.

2.0.1 (Saturday 27 June 2015)
=============================

Contributions from Ben Cipollini, Chris Markiewicz, Alexandre Gramfort,
Clemens Bauer, github user freec84.

* Bugfix release with minor new features;
* Added ``axis`` parameter to ``concat_images`` (pr/298) (Ben Cipollini);
* Fix for unsigned integer data types in ECAT images (pr/302) (MB, test data
  and issue report from Github user freec84);
* Added new ECAT and Freesurfer data files to automated testing;
* Fix for Freesurfer labels error on early numpies (pr/307) (Alexandre
  Gramfort);
* Fixes for PAR / REC header parsing (pr/312) (MB, issue reporting and test
  data by Clemens C. C. Bauer);
* Workaround for reading Freesurfer ico7 surface files (pr/315) (Chris
  Markiewicz);
* Changed to github pages for doc hosting;
* Changed docs to point to neuroimaging@python.org mailing list.

2.0.0 (Tuesday 9 December 2014)
===============================

This release had large contributions from Eric Larson, Brendan Moloney,
Nolan Nichols, Basile Pinsard, Chris Johnson and Nikolaas N. Oosterhof.

* New feature, bugfix release with minor API breakage;
* Minor API breakage: default write of NIfTI / Analyze image data offset
  value. The data offset is the number of bytes from the beginning of file
  to skip before reading the image data.  Nibabel behavior changed from
  keeping the value as read from file, to setting the offset to zero on
  read, and setting the offset when writing the header. The value of the
  offset will now be the minimum value necessary to make room for the header
  and any extensions when writing the file. You can override the default
  offset by setting value explicitly to some value other than zero. To read
  the original data offset as read from the header, use the ``offset``
  property of the image ``dataobj`` attribute;
* Minor API breakage: data scaling in NIfTI / Analyze now set to NaN when
  reading images.  Data scaling refers to the data intercept and slope
  values in the NIfTI / Analyze header.  To read the original data scaling
  you need to look at the ``slope`` and ``inter`` properties of the image
  ``dataobj`` attribute.  You can set scaling explicitly by setting the
  slope and intercept values in the header to values other than NaN;
* New API for managing image caching; images have an ``in_memory`` property
  that is true if the image data has been loaded into cache, or is already
  an array in memory; ``get_data`` has new keyword argument ``caching`` to
  specify whether the cache should be filled by ``get_data``;
* Images now have properties ``dataobj``, ``affine``, ``header``. We will
  slowly phase out the ``get_affine`` and ``get_header`` image methods;
* The image ``dataobj`` can be sliced using an efficient algorithm to avoid
  reading unnecessary data from disk.  This makes it possible to do very
  efficient reads of single volumes from a time series;
* NIfTI2 read / write support;
* Read support for MINC2;
* Much extended read support for PAR / REC, largely due to work from Eric
  Larson and Gregory R. Lee on new code, advice and code review. Thanks also
  to Jeff Stevenson and Bennett Landman for helpful discussion;
* ``parrec2nii`` script outputs images in LAS voxel orientation, which
  appears to be necessary for compatibility with FSL ``dtifit`` /
  ``fslview`` diffusion analysis pipeline;
* Preliminary support for Philips multiframe DICOM images (thanks to Nolan
  Nichols, Ly Nguyen and Brendan Moloney);
* New function to save Freesurfer annotation files (by Github user ohinds);
* Method to return MGH format ``vox2ras_tkr`` affine (Eric Larson);
* A new API for reading unscaled data from NIfTI and other images, using
  ``img.dataobj.get_unscaled()``. Deprecate previous way of doing this,
  which was to read data with the ``read_img_data`` function;
* Fix for bug when replacing NaN values with zero when writing floating
  point data as integers.  If the input floating point data range did not
  include zero, then NaN would not get written to a value corresponding to
  zero in the output;
* Improvements and bug fixes to image orientation calculation and DICOM
  wrappers by Brendan Moloney;
* Bug fixes writing GIfTI files. We were using a base64 encoding that didn't
  match the spec, and the wrong field name for the endian code. Thanks to
  Basile Pinsard and Russ Poldrack for diagnosis and fixes;
* Bug fix in ``freesurfer.read_annot`` with ``orig_ids=False`` when annot
  contains vertices with no label (Alexandre Gramfort);
* More tutorials in the documentation, including introductory tutorial on
  DICOM, and on coordinate systems;
* Lots of code refactoring, including moving to common code-base for Python
  2 and Python 3;
* New mechanism to add images for tests via git submodules.

1.3.0 (Tuesday 11 September 2012)
=================================

Special thanks to Chris Johnson, Brendan Moloney and JB Poline.

* New feature and bugfix release
* Add ability to write Freesurfer triangle files (Chris Johnson)
* Relax threshold for detecting rank deficient affines in orientation
  detection (JB Poline)
* Fix for DICOM slice normal numerical error (issue #137) (Brendan Moloney)
* Fix for Python 3 error when writing zero bytes for offset padding

1.2.2 (Wednesday 27 June 2012)
==============================

* Bugfix release
* Fix longdouble tests for Debian PPC (thanks to Yaroslav Halchecko for
  finding and diagnosing these errors)
* Generalize longdouble tests in the hope of making them more robust
* Disable saving of float128 nifti type unless platform has real IEEE
  binary128 longdouble type.

1.2.1 (Wednesday 13 June 2012)
==============================

Particular thanks to Yaroslav Halchecko for fixes and cleanups in this
release.

* Bugfix release
* Make compatible with pydicom 0.9.7
* Refactor, rename nifti diagnostic script to ``nib-nifti-dx``
* Fix a bug causing an error when analyzing affines for orientation, when the
  affine contained all 0 columns
* Add missing ``dicomfs`` script to installation list and rename to
  ``nib-dicomfs``

1.2.0 (Sunday 6 May 2012)
=========================

This release had large contributions from Krish Subramaniam, Alexandre
Gramfort, Cindee Madison, Félix C. Morency and Christian Haselgrove.

* New feature and bugfix release
* Freesurfer format support by Krish Subramaniam and Alexandre Gramfort.
* ECAT read write support by Cindee Madison and Félix C. Morency.
* A DICOM fuse filesystem by Christian Haselgrove.
* Much work on making data scaling on read and write more robust to rounding
  error and overflow (MB).
* Import of nipy functions for working with affine transformation matrices.
* Added methods for working with nifti sform and qform fields by Bago
  Amirbekian and MB, with useful discussion by Brendan Moloney.
* Fixes to read / write of RGB analyze images by Bago Amirbekian.
* Extensions to ``concat_images`` by Yannick Schwartz.
* A new ``nib-ls`` script to display information about neuroimaging files, and
  various other useful fixes by Yaroslav Halchenko.

1.1.0 (Thursday 28 April 2011)
==============================

Special thanks to Chris Burns, Jarrod Millman and Yaroslav Halchenko.

* New feature release
* Python 3.2 support
* Substantially enhanced gifti reading support (Stephan Gerhard)
* Refactoring of trackvis read / write to allow reading and writing of voxel
  points and mm points in tracks.  Deprecate use of negative voxel sizes;
  set voxel_order field in trackvis header.  Thanks to Chris Filo
  Gorgolewski for pointing out the problem and Ruopeng Wang in the trackvis
  forum for clarifying the coordinate system of trackvis files.
* Added routine to give approximate array orientation in form such as 'RAS'
  or 'LPS'
* Fix numpy dtype hash errors for numpy 1.2.1
* Other bug fixes as for 1.0.2

1.0.2 (Thursday 14 April 2011)
==============================

* Bugfix release
* Make inference of data type more robust to changes in numpy dtype hashing
* Fix incorrect thresholds in quaternion calculation (thanks to Yarik H for
  pointing this one out)
* Make parrec2nii pass over errors more gracefully
* More explicit checks for missing or None field in trackvis and other
  classes - thanks to Marc-Alexandre Cote
* Make logging and error level work as expected - thanks to Yarik H
* Loading an image does not change qform or sform - thanks to Yarik H
* Allow 0 for nifti scaling as for spec - thanks to Yarik H
* nifti1.save now correctly saves single or pair images

1.0.1 (Wednesday 23 Feb 2011)
=============================

* Bugfix release
* Fix bugs in tests for data package paths
* Fix leaks of open filehandles when loading images (thanks to Gael
  Varoquaux for the report)
* Skip rw tests for SPM images when scipy not installed
* Fix various windows-specific file issues for tests
* Fix incorrect reading of byte-swapped trackvis files
* Workaround for odd numpy dtype comparisons leading to header errors for
  some loaded images (thanks to Cindee Madison for the report)

1.0.0 (Thursday, 13, Oct 2010)
==============================

* This is the first public release of the NiBabel package.
* NiBabel is a complete rewrite of the PyNifti package in pure python.  It was
  designed to make the code simpler and easier to work with. Like PyNifti,
  NiBabel has fairly comprehensive NIfTI read and write support.
* Extended support for SPM Analyze images, including orientation affines from
  matlab ``.mat`` files.
* Basic support for simple MINC 1.0 files (MB).  Please let us know if you
  have MINC files that we don't support well.
* Support for reading and writing PAR/REC images (MH)
* ``parrec2nii`` script to convert PAR/REC images to NIfTI format (MH)
* Very preliminary, limited and highly experimental DICOM reading support (MB,
  Ian Nimmo Smith).
* Some functions (`nibabel.funcs`) for basic image shape changes, including
  the ability to transform to the image with data closest to the cononical
  image orientation (first axis left-to-right, second back-to-front, third
  down-to-up) (MB, Jonathan Taylor)
* Gifti format read and write support (preliminary) (Stephen Gerhard)
* Added utilities to use nipy-style data packages, by rip then edit of nipy
  data package code (MB)
* Some improvements to release support (Jarrod Millman, MB, Fernando Perez)
* Huge downward step in the quality and coverage by the docs, caused by MB,
  mostly fixed by a lot of good work by MH.
* NiBabel will not work with Python < 2.5, and we haven't even tested it with
  Python 3.  We will get to it soon...

****************
PyNifti releases
****************

Modifications are done by Michael Hanke, if not indicated otherwise. 'Closes'
statement IDs refer to the Debian bug tracking system and can be queried by
visiting the URL::

  http://bugs.debian.org/<bug id>

0.20100706.1 (Tue, 6 Jul 2010)
==============================

* Bugfix: NiftiFormat.vx2s() used the qform not the sform. Thanks to Tom
  Holroyd for reporting.

0.20100412.1 (Mon, 12 Apr 2010)
===============================

* Bugfix: Unfortunate interaction between Python garbage collection and C
  library caused memory problems. Thanks to Yaroslav Halchenko for the
  diagnose and fix.

0.20090303.1 (Tue, 3 Mar 2009)
==============================

* Bugfix: Updating the NIfTI header from a dictionary was broken.
* Bugfix: Removed left-over print statement in extension code.
* Bugfix: Prevent saving of bogus 'None.nii' images when the filename
  was previously assign, before calling NiftiImage.save() (Closes: #517920).
* Bugfix: Extension length was to short for all `edata` whos length matches
  n*16-8, for all integer n.

0.20090205.1 (Thu, 5 Feb 2009)
==============================

* This release is the first in a series that aims stabilize the API and
  finally result in PyNIfTI 1.0 with full support of the NIfTI1 standard.
* The whole package was restructured. The included renaming
  `nifti.nifti(image,format,clibs)` to `nifti.(image,format,clibs)`. Redirect
  modules make sure that existing user code will not break, but they will
  issue a DeprecationWarning and will be removed with the release of PyNIfTI
  1.0.
* Added a special extension that can embed any serializable Python object
  into the NIfTI file header. The contents of this extension is
  automatically expanded upon request into the `.meta` attribute of each
  NiftiImage. When saving files to disk the content of the dictionary is also
  automatically dumped into this extension.
  Embedded meta data is not loaded automatically, since this has security
  implications, because code from the file header is actually executed.
  The documentation explicitely mentions this risk.
* Added :class:`~nifti.extensions.NiftiExtensions`. This is a container-like
  handler to access and manipulate NIfTI1 header extensions.
* Exposed :class:`~nifti.image.MemMappedNiftiImage` in the root module.
* Moved :func:`~nifti.utils.cropImage` into the :mod:`~nifti.utils` module.
* From now on Sphinx is used to generate the documentation. This includes a
  module reference that replaces that old API reference.
* Added methods :meth:`~nifti.format.NiftiFormat.vx2q` and
  :meth:`~nifti.format.NiftiFormat.vx2s` to convert voxel indices into
  coordinates defined by qform or sform respectively.
* Updating the `cal_min` and `cal_max` values in the NIfTI header when
  saving a file is now conditional, but remains enabled by default.
* Full set of methods to query and modify axis units. This includes
  expanding the previous `xyzt_units` field in the header dictionary into
  editable `xyz_unit` and `time_unit` fields. The former `xyzt_units` field
  is no longer available. See:
  :meth:`~nifti.format.NiftiFormat.getXYZUnit`,
  :meth:`~nifti.format.NiftiFormat.setXYZUnit`,
  :meth:`~nifti.format.NiftiFormat.getTimeUnit`,
  :meth:`~nifti.format.NiftiFormat.setTimeUnit`,
  :attr:`~nifti.format.NiftiFormat.xyz_unit`,
  :attr:`~nifti.format.NiftiFormat.time_unit`
* Full set of methods to query and manuipulate qform and sform codes. See:
  :meth:`~nifti.format.NiftiFormat.getQFormCode`,
  :meth:`~nifti.format.NiftiFormat.setQFormCode`,
  :meth:`~nifti.format.NiftiFormat.getSFormCode`,
  :meth:`~nifti.format.NiftiFormat.setSFormCode`,
  :attr:`~nifti.format.NiftiFormat.qform_code`,
  :attr:`~nifti.format.NiftiFormat.sform_code`
* Each image instance is now able to generate a human-readable dump of its
  most important header information via `__str__()`.
* :class:`~nifti.image.NiftiImage` objects can now be pickled.
* Switched to NumPy's distutils for building the package. Cleaned and
  simplified the build procedure. Added optimization flags to SWIG call.
* :attr:`nifti.image.NiftiImage.filename` can now also be used to assign a
  filename.
* Introduced :data:`nifti.__version__` as canonical version string.
* Removed `updateQFormFromQuarternion()` from the list of public methods of
  :class:`~nifti.format.NiftiFormat`. This is an internal method that
  should not be used in user code. However, a redirect to the new method
  will remain in-place until PyNIfTI 1.0.
* Bugfix: :meth:`~nifti.image.NiftiImage.getScaledData` returns a
  unmodified data array if `slope` is set to zero (as required by the NIfTI
  standard). Thanks to Thomas Ross for reporting.
* Bugfix: Unicode filenames are now handled properly, as long as they do not
  contain pure-unicode characters (since the NIfTI library does not support
  them). Thanks to Gaël Varoquaux for reporting this issue.

0.20081017.1 (Fri, 17 Oct 2008)
===============================

* Updated included minimal copy of the nifticlibs to version 1.1.0.
* Few changes to the Makefiles to enhance Posix compatibility. Thanks to
  Chris Burns.
* When building on non-Debian systems, only add include and library paths
  pointing to the local nifticlibs copy, when it is actually built.
  On Debian system the local copy is still not used at all, as a proper
  nifticlibs package is guaranteed to be available.
* Added minimal setup_egg.py for setuptools users. Thanks to Gaël Varoquaux.
* PyNIfTI now does a proper wrapping of the image data with NumPy arrays,
  which no longer leads to accidental memory leaks, when accessing array
  data that has not been copied before (e.g. via the *data* property of
  NiftiImage). Thanks to Gaël Varoquaux for mentioning this possibility.

0.20080710.1 (Thu, 7 Jul 2008)
==============================

* Bugfix: Pointer bug introduced by switch to new NumPy API in 0.20080624
  Thanks to Christopher Burns for fixing it.
* Bugfix: Honored DeprecationWarning: sync() -> flush() for memory mapped
  arrays. Again thanks to Christopher Burns.
* More unit tests and other improvements (e.g. fixed circular imports) done
  by Christopher Burns.

0.20080630.1 (Tue, 30 Jun 2008)
===============================

* Bugfix: NiftiImage caused a memory leak by not calling the NiftiFormat
  destructor.
* Bugfix: Merged bashism-removal patch from Debian packaging.

0.20080624.1 (Tue, 24 Jun 2008)
===============================

* Converted all documentation (including docstrings) into the restructured
  text format.
* Improved Makefile.
* Included configuration and Makefile support for profiling, API doc
  generation (via epydoc) and code quality checks (with PyLint).
* Consistently import NumPy as N.
* Bugfix: Proper handling of [qs]form codes, which previously have not been
  handled at all. Thanks to Christopher Burns for pointing it out.
* Bugfix: Make NiftiFormat work without setFilename(). Thanks to Benjamin
  Thyreau for reporting.
* Bugfix: setPixDims() stored meaningless values.
* Use new NumPy API and replace deprecated function calls
  (`PyArray_FromDimsAndData`).
* Initial support for memory mapped access to uncompressed NIfTI files
  (`MemMappedNiftiImage`).
* Add a proper Makefile and setup.cfg for compiling PyNIfTI under Windows
  with MinGW.
* Include a minimal copy of the most recent nifticlibs (just libniftiio and
  znzlib; version 1.0), to lower the threshold to build PyNIfTI on systems
  that do not provide a developer package for those libraries.

0.20070930.1 (Sun, 30 Sep 2007)
===============================

* Relicense under the MIT license, to be compatible with SciPy license.
  http://www.opensource.org/licenses/mit-license.php
* Updated documentation.

0.20070917.1 (Mon, 17 Sep 2007)
===============================

* Bugfix: Can now update NIfTI header data when no filename is set
  (Closes: #442175).
* Unloading of image data without a filename set is no checked and prevented
  as it would damage data integrity and the image data could not be
  recovered.
* Added 'pixdim' property (Yaroslav Halchenko).

0.20070905.1  (Wed, 5 Sep 2007)
===============================

* Fixed a bug in the qform/quaternion handling that caused changes to the
  qform to vanish when saving to file (Yaroslav Halchenko).
* Added more unit tests.
* 'dim' vector in the NIfTI header is now guaranteed to only contain
  non-zero elements. This caused problems with some applications.

0.20070803.1 (Fri, 3 Aug 2007)
==============================

* Does not depend on SciPy anymore.
* Initial steps towards a unittest suite.
* pynifti_pst can now print the peristimulus signal matrix for a single
  voxel (onsets x time) for easier processing of this information in
  external applications.
* utils.getPeristimulusTimeseries() can now be used to compute mean and
  variance of the signal (among others).
* pynifti_pst is able to compute more than just the mean peristimulus
  timeseries (e.g. variance and standard deviation).
* Set default image description when saving a file if none is present.
* Improved documentation.

0.20070425.1 (Wed, 25 Apr 2007)
===============================

* Improved documentation. Added note about the special usage of the header
  property. Also added notes about the relevant properties in the docstring
  of the corresponding accessor methods.
* Added property and accessor methods to access/modify the repetition time
  of timeseries (dt).
* Added functions to manipulate the pixdim values.
* Added utils.py with some utility functions.
* Added functions/property to determine the bounding box of an image.
* Fixed a bug that caused a corrupted sform matrix when converting a NumPy
  array and a header dictionary into a NIfTI image.
* Added script to compute peristimulus timeseries (pynifti_pst).
* Package now depends on python-scipy.

0.20070315.1 (Thu, 15 Mar 2007)
===============================

* Removed functionality for "NiftiImage.save() raises an IOError
  exception when writing the image file fails." (Yaroslav Halchenko)
* Added ability to force a filetype when setting the filename or saving
  a file.
* Reverse the order of the 'header' and 'load' argument in the NiftiImage
  constructor. 'header' is now first as it seems to be used more often.
* Improved the source code documentation.
* Added getScaledData() method to NiftiImage that returns a copy of the data
  array that is scaled with the slope and intercept stored in the NIfTI
  header.

0.20070301.2 (Thu, 1 Mar 2007)
==============================

* Fixed wrong link to the source tarball in README.html.

0.20070301.1 (Thu, 1 Mar 2007)
==============================

* Initial upload to the Debian archive. (Closes: #413049)
* NiftiImage.save() raises an IOError exception when writing the image file
  fails.
* Added extent, volextent, and timepoints properties to NiftiImage
  class (Yaroslav Halchenko).

0.20070220.1 (Tue, 20 Feb 2007)
===============================

* NiftiFile class is renamed to NiftiImage.
* SWIG-wrapped libniftiio functions are no available in the nifticlib
  module.
* Fixed broken NiftiImage from Numpy array constructor.
* Added initial documentation in README.html.
* Fulfilled a number of Yarik's wishes ;)

0.20070214.1 (Wed, 14 Feb 2007)
===============================

* Does not depend on libfslio anymore.
* Up to seven-dimensional dataset are supported (as much as NIfTI can do).
* The complete NIfTI header dataset is modifiable.
* Most image properties are accessable via class attributes and accessor
  methods.
* Improved documentation (but still a long way to go).

0.20061114 (Tue, 14 Nov 2006)
=============================

* Initial release.

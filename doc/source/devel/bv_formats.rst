#########################
BrainVoyager file formats
#########################

With notes on nibabel support.

PR for some BrainVoyager support at https://github.com/nipy/nibabel/pull/216.

********
Overview
********

See :

* All files are little-endian byte order regardless of byte-order on the
  machine writing the data;
* BV apparently provides a "BVQXtools" library for reading writing BV files in
  MATLAB;

.. _bv-internal-axes:

***********************
BV internal format axes
***********************

BV files have a internal format that has axes named `X`, `Y` and `Z`.  Quoting
from the `VMR format definition`_::

    BV X front -> back = Y in Tal space
    BV Y top -> bottom = Z in Tal space
    BV Z left -> right = X in Tal space

Put another way |--| the correspondence of BV XYZ to Talairach axes is:

* BV X -> Anterior to Posterior;
* BV Y -> Superior to Inferior;
* BV Z -> Left to Right.

or:

* BV X -> Talairach -Y;
* BV Y -> Talairach -Z;
* BV Z -> Talairach  X;

Nice!

*****************
Types of BV files
*****************

There appear to be 38 BV file types at the time of writing of which 18 appear
to have a page of description on the `BV file format index page`_.

Here are some examples of BV formats:

* FMR |--| "FMR project files are simple text files containing the information
  defining a functional project created from raw MRI data".  This text file
  contains meta-data about the functional time course data, stored in one or
  more STC files.  See the `FMR format definition`_.
* STC |--| "A STC file (STC = "slice time course") contains the functional
  data (time series) of a FMR project."  The time-course data of a 4D
  ("single-slice") format STC file are stored on disk in
  fastest-to-slowest-changing order: columns, rows, time, slice.  STC files
  can also contain the data for one single slice ("multi-slice format"), in
  which case the data are in fast-to-slow order: columns, rows, time.  This is
  a raw data file where the relevant meta-data such as image size come from an
  associated FMR format file. See `STC format definition`_;
* VTC |--| "A VTC file contains the functional data (time series) of one
  experimental run (one functional scan) in the space of a 3D anatomical data
  set (VMR), e.g. in Talairach space.".  See `VTC format definition`_;
  This is a different format to the STC (raw data in native-space) format.
  The file is a header followed by ints or floats in
  fastest-to-slowest-changing order of: time; BV X; BV Y; BV Z; where BV X, BV
  Y, BV Z refer to the :ref:`bv-internal-axes`, and therefore Talairach -Y,
  -Z, X.
* NR-VMP |--| "A native resolution volume map (NR-VMP) file contains
  statistical results in 3D format.". See `NR-VMP format definition`_
* AR-VMP |--| "An anatomical-resolution VMP (volume map) file contains
  statistical results in 3D format" at anatomical scan resolution.  See
  `AR-VMP format definition`_;
* VMR |--| 'high-resolution anatomical MR' - see `VMR format definition`_.
* MSK |--| mask file.  Only documentation appears to be
  http://www.brainvoyager.com/ubb/Forum8/HTML/000087.html
* SMP |--| 'surface map'.  See `SMP format definition`_. Contains one or more
  "maps", where a map is a ``NrOfVertices`` (number of vertices) length vector
  of float64 values.

.. _BV file format index page: http://support.brainvoyager.com/automation-aamp-development/23-file-formats.html
.. _AR-VMP format definition: http://support.brainvoyager.com/automation-aamp-development/23-file-formats/376-users-guide-23-the-format-of-ar-vmp-files.html
.. _NR-VMP format definition: http://support.brainvoyager.com/automation-aamp-development/23-file-formats/377-users-guide-23-the-format-of-nr-vmp-files.html
.. _VTC format definition: http://support.brainvoyager.com/automation-aamp-development/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html
.. _BV file format overview: http://support.brainvoyager.com/automation-aamp-development/23-file-formats/382-developer-guide-26-file-formats-overview.html
.. _FMR format definition: http://support.brainvoyager.com/installation-introduction/23-file-formats/383-developer-guide-26-the-format-of-fmr-files.html
.. _STC format definition: http://support.brainvoyager.com/automation-aamp-development/23-file-formats/384-developer-guide-26-the-format-of-stc-files.html
.. _vmr format definition: http://support.brainvoyager.com/automation-aamp-development/23-file-formats/385-developer-guide-26-the-format-of-vmr-files.html
.. _SMP format definition: : http://support.brainvoyager.com/automation-aamp-development/23-file-formats/476-the-format-of-smp-files.html

.. include:: ../links_names.txt

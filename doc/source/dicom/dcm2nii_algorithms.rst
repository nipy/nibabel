.. _dcm2nii-algorithms:

====================
 dcm2nii algorithms
====================

dcm2nii_ is an open source DICOM_ to nifti_ conversion program, written
by Chris Rorden, in Delphi (object orientated pascal).  It's part of
Chris' popular mricron_ collection of programs.  The source appears to
be best found on the `mricron NITRC site`_.  It's BSD_ licensed. 

.. _mricron NITRC site: https://www.nitrc.org/projects/mricron

These are working notes looking at Chris' algorithms for working with
DICOM.

Compiling dcm2nii
=================

Follow the download / install instructions at the
http://www.lazarus.freepascal.org/ site.  I was on a Mac, and folowed the
instructions here:
http://wiki.lazarus.freepascal.org/Installing_Lazarus_on_MacOS_X .  Default
build with version 0.9.28.2 gave an error linking against Carbon, so I needed to
download a snapshot of fixed Lazarus 0.9.28.3 from
http://www.hu.freepascal.org/lazarus . Open ``<mricron>/dcm2nii/dcm2nii.lpi``
using the Lazarus GUI.  Follow instructions for compiler setup in the mricron
``Readme.txt``; in particular I set other compiler options to::

     -k-macosx_version_min -k10.5
     -XR/Developer/SDKs/MacOSX10.5.sdk/

Further inspiration for building also came from the ``debian/rules`` file in
Michael Hanke's mricron debian package:
http://neuro.debian.net/debian/pool/main/m/mricron/

Some tag modifications
======================

Note - Chris tells me that ``dicomfastread.pas`` was an attempt to do a fast
dicom read that is not yet fully compatible, and that the algorithm used is in
fact ``dicomcompat.pas``.

Looking in the source file ``<mricron>/dcm2nii/dicomfastread.pas``.

Named fields here are as from :ref:`dicom-fields`

* If 'MOSAIC' is the last string in 'ImageType', this is a mosaic
* 'DateTime' field is combination of 'StudyDate' and 'StudyTime'; fixes
  in file ``dicomtypes.pas`` for different scanner date / time formats.
* AcquisitionNumber read as normal, but then set to 1, if this a mosaic
  image, as set above.
* If 'EchoNumbers' > 0 and < 16, add 'EchoNumber' * 100 to the
  'AcquisitionNumber' - presumably to identify different echos from the
  same series as being different series.
* If 'ScanningSequence' sequence contains 'RM', add 100 to the
  'SeriesNumber' - maybe to differentiate research and not-research
  scans with the same acquisition number.
* is_4D flag labeling DICOM file as a 4D file:

   * There's a Philips private tag (2001, 1018) - labeled 'Number of
     Slices MR' by pydicom_ call this ``NS``
   * If ``NS>0`` and 'NumberofTemporalPositions' > 0, and
     'NumberOfFrames' is > 1

Sorting slices into volumes
===========================

Looking in the source file ``<mricron>/dcm2nii/sortdicom.pas``.

In function ``ShellSortDCM``:

Sort compares two dicom images, call them ``dcm1`` and ``dcm2``.   Tests are:

#. Are the two images 'repeats' - defined by same 'InstanceNumber'
   (0020, 0013), and 'AcquisitionNumber' (0020, 0012) and 'SeriesNumber'
   (0020, 0011) and a combination of 'StudyDate' and 'StudyTime')?  Then
   report an error about files having the same index, flag repeated values.
#. Is ``dcm1`` less than ``dcm2``, defined with comparisons in the
   following order:

   #. StudyDate/Time
   #. SeriesNumber
   #. AcquisitionNumber
   #. InstanceNumber

   This should obviously only ever be > or <, not ==, because of the
   first check.

Next remove repeated values as found in the first step above.

.. include:: ../links_names.txt

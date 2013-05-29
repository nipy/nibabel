.. _dicom-intro:

#####################
Introduction to DICOM
#####################

DICOM defines standards for storing data in memory and on disk, and for
communicating this data between machines over a network.

We are interested here in DICOM data.  Specifically we are interested in DICOM
files.

DICOM files are binary dumps of the objects in memory that DICOM sends across
the network.

We need to understand the format that DICOM uses to send messages across the
network to understand the terms the DICOM uses when storing data in files.

For example, I hope, by the time you reach the end of this document, you will
understand the following complicated and confusing statement from section 7 of
the DICOM standards document `PS 3.10`_:

    7   DICOM File Format

    The DICOM File Format provides a means to encapsulate in a file the Data Set
    representing a SOP Instance related to a DICOM IOD. As shown in Figure 7-1,
    the byte stream of the Data Set is placed into the file after the DICOM File
    Meta Information. Each file contains a single SOP Instance.

*****************
DICOM is messages
*****************

The fundamental task of DICOM is to allow different computers to send messages
to one another.  These messages can contain data, and the data is very often
medical images.

The messages are in the form of requests for an operation, or responses to those requests.

Let's call the requests and the responses - services.

Every DICOM message starts with a stream of bytes containing information about
the service.  This part of the message is called the DICOM Message Service
Element or DIMSE.  Depending on what the DIMSE was, there may follow some data
related to the request.

For example, there is a DICOM service called "C-ECHO".  This asks for a response
from another computer to confirm it has seen the echo request.  There is no
associated data following the "C-ECHO" DIMSE part.  So, the full message is the
DIMSE "C-ECHO".

There is another DICOM service called "C-STORE".  This is a request for the
other computer to store some data, such as an image.  The data to be stored
follows the "C-STORE" DIMSE part.

We go into more detail on this later in the page.

Both the DIMSE and the subsequent data have a particular binary format -
consisting of DICOM elements (see below).

Here we will cover:

* what DICOM elements are;
* how DICOM elements are arranged to form complicated data structures such as images;
* how the service part and the data part go together to form whole messages
* how these parts relate to DICOM files.

******************
The DICOM standard
******************

The documents defining the standard are:

+-------------+------------------------------------------------------------------+
| Number      | Name                                                             |
+=============+==================================================================+
| `PS 3.1`_   | Introduction and Overview                                        |
+-------------+------------------------------------------------------------------+
| `PS 3.2`_   | Conformance                                                      |
+-------------+------------------------------------------------------------------+
| `PS 3.3`_   | Information Object Definitions                                   |
+-------------+------------------------------------------------------------------+
| `PS 3.4`_   | Service Class Specifications                                     |
+-------------+------------------------------------------------------------------+
| `PS 3.5`_   | Data Structure and Encoding                                      |
+-------------+------------------------------------------------------------------+
| `PS 3.6`_   | Data Dictionary                                                  |
+-------------+------------------------------------------------------------------+
| `PS 3.7`_   | Message Exchange                                                 |
+-------------+------------------------------------------------------------------+
| `PS 3.8`_   | Network Communication Support for Message Exchange               |
+-------------+------------------------------------------------------------------+
| PS 3.9      | Retired                                                          |
+-------------+------------------------------------------------------------------+
| `PS 3.10`_  | Media Storage / File Format for Media Interchange                |
+-------------+------------------------------------------------------------------+
| `PS 3.11`_  | Media Storage Application Profiles                               |
+-------------+------------------------------------------------------------------+
| `PS 3.12`_  | Media Formats / Physical Media for Media Interchange             |
+-------------+------------------------------------------------------------------+
| PS 3.13     | Retired                                                          |
+-------------+------------------------------------------------------------------+
| `PS 3.14`_  | Grayscale Standard Display Function                              |
+-------------+------------------------------------------------------------------+
| `PS 3.15`_  | Security and System Management Profiles                          |
+-------------+------------------------------------------------------------------+
| `PS 3.16`_  | Content Mapping Resource                                         |
+-------------+------------------------------------------------------------------+
| `PS 3.17`_  | Explanatory Information                                          |
+-------------+------------------------------------------------------------------+
| `PS 3.18`_  | Web Access to DICOM Persistent Objects (WADO)                    |
+-------------+------------------------------------------------------------------+
| `PS 3.19`_  | Application Hosting                                              |
+-------------+------------------------------------------------------------------+
| `PS 3.20`_  | Transformation of DICOM to and from HL7 Standards                |
+-------------+------------------------------------------------------------------+

*****************
DICOM data format
*****************

DICOM data is stored in memory and on disk as a sequence of *DICOM elements*
(section 7 of `PS 3.5`_).

DICOM elements
==============

A DICOM element is made up of three or four fields.  These are (Attribute Tag,
[Value Representation, ], Value Length, Value Field), where *Value
Representation* may be present or absent, depending on the type of "Value
Representation Encoding" (see below)

Attribute Tag
-------------

The attribute tag is a pair of 16-bit unsigned integers of form (Group number,
Element number).  The tag uniquely identifies the element.

The *Element number* is badly named, because the element number does not give a
unique number for the element, but only for the element within the group (given
by the *Group number*).

The (Group number, Element number) are nearly always written as hexadecimal
numbers in the following format: ``(0010, 0010)``.  The decimal representation
of hexadecimal 0010 is 16, so this tag refers to group number 16, element number
16.  If you look this tag up in the DICOM data dictionary (`PS 3.6`_) you'll see
this must be the element called "PatientName".

These tag groups have special meanings:

+-----------+--------------------------------+
| Tag group | Meaning                        |
+===========+================================+
| 0000      | Command elements               |
+-----------+--------------------------------+
| 0002      | File meta elements             |
+-----------+--------------------------------+
| 0004      | Directory structuring elements |
+-----------+--------------------------------+
| 0006      | (not used)                     |
+-----------+--------------------------------+

See Annex E (command dictionary) of `PS 3.7`_ for details on group 0000.  See
sections 7 and 8 of `PS 3.6`_ for details of groups 2 and 4 respectively.

Tags in groups 0000, 0002, 0004 are therefore not *data* elements, but Command
elements; File meta elements; directory structuring elements.

Tags with groups from 0008 are *data* element tags.

Standard attribute tags
^^^^^^^^^^^^^^^^^^^^^^^

*Standard* tags are tags with an even group number (see below).  There is a full
list of all *standard* data element tags in the DICOM data dictionary in section
6 of DICOM standard `PS 3.6`_.

Even numbered groups are defined in the DICOM standard data dictionary.  Odd
numbered groups are "private", are *not* defined in the standard data dictionary
and can be used by manufacturers as they wish (see below).

Quoting from section 7.1 of `PS 3.5`_:

    Two types of Data Elements are defined:

    -- Standard Data Elements have an even Group Number that is not (0000,eeee),
    (0002,eeee), (0004,eeee), or (0006,eeee).

    Note: Usage of these groups is reserved for DIMSE Commands (see PS 3.7) and
    DICOM File Formats.

    -- Private Data Elements have an odd Group Number that is not (0001,eeee),
    (0003,eeee), (0005,eeee), (0007,eeee), or (FFFF,eeee). Private Data Elements
    are discussed further in Section 7.8.

Private attribute tags
^^^^^^^^^^^^^^^^^^^^^^

Private attribute tags are tags with an odd group number. A private element is
an element with a private tag.

Private elements still use the (Tag, [Value Representation, ] Value Length,
Value Field) DICOM data format.

The same odd group may be used by different manufacturers in different ways.

To try and avoid collisions of private tags from different manufacturers, there
is a mechanism by which a manufacturer can tell other users of a DICOM dataset
that it has reserved a block in the (Group number, Element number) space for
their own use.  To do this they write a "Private Creator" element where the tag
is of the form ``(gggg, 00xx)``, the Value Representation (see below) is "LO"
(Long String) and the Value Field is a string identifying what the space is
reserved for.  Here ``gggg`` is the odd group we are reserving a portion of and
the ``xx`` is the block of elements we are reserving.  A tag of ``(gggg, 00xx)``
reserves the 256 elements in the range ``(gggg, xx00)`` to ``(gggg, xxFF)``.

For example, here is a real data element from a Siemens DICOM dataset::

  (0019, 0010) Private Creator                     LO: 'SIEMENS MR HEADER'

This reserves the tags from ``(0019, 1000)`` to ``(0019, 10FF)`` for information
on the "SIEMENS MR HEADER"

The odd group ``gggg`` must be greater than ``0008`` and the block reservation
``xx`` must be greater than or equal to ``0010`` and less than ``0100``.

Here is the start of the relevant section from PS 3.5:

  7.8.1 PRIVATE DATA ELEMENT TAGS

  It is possible that multiple implementors may define Private Elements with the
  same (odd) group number.  To avoid conflicts, Private Elements shall be
  assigned Private Data Element Tags according to the following rules.

  a) Private Creator Data Elements numbered (gggg,0010-00FF) (gggg is odd) shall
  be used to reserve a block of Elements with Group Number gggg for use by an
  individual implementor.  The implementor shall insert an identification code
  in the first unused (unassigned) Element in this series to reserve a block of
  Private Elements. The VR of the private identification code shall be LO (Long
  String) and the VM shall be equal to 1.

  b) Private Creator Data Element (gggg,0010), is a Type 1 Data Element that
  identifies the implementor reserving element (gggg,1000-10FF), Private Creator
  Data Element (gggg,0011) identifies the implementor reserving elements
  (gggg,1100-11FF), and so on, until Private Creator Data Element (gggg,00FF)
  identifies the implementor reserving elements (gggg,FF00- FFFF).

  c) Encoders of Private Data Elements shall be able to dynamically assign
  private data to any available (unreserved) block(s) within the Private group,
  and specify this assignment through the blocks corresponding Private Creator
  Data Element(s). Decoders of Private Data shall be able to accept reserved
  blocks with a given Private Creator identification code at any position within
  the Private group specified by the blocks corresponding Private Creator Data
  Element.

Value Representation
--------------------

Value Representation is often abbreviated to VR.

The VR is a two byte character string giving the code for the encoding of the
subsequent data in the Value Field (see below).

The VR appears in DICOM data that has "Explicit Value Representation", and is
absent for data with "Implicit Value Representation".  "Implicit Value
Representation" uses the fact that the DICOM data dictionary gives VR values for
each tag in the standard DICOM data dictionary, so the VR value is implied by
the tag value, given the data dictionary.

Most DICOM data uses "Explicit Value Representation" because the DICOM data
dictionary only gives VRs for standard (even group number, not private) data
elements.  Each manufacturer writes their own private data elements, and the VR
of these elements is not defined in the standard, and therefore may not be known
to software not from that manufacturer.

The VR codes have to be one of the values from this table (section 6.2 of DICOM
standard `PS 3.5`_):

+----------------------+---------------------------------+
| Value Representation | Description                     |
+======================+=================================+
| AE                   | Application Entity              |
+----------------------+---------------------------------+
| AS                   | Age String                      |
+----------------------+---------------------------------+
| AT                   | Attribute Tag                   |
+----------------------+---------------------------------+
| CS                   | Code String                     |
+----------------------+---------------------------------+
| DA                   | Date                            |
+----------------------+---------------------------------+
| DS                   | Decimal String                  |
+----------------------+---------------------------------+
| DT                   | Date/Time                       |
+----------------------+---------------------------------+
| FL                   | Floating Point Single (4 bytes) |
+----------------------+---------------------------------+
| FD                   | Floating Point Double (8 bytes) |
+----------------------+---------------------------------+
| IS                   | Integer String                  |
+----------------------+---------------------------------+
| LO                   | Long String                     |
+----------------------+---------------------------------+
| LT                   | Long Text                       |
+----------------------+---------------------------------+
| OB                   | Other Byte                      |
+----------------------+---------------------------------+
| OF                   | Other Float                     |
+----------------------+---------------------------------+
| OW                   | Other Word                      |
+----------------------+---------------------------------+
| PN                   | Person Name                     |
+----------------------+---------------------------------+
| SH                   | Short String                    |
+----------------------+---------------------------------+
| SL                   | Signed Long                     |
+----------------------+---------------------------------+
| SQ                   | Sequence of Items               |
+----------------------+---------------------------------+
| SS                   | Signed Short                    |
+----------------------+---------------------------------+
| ST                   | Short Text                      |
+----------------------+---------------------------------+
| TM                   | Time                            |
+----------------------+---------------------------------+
| UI                   | Unique Identifier               |
+----------------------+---------------------------------+
| UL                   | Unsigned Long                   |
+----------------------+---------------------------------+
| UN                   | Unknown                         |
+----------------------+---------------------------------+
| US                   | Unsigned Short                  |
+----------------------+---------------------------------+
| UT                   | Unlimited Text                  |
+----------------------+---------------------------------+

Value length
------------

Value length gives the length of the data contained in the Value Field tag, or
is a flag specifying the Value Field is of undefined length, and thus must be
terminated later in the data stream with a special Item or Sequence Delimitation
tag.

Quoting from section 7.1.1 of `PS 3.5`_:

    Value Length:  Either:

    a 16 or 32-bit (dependent on VR and whether VR is explicit or implicit)
    unsigned integer containing the Explicit Length of the Value Field as the
    number of bytes (even) that make up the Value. It does not include the
    length of the Data Element Tag, Value Representation, and Value Length
    Fields.

    a 32-bit Length Field set to Undefined Length (FFFFFFFFH). Undefined
    Lengths may be used for Data Elements having the Value Representation
    (VR) Sequence of Items (SQ) and Unknown (UN). For Data Elements with
    Value Representation OW or OB Undefined Length may be used depending
    on the negotiated Transfer Syntax (see Section 10 and Annex A).

Value field
-----------

An even number of bytes storing the value(s) of the data element.  The exact
format of this data depends on the Value Representation (see above) and the
Value Multiplicity (see next section).

Data element tags and data dictionaries
=======================================

We can look up data element tags in a *data dictionary*.

As we've seen, data element tags with even group numbers are *standard* data
element tags.  We can look these up in the standard data dictionary in section 6
of `PS 3.6`_.

Data element tags with odd group numbers are *private* data element tags. These
can be used by manufacturers for information that may be specific to the
manufacturer.  To look up these tags, we need the private data dictionary of the
manufacturer.

A data dictionary lists (Attribute tag, Attribute name, Attribute Keyword, Value
Representation, Value Multiplicity) for all tags.

For example, here is an excerpt from the table in PS 3.6 section 6:

+-------------+------------------------------------------+-------------------------------------+----+----+
| Tag         | Name                                     | Keyword                             | VR | VM |
+=============+==========================================+=====================================+====+====+
| (0010,0010) | Patient's Name                           | PatientName                         | PN | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+
| (0010,0020) | Patient ID                               | PatientID                           | LO | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+
| (0010,0021) | Issuer of Patient ID                     | IssuerOfPatientID                   | LO | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+
| (0010,0022) | Type of Patient ID                       | TypeOfPatientID                     | CS | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+
| (0010,0024) | Issuer of Patient ID Qualifiers Sequence | IssuerOfPatientIDQualifiersSequence | SQ | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+
| (0010,0030) | Patient's Birth Date                     | PatientBirthDate                    | DA | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+
| (0010,0032) | Patient's Birth Time                     | PatientBirthTime                    | TM | 1  |
+-------------+------------------------------------------+-------------------------------------+----+----+

The "Name" column gives a standard name for the tag.  "Keyword" gives a shorter
equivalent to the name without spaces that can be used as a variable or
attribute name in code.

Value Representation in the data dictionary
-------------------------------------------

The "VR" column in the data dictionary gives the Value Representation.  There is
usually only one possible VR for each tag [#can_be_two]_.

If a particular stream of data elements is using "Implicit Value Representation
Encoding" then the data elements consist of (tag, Value Length, Value Field) and
the Value Representation is implicit.  In this case we have to get the Value
Representation from the data dictionary.  If a stream is using "Explicit Value
Representation Encoding", the elements consist of (tag, Value Representation,
Value Length, Value Field) and the Value Representation is therefore already
specified along with the data.

Value Multiplicity in the data dictionary
-----------------------------------------

The "VM" column in the dictionary gives the Value Multiplicity for this tag.
Quoting from PS 3.5 section 6.4:

    The Value Multiplicity of a Data Element specifies the number of Values that
    can be encoded in the Value Field of that Data Element. The VM of each Data
    Element is specified explicitly in PS 3.6. If the number of Values that may
    be encoded in an element is variable, it shall be represented by two numbers
    separated by a dash; e.g., "1-10" means that there may be 1 to 10 Values in
    the element.

The most common values for Value Multiplicity in the standard data dictionary
are (in decreasing frequency) '1', '1-n', '3', '2', '1-2', '4' with other values
being less common.

The data dictionary is the only way to know the Value Multiplicity of a
particular tag.  This means that we need the manufacturer's private data
dictionary to know the Value Multiplicity of private attribute tags.


DICOM data structures
=====================

A data set
----------

A DICOM *data set* is a ordered list of data elements.  The order of the list is
the order of the tags of the data elements.  Here is the definition from section
3.10 of `PS 3.5`_:

    DATA SET: Exchanged information consisting of a structured set of Attribute
    values directly or indirectly related to Information Objects. The value of
    each Attribute in a Data Set is expressed as a Data Element.  A collection
    of Data Elements ordered by increasing Data Element Tag number that is an
    encoding of the values of Attributes of a real world object.

Background - the DICOM world
----------------------------

DICOM has abstract definitions of a set of entities (objects) in the "Real
World".  These real world objects have relationships between them. Section 7 of
`PS 3.3`_ has the title "DICOM model of the real world".  Examples of Real World
entities are Patient, Study, Series.

Here is a selected list of real world entities compiled from section 7 of PS
3.3:

* Patient
* Visit
* Study
* Modality Performed Procedure Steps
* Frame of Reference
* Equipment
* Series
* Registration
* Fiducials
* Image
* Presentation State
* SR Document
* Waveform
* MR Spectroscopy
* Raw Data
* Encapsulated Document
* Real World Value Mapping
* Stereometric Relationship
* Surface
* Measurements

DICOM refers to its model of the entities and their relationships in the real
world as the DICOM Application Model.  PS 3.3:

    3.8.5 DICOM application model: an Entity-Relationship diagram used to model
    the relationships between Real-World Objects which are within the area of
    interest of the DICOM Standard.

DICOM Entities and Information Object Definitions
-------------------------------------------------

This is rather confusing.

PS 3.3 gives definitions of fundamental DICOM objects called *Information Object
Definitions* (IODs).  Here is the definition of an IOD from section 3.8.7 of PS
3.3:

    3.8.7 Information object definition (IOD): a data abstraction of a class of
    similar Real-World Objects which defines the nature and Attributes relevant
    to the class of Real-World Objects represented.

IODs give lists of attributes (data elements) that refer to one or more objects
in the DICOM Real World.

A single IOD is the usual atom of data sent in a single DICOM message.

An IOD that contains attributes (data elements) for only one object in the DICOM
Real World is a *Normalized IOD*. From PS 3.3:

    3.8.10 Normalized IOD: an Information Object Definition which represents a
    single entity in the DICOM Application Model. Such an IOD includes
    Attributes which are only inherent in the Real-World Object that the IOD
    represents.

Annex B of PS 3.3 defines the normalized IODs.

Many DICOM Real World objects do not have corresponding normalized IODs,
presumably because there is no common need to send data only corresponding to -
say - a patient - without also sending related information like - say - an
image.  If you do want to send information relating to a patient with
information relating to an image, you need a *composite IOD*.

An IOD that contains attributes from more than one object in the DICOM Real
World is a *Composite IOD*.  PS 3.3 again:

    3.8.2 Composite IOD: an Information Object Definition which represents parts
    of several entities in the DICOM Application Model. Such an IOD includes
    Attributes which are not inherent in the Real-World Object that the IOD
    represents but rather are inherent in related Real-World Objects

Annex A of PS 3.3 defines the composite IODs.

DICOM MR or CT image IODs are classic examples of composite IODs, because they
contain information not just about the image itself, but also information about
the patient, the study, the series, the frame of reference and the equipment.

The term *Information Entity* (IE) refers to a part of a composite IOD that
relates to a single DICOM Real World object.  PS 3.3:

    3.8.6 Information entity: that portion of information defined by a Composite
    IOD which is related to one specific class of Real-World Object. There is a
    one-to-one correspondence between Information Entities and entities in the
    DICOM Application Model.

IEs are names of DICOM Real World objects that label parts of a composite IOD.
IEs have no intrinsic content, but serve as meaningful labels for a group of
*modules* (see below) that refer to the same Real World object.

Annex A 1.2, PS 3.3 lists all the IEs used in composite IODs.

For example, section A.4 in PD 3.3 defines the composite IOD for an MR Image -
the Magnetic Resonance Image Object Definition. The definition looks like this
(table A.4-1 of PS 3.3)

+--------------------+-------------------------+-----------+---------------------------------------------------------+
| IE                 | Module                  | Reference | Usage                                                   |
+====================+=========================+===========+=========================================================+
| Patient            | Patient                 | C.7.1.1   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Clinical Trial Subject  | C.7.1.3   | U                                                       |
+--------------------+-------------------------+-----------+---------------------------------------------------------+
| Study              | General Study           | C.7.2.1   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Patient Study           | C.7.2.2   | U                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Clinical Trial Study    | C.7.2.3   | U                                                       |
+--------------------+-------------------------+-----------+---------------------------------------------------------+
| Series             | General Series          | C.7.3.1   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Clinical Trial Series   | C.7.3.2   | U                                                       |
+--------------------+-------------------------+-----------+---------------------------------------------------------+
| Frame of Reference | Frame of Reference      | C.7.4.1   | M                                                       |
+--------------------+-------------------------+-----------+---------------------------------------------------------+
| Equipment          | General Equipment       | C.7.5.1   | M                                                       |
+--------------------+-------------------------+-----------+---------------------------------------------------------+
| Image              | General Image           | C.7.6.1   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Image Plane             | C.7.6.2   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Image Pixel             | C.7.6.3   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Contrast/bolus          | C.7.6.4   | C - Required if contrast media was used in this image   |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Device                  | C.7.6.12  | U                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Specimen                | C.7.6.22  | U                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | MR Image                | C.8.3.1   | M                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | Overlay Plane           | C.9.2     | U                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | VOI LUT                 | C.11.2    | U                                                       |
|                    +-------------------------+-----------+---------------------------------------------------------+
|                    | SOP Common              | C.12.1    | M                                                       |
+--------------------+-------------------------+-----------+---------------------------------------------------------+

As you can see, the MR Image IOD is composite and composed of Patient, Study,
Series, Frame of Reference, Equipment and Image IEs.

The *module* heading defines which modules make up the information relevant to
the IE.

A module is a named and defined grouping of attributes (data elements) with
related meaning.  PS 3.3:

    3.8.8 Module: A set of Attributes within an Information Entity or Normalized
    IOD which are logically related to each other.

Grouping attributes into modules simplifies the definition of multiple composite
IODs.  For example, the composite IODs for a CT image and an MR Image both have
modules for Patient, Clinical Trial Subject, etc.

Annex C of PS 3.3 defines all the modules used for the IOD definitions.  For
example, from the table above, we see that the "Patient" module is at section
C.7.1.1 of PS 3.3.  This section gives a table of all the attributes (data
elements) in this module.

The last column in the table above records whether the particular module is
Mandatory, Conditional or User Option (defined in section A 1.3 of PS 3.3)

Lastly module definitions may make use of *Attribute macros*.  Attribute macros
are very much like modules, in that they are a named group of attributes that
often occur together in module definitions, or definitions of other macros.
From PS 3.3:

    3.11.1 Attribute Macro: a set of Attributes that are described in a single
    table that is referenced by multiple Modules or other tables.

For example, here is the Patient Orientation Macro definition table from section
10.12 in PS 3.3:

+------------------------------------------------+-------------------------+--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Attribute Name                                 | Tag                     | Type   | Attribute Description                                                                                                                                                                            |
+================================================+=========================+========+==================================================================================================================================================================================================+
| Patient Orientation Code Sequence              | (0054,0410)             | 1      | Sequence that describes the orientation of the patient with respect to gravity. See C.8.11.5.1.2 for further explanation. Only a single Item shall be included in this Sequence.                 |
+------------------------------------------------+-------------------------+--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| >Include 'Code Sequence Macro' Table 8.8-1.                                       | Baseline Context ID 19                                                                                                                                                                           |
+------------------------------------------------+-------------------------+--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| >Patient Orientation Modifier Code Sequence    | (0054,0412)             | 1C     | Patient orientation modifier. Required if needed to fully specify the orientation of the patient with respect to gravity. Only a single Item shall be included in this Sequence.                 |
+------------------------------------------------+-------------------------+--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| >>Include 'Code Sequence Macro' Table 8.8-1.                                      | Baseline Context ID 20                                                                                                                                                                           |
+------------------------------------------------+-------------------------+--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Patient Gantry Relationship Code Sequence      | (0054,0414)             | 3      | Sequence that describes the orientation of the patient with respect to the head of the table. See Section C.8.4.6.1.3 for further explanation. Only a single Item is permitted in this Sequence. |
+------------------------------------------------+-------------------------+--------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| >Include 'Code Sequence Macro' Table 8.8-1.                                       | Baseline Context ID 21                                                                                                                                                                           |
+-----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

As you can see, this macro specifies some tags that should appear when this
macro is "Included" - and also includes other macros.

DICOM services (DIMSE)
======================

We now go back to messages.

The DICOM application sending the message is called the Service Class User
(SCU). We might also call this the client.

The DICOM application receiving the message is called the Service Class Provider
(SCP).  We might also call this the server - for this particular message.

Quoting from `PS 3.7`_ section 6.3:

    A Message is composed of a Command Set followed by a conditional Data Set
    (see PS 3.5 for the definition of a Data Set). The Command Set is used to
    indicate the operations/notifications to be performed on or with the Data
    Set.

The command set consists of command elements (elements with group number 0000).

Valid sequences of command elements in the command set form valid DICOM Message
Service Elements (DIMSEs).  Sections 9 and 10 of PS 3.7 define the valid DIMSEs.

For example, there is a DIMSE service called "C-ECHO" that requests confirmation
from the responding application that the echo message arrived.

The definition of the DIMSE services specifies, for a particular DIMSE service,
whether the DIMSE commend set should be followed by a data set.

In particular, the data set will be a full Information Object Definition's worth
of data.

Of most interest to us, the "C-STORE" service command set should always be
followed by a data set conforming to an image data IOD.

DICOM service object pairs (SOPs)
=================================

As we've seen, some DIMSE services should be followed by particular types of
data.

For example, the "C-STORE" DIMSE command set should be followed by an IOD of
data to store, but the "C-ECHO" has no data object following.

The association of a particular type of DIMSE (command set) with the associated
IOD's-worth of data is a Service Object Pair. The DIMSE is the "Service" and the
data IOD is the "Object".  Thus the combination of a "C-STORE" DIMSE and an "MR
Image" IOD would be a SOP.  Services that do not have data following are a
particular type of SOP where the Object is null.  For example, the "C-ECHO"
service is the entire contents of a Verification SOP (PS 3.4, section A.4).

DICOM defines which pairings are possible, by listing them all as Service Object
Pair classes (SOP classes).

Usually a SOP class describes the pairing of exactly one DIMSE service with one
defined IOD.  For example, the "MR Image storage" SOP class pairs the "C-STORE"
DIMSE with the "MR Image" IOD.

Sometimes a SOP class describes the pairings of one of several possible DIMSEs
with a particular IOP.  For example, the "Modality Performed Procedure Step" SOP
class describes the pairing of *either* ("N-CREATE", Modality Performed
Procedure Step IOD) *or* ("N-SET", Modality Performed Procedure Step IOD) (see
PS 3.4 F.7.1).  For this reason a SOP class is best described as the pairing of
a *DIMSE service group* with an IOD, where the DIMSE service group usually
contains just one DIMSE service, but sometimes has more. For example, the "MR
Image Storage" SOP class has a DIMSE service group of one element ["C-STORE"].
The "Modality Performed Procedure Step" SOP class has a DIMSE service group with
two elements: ["N-CREATE", "N-SET"].

From PS 3.4:

    6.4 DIMSE SERVICE GROUP

    DIMSE Service Group specifies one or more operations/notifications defined
    in PS 3.7 which are applicable to an IOD.

    DIMSE Service Groups are defined in this Part of the DICOM Standard, in the
    specification of a Service - Object Pair Class.

    6.5 SERVICE-OBJECT PAIR (SOP) CLASS

    A Service-Object Pair (SOP) Class is defined by the union of an IOD and a
    DIMSE Service Group. The SOP Class definition contains the rules and
    semantics which may restrict the use of the services in the DIMSE Service
    Group and/or the Attributes of the IOD.

The Annexes of `PS 3.4`_ define the SOP classes.

A pairing of actual data of form (DIMSE group, IOD) that conforms to the SOP
class definition, is a SOP class instance. That is, the instance comprises the
actual values of the service and data elements being transmitted.

For example, there is a SOP class called "MR Image Storage".  This is the
association of the "C-STORE" DIMSE command with the "MR Image" IOD.  A
particular "C-STORE" request command set along with the particular "MR Image"
IOD data set would be an *instance* of the MR Image SOP class.

DICOM files
===========

Now let us return to the confusing definition of the DICOM file format from
section 7 of PS 3.10:

    7 DICOM File Format

    The DICOM File Format provides a means to encapsulate in a file the Data Set
    representing a SOP Instance related to a DICOM IOD. As shown in Figure 7-1,
    the byte stream of the Data Set is placed into the file after the DICOM File
    Meta Information. Each file contains a single SOP Instance.

The DICOM file Meta Information is:

* File preamble - 128 bytes, content unspecified
* DICOM prefix - 4 bytes "DICM" character string
* 5 meta information elements (group 0002) as defined in table 7.1 of PS 3.10

There follows the IOD dataset part of the SOP instance.  In the case of a file
storing an MR Image, this dataset will be of IOD type "MR Image"

.. rubric:: Footnotes

.. [#can_be_two] Actually, it is not quite true that there can be only one VR
   associated with a particular tag.  A small number of tags have VRs which can
   be either Unsigned Short (US) or Signed Short (SS). An even smaller number
   of tags can be either Other Byte (OB) or Other Word (OW).  For all the
   relevant tags the VM is a set number (1, 3, or 4). So, in the OB / OW cases
   you can tell which of OB or OW you have by the Value Length.  The US / SS
   cases seem to refer to pixel values; presumably they are US if the Pixel
   Representation (tag 0028, 0103) is 0 (for unsigned) and SS if the Pixel
   Representation is 1 (for signed)

.. include:: ../links_names.txt

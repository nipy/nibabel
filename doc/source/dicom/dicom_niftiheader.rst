.. _dicom-niftiheader:

##############################
DICOM Tags in the NIfTI Header
##############################

NIfTI images include an extended header (see the `NIfTI Extensions Standard`_)
to store, amongst others, DICOM tags and attributes. When NiBabel loads a NIfTI
file containing DICOM information (a NIfTI extension with ``ecode == 2``), it
parses it and returns a pydicom dataset as the content of the NIfTI extension.
This can be read and written to in order to facilitate communication with
software that uses specific DICOM codes found in the NIfTI header.

For example, the commercial PMOD software stores the Frame Start and Duration
times of images using the DICOM tags (0055, 1001) and (0055, 1004). Here's an
example of an image created in PMOD with those stored times accessed through
nibabel.

.. code:: python

    >> import nibabel as nib
    >> nim = nib.load('pmod_pet.nii')
    >> dcmext = nim.header.extensions[0]
    >> dcmext
    Nifti1Extension('dicom', '(0054, 1001) Units                               CS: 'Bq/ml'
    (0055, 0010) Private Creator                     LO: 'PMOD_1'
    (0055, 1001) [Frame Start Times Vector]          FD: [0.0, 30.0, 60.0, ..., 13720.0, 14320.0]
    (0055, 1004) [Frame Durations (ms) Vector]       FD: [30000.0, 30000.0, 30000.0,600000.0, 600000.0]'))

+-------------+--------------------------------+---------------------------------------------------------+
| Tag         | Name                           | Value                                                   |
+=============+================================+=========================================================+
| (0054, 1001)| Units                          | CS: 'Bq/ml'                                             |
+-------------+--------------------------------+---------------------------------------------------------+
|(0055, 0010) | Private Creator                | LO: 'PMOD_1'                                            |
+-------------+--------------------------------+---------------------------------------------------------+
|(0055, 1001) | [Frame Start Times Vector]     | FD: [0.0, 30.0, 60.0, ..., 13720.0, 14320.0             |
+-------------+--------------------------------+---------------------------------------------------------+
|(0055, 1004) | [Frame Durations (ms) Vector]  | FD: [30000.0, 30000.0, 30000.0, ..., 600000.0, 600000.0 |
+-------------+--------------------------------+---------------------------------------------------------+

Access each value as you would with pydicom::

    >> ds = dcmext.get_content()
    >> start_times = ds[0x0055, 0x1001].value
    >> durations   = ds[0x0055, 0x1004].value

Creating a PMOD-compatible header is just as easy::

    >> nim = nib.load('pet.nii')
    >> nim.header.extensions
    []
    >> from dicom.dataset import Dataset
    >> ds = Dataset()
    >> ds.add_new((0x0054,0x1001),'CS','Bq/ml')
    >> ds.add_new((0x0055,0x0010),'LO','PMOD_1')
    >> ds.add_new((0x0055,0x1001),'FD',[0.,30.,60.,13720.,14320.])
    >> ds.add_new((0x0055,0x1004),'FD',[30000.,30000.,30000.,600000.,600000.])
    >> dcmext = nib.nifti1.Nifti1DicomExtension(2,ds)  # Use DICOM ecode 2
    >> nim.header.extensions.append(dcmext)
    >> nib.save(nim,'pet_withdcm.nii')

Be careful! Many imaging tools don't maintain information in the extended
header, so it's possible [likely] that this information may be lost during
routine use. You'll have to keep track, and re-write the information if
required.

Optional Dependency Note: If pydicom is not installed, nibabel uses a generic
:class:`nibabel.nifti1.Nifti1Extension` header instead of parsing DICOM data.

.. _`NIfTI Extensions Standard`: http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/extension.html

.. include:: ../links_names.txt

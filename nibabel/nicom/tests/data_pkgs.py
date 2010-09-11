''' Data packages for DICOM testing '''

from ... import data as nibd

PUBLIC_PKG_DEF = dict(
    name = 'nipy-dicom_public',
    version = '0.1')

PRIVATE_PKG_DEF = dict(
    name = 'nipy-dicom_private',
    version = '0.1')


PUBLIC_DS = nibd.datasource_or_bomber(PUBLIC_PKG_DEF)
PRIVATE_DS = nibd.datasource_or_bomber(PRIVATE_PKG_DEF)

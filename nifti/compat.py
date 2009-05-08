#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""The module provides the NiftiImage interface, which is backward-compatible
to the previous C-based implementation.
"""

__docformat__ = 'restructuredtext'

import nifti.nifti1

class NiftiImage(nifti.nifti1.Nifti1Image):
    def __init__(self, source, header=None, loadmeta=False):
        raise NotImplementedError


    def asDict(self):
        raise NotImplementedError


    def updateFromDict(self, hdrdict):
        raise NotImplementedError


    def vx2q(self, coord):
        raise NotImplementedError


    def vx2s(self, coord):
        raise NotImplementedError


    def getVoxDims(self):
        raise NotImplementedError


    def setVoxDims(self, value):
        raise NotImplementedError


    def setPixDims(self, value):
        raise NotImplementedError


    def getPixDims(self):
        raise NotImplementedError


    def getExtent(self):
        raise NotImplementedError


    def getVolumeExtent(self):
        raise NotImplementedError


    def getTimepoints(self):
        raise NotImplementedError


    def getRepetitionTime(self):
        raise NotImplementedError


    def setRepetitionTime(self, value):
        raise NotImplementedError


    def setSlope(self, value):
        raise NotImplementedError


    def setIntercept(self, value):
        raise NotImplementedError


    def setDescription(self, value):
        raise NotImplementedError


    def setXFormCode(self, xform, code):
        raise NotImplementedError


    def setQFormCode(self, code):
        raise NotImplementedError


    def getQFormCode(self, as_string = False):
        raise NotImplementedError


    def getSFormCode(self, as_string = False):
        raise NotImplementedError


    def setSFormCode(self, code):
        raise NotImplementedError


    def getSForm(self):
        raise NotImplementedError


    def setSForm(self, m, code='mni152'):
        raise NotImplementedError


    def getInverseSForm(self):
        raise NotImplementedError


    def getQForm(self):
        raise NotImplementedError


    def getInverseQForm(self):
        raise NotImplementedError


    def setQForm(self, m, code='scanner'):
        raise NotImplementedError


    def setQuaternion(self, value, code='scanner'):
        raise NotImplementedError


    def getQuaternion(self):
        raise NotImplementedError


    def setQOffset(self, value, code='scanner'):
        raise NotImplementedError


    def getQOffset(self):
        raise NotImplementedError


    def setQFac(self, value, code='scanner'):
        raise NotImplementedError


    def getQOrientation(self, as_string = False):
        raise NotImplementedError


    def getSOrientation(self, as_string = False):
        raise NotImplementedError


    def getXYZUnit(self, as_string = False):
        raise NotImplementedError


    def setXYZUnit(self, value):
        raise NotImplementedError


    def getTimeUnit(self, as_string = False):
        raise NotImplementedError


    def setTimeUnit(self, value):
        raise NotImplementedError


    def getFilename(self):
        raise NotImplementedError


    def save(self, filename=None, filetype = 'NIFTI', update_minmax=True):
        raise NotImplementedError


    def copy(self):
        raise NotImplementedError


    def load(self):
        raise NotImplementedError


    def unload(self):
        raise NotImplementedError


    def updateCalMinMax(self):
        raise NotImplementedError


    def updateHeader(self, hdrdict):
        raise NotImplementedError


    def getScaledData(self):
        raise NotImplementedError


    def setDataArray(self, data):
        raise NotImplementedError


    def getDataArray(self):
        raise NotImplementedError


    def asarray(self, copy = True):
        raise NotImplementedError


    def setFilename(self, filename, filetype = 'NIFTI'):
        raise NotImplementedError


    def getFilename(self):
        raise NotImplementedError


    #
    # class properties
    #

    # read only
    data =          property(fget=getDataArray, fset=setDataArray)
    nvox =          property(fget=lambda self: self.__nimg.nvox)
    max =           property(fget=lambda self: self.__nimg.cal_max)
    min =           property(fget=lambda self: self.__nimg.cal_min)
    sform_inv =     property(fget=getInverseSForm)
    qform_inv =     property(fget=getInverseQForm)
    extent =        property(fget=getExtent)
    volextent =     property(fget=getVolumeExtent)
    timepoints =    property(fget=getTimepoints)
    raw_nimg =      property(fget=lambda self: self.__nimg)
    filename =      property(fget=getFilename)

    # read and write
    filename =      property(fget=getFilename, fset=setFilename)
    bbox =          property(fget=imgfx.getBoundingBox, fset=imgfx.crop)

    slope =         property(fget=lambda self: self.__nimg.scl_slope,
                             fset=setSlope)
    intercept =     property(fget=lambda self: self.__nimg.scl_inter,
                             fset=setIntercept)
    voxdim =        property(fget=getVoxDims, fset=setVoxDims)
    pixdim =        property(fget=getPixDims, fset=setPixDims)
    description =   property(fget=lambda self: self.__nimg.descrip,
                             fset=setDescription)
    header =        property(fget=asDict, fset=updateFromDict)
    sform =         property(fget=getSForm, fset=setSForm)
    sform_code =    property(fget=getSFormCode, fset=setSFormCode)
    qform =         property(fget=getQForm, fset=setQForm)
    qform_code =    property(fget=getQFormCode, fset=setQFormCode)
    quatern =       property(fget=getQuaternion, fset=setQuaternion)
    qoffset =       property(fget=getQOffset, fset=setQOffset)
    qfac =          property(fget=lambda self: self.__nimg.qfac, fset=setQFac)
    rtime =         property(fget=getRepetitionTime, fset=setRepetitionTime)
    xyz_unit =      property(fget=getXYZUnit, fset=setXYZUnit)
    time_unit =     property(fget=getTimeUnit, fset=setTimeUnit)

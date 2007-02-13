import clibs
import os
import numpy

def numpydtype2niftidtype(array):
	
	# get the real datatype from numpy type dictionary
	dtype = numpy.typeDict[str(array.dtype)]

	if not DTnumpy2nifti.has_key(dtype):
		raise ValueError, "Unsupported datatype '%s'" % str(array.dtype)

	return DTnumpy2nifti[dtype]


def setFileNamesAndType(niftiptr, base, filetype = 'NIFTI_GZ'):
	if not filetype in nifti_filetype_ids:
		raise ValueError, \
			"Unknown filetype '%s'. Known filetypes are: %s" % (filetype, ' '.join(nifti_filetype_ids))

	#initial names
	fname = base
	iname = base
	
	# append filtype specific parts
	if filetype == 'ANALYZE':
		fname += '.hdr'
		iname += '.img'
		niftiptr.nifti_type = clibs.NIFTI_FTYPE_ANALYZE
	elif filetype == 'NIFTI_PAIR':
		fname += '.hdr'
		iname += '.img'
		niftiptr.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_2
	elif filetype == 'NIFTI':
		fname += '.nii'
		iname = fname
		niftiptr.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_1
	elif filetype == 'NIFTI_GZ':
		fname += '.nii.gz'
		iname = fname
		niftiptr.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_1
	elif filetype == 'ANALYZE_GZ':
		fname += '.hdr.gz'
		iname += '.img.gz'
		niftiptr.nifti_type = clibs.NIFTI_FTYPE_ANALYZE
	elif filetype == 'NIFTI_PAIR_GZ':
		fname += '.hdr.gz'
		iname += '.img.gz'
		niftiptr.nifti_type = clibs.NIFTI_FTYPE_NIFTI1_2
	else:
		raise RuntimeError, "Mismatch between supported filetypes and actually handled filetypes.\nI'm a bug. Please report me."

	# set the names
	niftiptr.fname = fname
	niftiptr.iname = iname



nifti_filetype_ids = [ 'ANALYZE', 'NIFTI', 'NIFTI_PAIR',
           	         'ANALYZE_GZ', 'NIFTI_GZ', 'NIFTI_PAIR_GZ' ]

nifti_unit_ids = [ 'm', 'mm', 'um' ]

DTnumpy2nifti = { numpy.uint8: clibs.NIFTI_TYPE_UINT8,
   	              numpy.int8 : clibs.NIFTI_TYPE_INT8,
		   		  numpy.uint16: clibs.NIFTI_TYPE_UINT16,
				  numpy.int16 : clibs.NIFTI_TYPE_INT16,
				  numpy.uint32: clibs.NIFTI_TYPE_UINT32,
				  numpy.int32 : clibs.NIFTI_TYPE_INT32,
				  numpy.uint64: clibs.NIFTI_TYPE_UINT64,
				  numpy.int64 : clibs.NIFTI_TYPE_INT64,
				  numpy.float32: clibs.NIFTI_TYPE_FLOAT32,
				  numpy.float64: clibs.NIFTI_TYPE_FLOAT64,
				  numpy.complex128: clibs.NIFTI_TYPE_COMPLEX128
				}
	


class NiftiFile(object):
	"""Wrapper class for convenient access to NIfTI data.
	
	The class can either load an image from file or convert a 3d/4d NumPy 
	array into a NIfTI file structure. Either way is automatically determined
	by the type of the 'source' argument (string == filename, array == Numpy).

	One can optionally specify whether the image data should be loaded into 
	memory when opening NIfTI data from files ('load'). When converting a NumPy
	array one can optionally specify the 'voxelsize' (x,y,z) and the repetition 
	time ('tr') as well as the measurment 'unit'. 

	Optional arguments of the respective other mode are ignored.
	"""
	def __init__(self, source, load=False, voxelsize=(1,1,1), tr=1, unit='mm'):
		"""
		"""

		self.fslio = None

		if type( source ) == numpy.ndarray:
			self.__newFromArray( source, voxelsize, tr, unit )
		elif type ( source ) == str:
			self.__newFromFile( source, load )
		else:
			raise ValueError, "Unsupported source type. Only NumPy arrays and filename string are supported."

	def __del__(self):
		self.__close()

	def __close(self):
		"""Close the file and free all unnecessary memory.
		"""
		if self.fslio:
			clibs.FslClose(self.fslio)
			clibs.nifti_image_free(self.fslio.niftiptr)

		self.fslio = clibs.FslInit()

	def __newFromArray(self, data, voxelsize, tr, unit):
		
		if len(data.shape) > 4 or len(data.shape) < 3:
			raise ValueError, "Only 3d or 4d array are supported"

		if not unit in nifti_unit_ids:
			raise ValueError, "Unsupported unit '%s'. Supported units are '%s'" % (unit, ", ".join(nifti_unit_ids))

		# make clean table
		self.__close()

		dim = len(data.shape)

		if dim == 4:
			timesteps = data.shape[-4]
		else:
			timesteps = 1

		# init the data structure
		clibs.FslInitHeader( self.fslio,
							 numpydtype2niftidtype(data),
							 data.shape[-1], data.shape[-2], data.shape[-3],
							 timesteps,
							 voxelsize[0], voxelsize[1], voxelsize[2],
							 tr,
							 dim,
							 unit)

		# allocate memory for image data
		if not clibs.allocateImageMemory(self.fslio.niftiptr):
			raise RuntimeError, "Could not allocate memory for image data."
	
		# assign data
		self.asarray()[:] = data[:]

	def __newFromFile(self, filename, load=False):
		"""Open a NIfTI file.

		If there is already an open file it is closed first. If 'load' is True
		the image data is loaded into memory.
		"""
		self.__close()
		self.fslio = clibs.FslOpen(filename, 'r+')

		if not self.fslio:
			raise RuntimeError, "Error while opening nifti header."

		if load:
			self.load()
	
	def save(self, basename=None, filetype='NIFTI_GZ'):
		"""Save the image to its original file.

		If the image was created using array data (not loaded from a file) one
		has to specify a base filename (and filetype optionally).
		
		If no image data is present in memory, this method does nothing.
		"""
		# saving for the first time?
		if not self.fslio.niftiptr.fname:
			if not basename:
				raise ValueError, "When saving an image for the first time a filename has to be specified."
			
			# set filename
			setFileNamesAndType(self.fslio.niftiptr, basename, filetype)
		
		if self.__haveImageData():
			# and save it
			clibs.nifti_image_write_hdr_img(self.fslio.niftiptr, 1, 'wb')
	
	def saveAs(self, filename, filetype = 'NIFTI_GZ'):
		"""Save the image to a new file.

		By default a compressed NIfTI file is used to save the image.
		Other supported filetypes can be passed to the 'filetype' parameter.

		If there is no image data in memory it is loaded first.
		"""
		if not filetype in nifti_filetype_ids:
			raise ValueError, \
				"Unknown filetype '%s'. Known filetypes are: %s" % (filetype, ' '.join(nifti_filetype_ids))
		
		if not self.__haveImageData():
			self.load()

		# create a temporary fslio structure
		out = clibs.FslInit()

		# copy the header data into the temp structure
		clibs.FslCloneHeader(out, self.fslio)

		# also assign the data blob 
		out.niftiptr.data = self.fslio.niftiptr.data

		# set proper filenames and filetype
		setFileNamesAndType(out.niftiptr, filename, filetype)

		# and save it
		clibs.nifti_image_write_hdr_img(out.niftiptr, 1, 'wb')
		
	def __haveImageData(self):
		"""Returns true if the image data was loaded into memory.
		or False if not.

		See: load(), unload()
		"""
		self.__checkForNiftiImage()

		if self.fslio.niftiptr.data:
			return True
		else:
			return False

	def load(self):
		"""Load the image data into memory.

		It is save to call this method several times.
		"""
		self.__checkForNiftiImage()

		if clibs.nifti_image_load( self.fslio.niftiptr ) < 0:
			raise RuntimeError, "Unable to load image data." 
	
	def unload(self):
		"""Unload image data and free allocated memory.
		"""
		clibs.nifti_image_unload(self.fslio.niftiptr)
		
	def asarray(self, copy = False):
		"""Convert the image data into a multidimensional array.

		Attention: If copy == False (the default) the array only wraps 
		the image data. Any modification done to the array is also done 
		to the image data. If copy is true the array contains a copy
		of the image data.

		Changing the shape or size of the wrapping array is not supported
		and will most likely result in a fatal error.
		"""
		if not self.__haveImageData():
			self.load()

		a = clibs.wrapImageDataWithArray(self.fslio.niftiptr)

		if copy:
			return a.copy()
		else:
			return a

	def __checkForNiftiImage(self):
		"""Check whether a NIfTI image is present.

		Returns True if there is a nifti image file structure or False otherwise.
		One can create a file structure by calling open().
		"""
		if not self.fslio.niftiptr:
			raise RuntimeError, "There is no NIfTI image file structure."

	def dims(self):
		"""Returns a tuple containing the size of the image.

		In case of a 3d image the order of dimensions is (z, y, x). In case of 
		a 4d image the tuple contains (t, z, y, x).

		Only 3d and 4d images are supported. For images containing less than 
		three dimensions the tuple nevertheless contains three values with the
		missing dimensions set to 1.

		The returned value also reflects the structure of the array returned by
		asarray() -- NiftiFile.dims() == NiftiFile.asarray().shape
		"""
		self.__checkForNiftiImage()

		if self.fslio.niftiptr.ndim == 3:
			return (self.fslio.niftiptr.nz,
					self.fslio.niftiptr.ny,
					self.fslio.niftiptr.nx)
		elif self.fslio.niftiptr.ndim == 4:
			return (self.fslio.niftiptr.nt,
					self.fslio.niftiptr.nz,
					self.fslio.niftiptr.ny,
					self.fslio.niftiptr.nx)
		else:
			raise ValueError, "Only 3d or 4d images are supported."

	def nVoxels(self):
		return self.fslio.niftiptr.nvox

	def description(self):
		return self.fslio.niftiptr.descrip

	def setDescription(self, description):
		if len(description) > 79:
			raise ValueError, "The NIfTI format only support descriptions shorter than 80 chars."

		self.fslio.niftiptr.descrip = description
		
	def filenames(self):
		return (self.fslio.niftiptr.fname, self.fslio.niftiptr.iname)

	def datatype(self):
		return clibs.nifti_datatype_string(self.fslio.niftiptr.datatype)

	def voxDims(self):
		"""Returns the dimensions of a single voxel as a tuple (x,y,z).
		"""
		return ( self.fslio.niftiptr.dx,
		         self.fslio.niftiptr.dy,
				 self.fslio.niftiptr.dz
			   )

	def tr(self):
		return self.fslio.niftiptr.dt

	def slope(self):
		return self.fslio.niftiptr.scl_slope
		
	def intercept(self):
		return self.fslio.niftiptr.scl_inter
	
	def q2xyz(self):
		return clibs.mat44ToArray(self.fslio.niftiptr.qto_xyz)

	def q2ijk(self):
		return clibs.mat44ToArray(self.fslio.niftiptr.qto_ijk)

	def s2xyz(self):
		return clibs.mat44ToArray(self.fslio.niftiptr.sto_xyz)

	def s2ijk(self):
		return clibs.mat44ToArray(self.fslio.niftiptr.sto_ijk)




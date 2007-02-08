%define DOCSTRING
"
This is a comprehensive description of this module as a
multiline docstring...
"
%enddef

%module (package="nifti", docstring=DOCSTRING) clibs
%{
#include <nifti1_io.h>
#include <fslio.h>
#include <znzlib.h>

#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* mat44ToArray(mat44* _mat)
{ 
    if (!_mat)
    {
        PyErr_SetString(PyExc_RuntimeError, "Zero pointer passed instead of valid mat44 struct pointer.");
        return(NULL);
    }
       
    int dims[2] = {4,4};
   
    PyObject* array = 0;
    array = PyArray_FromDims ( 2, dims, NPY_FLOAT );
    
    /* mat44 subscription is [row][column] */
    PyArrayObject* a = (PyArrayObject*) array;

    float* data = (float *)a->data;

    int i,j;
    
    for (i = 0; i<4; i+=1)
    {
        for (j = 0; j<4; j+=1)
        {
            data[4*i+j] = _mat->m[i][j];
        }
    }

    return PyArray_Return ( (PyArrayObject*) array  );
}


static PyObject* wrapImageDataWithArray(nifti_image* _img)
{ 
    if (!_img)
    {
        PyErr_SetString(PyExc_RuntimeError, "Zero pointer passed instead of valid nifti_image struct.");
        return(NULL);
    }
       
    if (_img->ndim > 4)
    {
        PyErr_SetString(PyExc_RuntimeError, "Data with more than 4 dimensions is not supported.");
        return(NULL);
    }
       
    if (!_img->data)
    {
        PyErr_SetString(PyExc_RuntimeError, "There is no data in the image (not loaded yet?).");
        return(NULL);
    }

    int* dims = (int*) malloc (sizeof(int) * _img->ndim);
   
    switch (_img->ndim)
    {
        case 3:
            dims[0] = _img->nz;
            dims[1] = _img->ny;
            dims[2] = _img->nx;
            break;
        case 4:
            dims[0] = _img->nt;
            dims[1] = _img->nz;
            dims[2] = _img->ny;
            dims[3] = _img->nx; 
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Only 3d or 4d data is supported.");
            return(NULL);
    }

    int array_type=0;

    switch(_img->datatype)
    {
       case NIFTI_TYPE_UINT8:
           array_type = NPY_UBYTE;
           break;
       case NIFTI_TYPE_INT8:
           array_type = NPY_BYTE;
           break;
       case NIFTI_TYPE_UINT16:
           array_type = NPY_USHORT;
           break;
       case NIFTI_TYPE_INT16:
           array_type = NPY_SHORT;
           break;
       case NIFTI_TYPE_UINT32:
           array_type = NPY_UINT;
           break;
       case NIFTI_TYPE_INT32:
           array_type = NPY_INT;
           break;
       case NIFTI_TYPE_UINT64:
       case NIFTI_TYPE_INT64:
           array_type = NPY_LONG;
           break;
       case NIFTI_TYPE_FLOAT32:
           array_type = NPY_FLOAT;
           break;
       case NIFTI_TYPE_FLOAT64:
           array_type = NPY_DOUBLE;
           break;
       case NIFTI_TYPE_COMPLEX128:
           array_type = NPY_CFLOAT;
           break;
       case NIFTI_TYPE_COMPLEX256:
           array_type = NPY_CDOUBLE;
           break;
       default:
           PyErr_SetString(PyExc_RuntimeError, "Unsupported datatype");
           return(NULL);
     }
     

     PyObject* volarray = 0;
     
     volarray = PyArray_FromDimsAndData ( _img->ndim, dims, array_type, ( char* ) _img->data );

     /*cleanup*/
     free(dims);

     return PyArray_Return ( (PyArrayObject*) volarray  );
}

int allocateImageMemory(nifti_image* _nim)
{
  if (_nim == NULL)
  {
    fprintf(stderr, "NULL pointer passed to allocateImageMemory()");
    return(0);
  }

  if (_nim->data != NULL)
  {
    fprintf(stderr, "There seems to be allocated memory already (valid nim->data pointer found).");
    return(0);
  }

  /* allocate memory */
  _nim->data = (void*) calloc(1,nifti_get_volsize(_nim));

  if (_nim->data == NULL)
  {
    fprintf(stderr, "Failed to allocate %d bytes for image data\n", (int)nifti_get_volsize(_nim));
    return(0);
  }

  return(1);
}

%}


%init 
%{
    import_array();
%}

%include znzlib.h
%include nifti1.h
%include nifti1_io.h
%include fslio.h

static PyObject * wrapImageDataWithArray(nifti_image* _img);
int allocateImageMemory(nifti_image* _nim);
static PyObject* mat44ToArray(mat44* _mat);


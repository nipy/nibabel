import operator

import numpy as np

from .orientations import aff2axcodes


support_np_type = (
    np.int8,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex128)


class OperableImage:
    def _binop(self, val, *, op):
        """Apply operator to Nifti1Image.

        Arithmetic and logical operation on Nifti image.
        Currently support: +, -, *, /, //, &, |
        The nifit image should contain the same header information and affine.
        Images should be the same shape.

        Parameters
        ----------
        op :
            Python operator.
        """
        affine, header = self.affine, self.header
        self_, val_ = _input_validation(self, val)
        # numerical operator should work work

        if op.__name__ in ["add", "sub", "mul", "truediv", "floordiv"]:
            dataobj = op(self_, val_)
        if op.__name__ in ["and_", "or_"]:
            self_ = self_.astype(bool)
            val_ = val_.astype(bool)
            dataobj = op(self_, val_).astype(int)
        return self.__class__(dataobj, affine, header)


    def _unop(self, *, op):
        """
        Parameters
        ----------
        op :
            Python operator.
        """
        _type_check(self)
        if op.__name__ in ["pos", "neg", "abs"]:
            dataobj = op(np.asanyarray(self.dataobj))
        return self.__class__(dataobj, self.affine, self.header)


    def __add__(self, other):
        return self._binop(other, op=operator.__add__)

    def __sub__(self, other):
        return self._binop(other, op=operator.__sub__)

    def __mul__(self, other):
        return self._binop(other, op=operator.__mul__)

    def __truediv__(self, other):
        return self._binop(other, op=operator.__truediv__)

    def __floordiv__(self, other):
        return self._binop(other, op=operator.__floordiv__)

    def __and__(self, other):
        return self._binop(other, op=operator.__and__)

    def __or__(self, other):
        return self._binop(other, op=operator.__or__)

    def __pos__(self):
        return self._unop(op=operator.__pos__)

    def __neg__(self):
        return self._unop(op=operator.__neg__)

    def __abs__(self):
        return self._unop(op=operator.__abs__)



def _input_validation(self, val):
    """Check images orientation, affine, and shape muti-images operation."""
    _type_check(self)
    if isinstance(val, self.__class__):
        _type_check(val)
        # Check orientations are the same
        if aff2axcodes(self.affine) != aff2axcodes(val.affine):
            raise ValueError("Two images should have the same orientation")
        # Check affine
        if (self.affine != val.affine).all():
            raise ValueError("Two images should have the same affine.")
        # Check shape.
        if self.shape[:3] != val.shape[:3]:
            raise ValueError("Two images should have the same shape except"
                                "the time dimension.")

        # if 4th dim exist in a image,
        # reshape the 3d image to ensure valid projection
        ndims = (len(self.shape), len(val.shape))
        if 4 not in ndims:
            self_ = np.asanyarray(self.dataobj)
            val_ = np.asanyarray(val.dataobj)
            return self_, val_

        reference = None
        imgs = []
        for ndim, img in zip(ndims, (self, val)):
            img_ = np.asanyarray(img.dataobj)
            if ndim == 3:
                reference = tuple(list(img.shape) + [1])
                img_ = np.reshape(img_, reference)
            imgs.append(img_)
        return imgs
    else:
        self_ = np.asanyarray(self.dataobj)
        val_ = val
        return self_, val_

def _type_check(*args):
    """Ensure image contains correct nifti data type."""
    # Check types
    dtypes = [img.get_data_dtype().type for img in args]
    # check allowed dtype based on the operator
    if set(support_np_type).union(dtypes) == 0:
        raise ValueError("Image contains illegal datatype for Nifti1Image.")

import numpy as np
import operator


support_np_type = (
    np.int8,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex128)


class OperableImage:
    def _op(self, other, op):
        """Apply operator to Nifti1Image.

        This is a draft and experiment.

        Parameters
        ----------
        op :
            Python operator.
        """
        # Check header... need to have axes in the same places
        if self.header.get_data_shape() != other.header.get_data_shape():
            raise ValueError("Two images should has the same shape.")
        # Check affine
        if (self.affine != other.affine).all():
            raise ValueError("Two images should has the same affine.")
        # Check shape. Handle identical stuff for now.
        if self.shape != other.shape:
            raise ValueError("Two images should has the same shape.")

        # Check types? Problematic types will be caught by numpy,
        # but might be cheaper to check before loading data.
        # collect dtype
        dtypes = [img.get_data_dtype().type for img in (self, other)]
        # check allowed dtype based on the operator
        if set(support_np_type).union(dtypes) == 0:
            raise ValueError("Image contains illegal datatype for arithmatic.")
        if op.__name__ in ["add", "sub", "mul", "div"]:
            dataobj = op(np.asanyarray(self.dataobj), np.asanyarray(other.dataobj))
        if op.__name__ in ["and_", "or_"]:
            self_ = self.dataobj.astype(bool)
            other_ = other.dataobj.astype(bool)
            dataobj = op(self_, other_).astype(int)
        return self.__class__(dataobj, self.affine, self.header)

    def __add__(self, other):
        return self._op(other, operator.__add__)
    def __and__(self, other):
        return self._op(other, operator.__and__)

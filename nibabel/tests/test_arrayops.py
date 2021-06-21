import numpy as np
from .. import Nifti1Image
from numpy.testing import assert_array_equal
import pytest


def test_binary_operations():
    data1 = np.random.rand(5, 5, 2)
    data2 = np.random.rand(5, 5, 2)
    data1[0, 0, :] = 0
    img1 = Nifti1Image(data1, np.eye(4))
    img2 = Nifti1Image(data2, np.eye(4))

    output = img1 + 2
    assert_array_equal(output.dataobj, data1 + 2)

    output = img1 + img2
    assert_array_equal(output.dataobj, data1 + data2)

    output = img1 + img2 + img2
    assert_array_equal(output.dataobj, data1 + data2 + data2)

    output = img1 - img2
    assert_array_equal(output.dataobj, data1 - data2)

    output = img1 * img2
    assert_array_equal(output.dataobj, data1 * data2)

    output = img1 / img2
    assert_array_equal(output.dataobj, data1 / data2)

    output = img1 // img2
    assert_array_equal(output.dataobj, data1 // data2)

    output = img2 / img1
    assert_array_equal(output.dataobj, data2 / data1)

    output = img2 // img1
    assert_array_equal(output.dataobj, data2 // data1)

    output = img1 & img2
    assert_array_equal(output.dataobj, (data1.astype(bool) & data2.astype(bool)).astype(int))

    output = img1 | img2
    assert_array_equal(output.dataobj, (data1.astype(bool) | data2.astype(bool)).astype(int))


def test_binary_operations_4d():
    data1 = np.random.rand(5, 5, 2, 3)
    data2 = np.random.rand(5, 5, 2)
    img1 = Nifti1Image(data1, np.eye(4))
    img2 = Nifti1Image(data2, np.eye(4))
    data2_ = np.reshape(data2, (5, 5, 2, 1))
    output = img1 * img2
    assert_array_equal(output.dataobj, data1 * data2_)


def test_unary_operations():
    data = np.random.rand(5, 5, 2)
    img = Nifti1Image(data, np.eye(4))

    output = -img
    assert_array_equal(output.dataobj, -data)

    output = abs(img)
    assert_array_equal(output.dataobj, abs(data))

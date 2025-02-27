"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]
        return out_grad * b * power(a, b - 1), out_grad * power(a, b) * log(a)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]
        return out_grad / b, -out_grad * a / (b ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is None:
            return a.permute(axes[0:-2] + [axes[-1]] + [axes[-2]])
        else:
            axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
            return a.permute(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        expect_size = 1
        for i in self.shape:
            expect_size *= i
        real_size = 1
        for i in a.shape:
            real_size *= i
        assert expect_size == real_size , "The reshape size is not compatible"
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)



class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def _get_summed_dims(self, old_shape, new_shape):
        if len(old_shape) == 0:
            return tuple(range(len(new_shape)))

        broadcasted = [True for _ in new_shape]
        for old_idx in reversed(range(len(old_shape))):
            new_idx = old_idx + len(new_shape) - len(old_shape)

            if old_shape[old_idx] == new_shape[new_idx]:
                broadcasted[new_idx] = False
                continue

            assert old_shape[old_idx] == 1, "Should never happen"

        return tuple(i for i, is_broadcast in enumerate(broadcasted) if is_broadcast)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]

        if out_grad.shape != a.shape:
            summed_dims = self._get_summed_dims(a.shape, out_grad.shape)
            out_grad = out_grad.sum(summed_dims).reshape(a.shape)

        return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is not None and not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        shape = list(a.shape)
        axes = self.axes
        if axes is None:
            axes = list(range(len(shape)))
        for _ in axes:
            shape[_] = 1
        return broadcast_to(reshape(out_grad, shape), a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):

    def _align_shape(self, x, y):
      if x.shape == y.shape:
        return x

      summed_axes = tuple(range(len(x.shape) - len(y.shape)))
      assert len(summed_axes) > 0
      return x.sum(summed_axes)

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        a_grad = out_grad @ b.transpose()
        b_grad = a.transpose() @ out_grad

        a_grad = self._align_shape(a_grad, a)
        b_grad = self._align_shape(b_grad, b)

        assert a.shape == a_grad.shape
        assert b.shape == b_grad.shape

        return a_grad, b_grad
        ### END YOUR SOLUTION

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        relu_mask = Tensor((node.inputs[0].cached_data > 0), device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * relu_mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return out_grad * (1 - node.inputs[0].tanh()**2)
        # one = Tensor(np.ones_like(node.inputs[0].array))
        return out_grad * (1 - tanh(node.inputs[0])**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # print(f"come into STACK", flush=True)
        ### BEGIN YOUR SOLUTION
        if len(args) > 0:
            shape = args[0].shape
            for arg in args:
                assert arg.shape == shape, "The shape of all tensors should be the same"
            ret_shape = list(shape)
            ret_shape.insert(self.axis, len(args))
            ret = array_api.empty(ret_shape, device=args[0].device)
            for i, arg in enumerate(args):
                slices = [slice(None)] * len(ret_shape)
                slices[self.axis] = i
                ret[tuple(slices)] = arg
            return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    try:
        # import traceback
        # import ipdb
        # ipdb.set_trace()
        # traceback.print_stack()
        # r = 1 / 0
        return Stack(axis)(make_tuple(*args))
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc()
        sys.exit(1)

class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        ret = []
        ret_shape = list(A.shape)
        ret_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            ret.append((A[tuple(slices)]).compact().reshape(ret_shape))
        return tuple(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(axes, list):
            axes = tuple(axes)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        out_shape = list(a.shape)
        for i in self.axes:
            out_shape[i] *= self.dilation + 1
        out = array_api.full(out_shape, 0, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for dim in self.axes:
            slices[dim] = slice(None, None, self.dilation+1)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        out_shape = list(a.shape)
        for i in self.axes:
            out_shape[i] //= self.dilation + 1
        out = array_api.empty(out_shape, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for dim in self.axes:
            slices[dim] = slice(None, None, self.dilation+1)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        assert len(A.shape) == 4, "The input tensor should be 4D"
        assert len(B.shape) == 4, "The kernel tensor should be 4D"
        A = A.compact()
        B = B.compact()
        batch_size, in_height, in_width, in_channel = A.shape
        bs, hs, ws, cs = A.strides
        kernel_height, kernel_width, in_channel, out_channel = B.shape
        
        
        
        pad_A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))).compact()
        batch_size, in_height, in_width, in_channel = pad_A.shape
        bs, hs, ws, cs = pad_A.strides
        receiptive_field_shape = (batch_size, (in_height - kernel_height) // self.stride + 1, (in_width - kernel_width) // self.stride + 1, kernel_height, kernel_width, in_channel)
        receiptive_field_strides = (bs, hs * self.stride, ws * self.stride, hs, ws, cs)
        receiptive_field = pad_A.as_strided(receiptive_field_shape, receiptive_field_strides).compact()
        reveiptive_vector = receiptive_field.reshape((receiptive_field.size //(kernel_height * kernel_width * in_channel), kernel_height * kernel_width * in_channel)).compact()
        kernel_vector = B.reshape((kernel_height * kernel_width * in_channel, out_channel)).compact()
        out = reveiptive_vector @ kernel_vector
        out = out.reshape((batch_size, (in_height - kernel_height) // self.stride + 1, (in_width - kernel_width) // self.stride + 1, out_channel)).compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        s, _, _, _ = W.shape
        
        # 计算X_grad
        W_flipped = flip(W, (0, 1))
        W_flipped_permuted = transpose(W_flipped, (2, 3)) # transpose 只支持两个维度的交换
        outgrad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        X_grad = conv(outgrad_dilated, W_flipped_permuted, padding=s - 1 - self.padding)
        
        # 计算W_grad
        # outgrad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        outgrad_dilated_permuted = permute(outgrad_dilated, (1, 2, 0, 3))
        X_permuted = permute(X, (3, 1, 2, 0))
        W_grad = conv(X_permuted, outgrad_dilated_permuted, padding=self.padding)
        W_grad = permute(W_grad, (1, 2, 0, 3))
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().permute(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        index = [0] * len(self.axes)
        for i in range(len(self.axes)):
            index[self.axes[i]] = i
        return permute(out_grad, tuple(index))
        ### END YOUR SOLUTION
        
def permute(a, axes):
    return Permute(axes)(a)

from typing import Callable, List, NamedTuple, Optional

import numpy
import numpy as np
from scipy import special
import matplotlib
import pytest
import requests
import tqdm
Array = np.ndarray


def ensure_array(array):
    return np.array(array, dtype="float32", copy=False)


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[Array], Array]


class Tensor:
    def __init__(
        self,
        data,
        depends_on: Optional[List[Dependency]] = None,
        requires_grad: bool = False,
    ) -> None:
        self.data = ensure_array(data)
        self.depends_on = depends_on or []
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None

    def __sub__(self, other) -> "Tensor":
        return sub(self, other)

    def __mul__(self, other) -> "Tensor":
        return mul(self, other)

    def __pow__(self, other) -> "Tensor":
        return power(self, other)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def sum(self) -> "Tensor":
        return reduce_sum(self)

    def sigmoid(self) -> "Tensor":
        return sigmoid(self)

    def zero_grad_(self) -> None:
        # TODO
        self.grad = Tensor(np.zeros_like(self.data))

    def tolist(self):
        return self.data.tolist()

    @property
    def shape(self):   # [N, 1], [N, M]
        return self.data.shape

    def backward(self, grad: Optional["Tensor"] = None) -> None:
        if grad is None:
            if np.prod(self.data.shape) == 1:
                grad = Tensor(1)
            else:
                raise RuntimeError

        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))

        #raise NotImplementedError
        # TODO
        # hint: 4 lines of code
        self.grad.data += grad.data
        for depends in self.depends_on:
            grad = depends.grad_fn(self.grad.data)
            depends.tensor.backward(Tensor(grad))


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def reduce_sum(inp: Tensor) -> Tensor:
    # TODO
    data_sum = inp.data.sum()

    requires_grad = inp.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            # TODO
            adj = np.ones_like(inp.data)

            return adj * grad
        depends_on.append(Dependency(tensor=inp, grad_fn=grad_fn))

    return Tensor(data_sum, depends_on=depends_on, requires_grad=requires_grad)


def sub(left: Tensor, right: Tensor) -> Tensor:
    # TODO
    data_sub = np.subtract(left.data, right.data)

    depends_on = []
    if left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            # TODO
            grad_mul = np.ones_like(left.data)

            return grad_mul * grad

        depends_on.append(Dependency(tensor=left, grad_fn=grad_fn_left))

    if right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            # TODO
            grad_mul = np.negative(np.ones_like(right.data))

            return grad_mul * grad

        depends_on.append(Dependency(tensor=right, grad_fn=grad_fn_right))

    requires_grad = left.requires_grad or right.requires_grad

    return Tensor(data=data_sub, depends_on=depends_on, requires_grad=requires_grad)


def mul(left: Tensor, right: Tensor) -> Tensor:
    # TODO
    data = np.multiply(left.data, right.data)

    depends_on = []
    if left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            # TODO
            return right.data * grad

        depends_on.append(Dependency(tensor=left, grad_fn=grad_fn_left))

    if right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            # TODO
            return left.data * grad
        depends_on.append(Dependency(tensor=right, grad_fn=grad_fn_right))

    requires_grad = left.requires_grad or right.requires_grad

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def power(inp: Tensor, exponent: int) -> Tensor:
    # TODO
    data_exp = inp.data ** exponent

    requires_grad = inp.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            # TODO
            adj = exponent * (inp.data ** (exponent - 1))
            return adj * grad

        depends_on.append(Dependency(tensor=inp, grad_fn=grad_fn))

    return Tensor(data=data_exp, depends_on=depends_on, requires_grad=requires_grad)


def sigmoid(inp: Tensor) -> Tensor:
    # TODO
    data_sig = special.expit(inp.data)

    requires_grad = inp.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            grad_mul = data_sig * (1 - data_sig)
            return np.multiply(grad, grad_mul)

        depends_on.append(Dependency(tensor=inp, grad_fn=grad_fn))

    return Tensor(data=data_sig, depends_on=depends_on, requires_grad=requires_grad)


def matmul(left: Tensor, right: Tensor) -> Tensor:
    # TODO
    data_mul = np.matmul(left.data, right.data)

    depends_on = []
    if right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            # TODO
            n = (len(grad.shape) + len(left.data.T.shape) - len(right.shape)) // 2
            return np.tensordot(left.data.T, grad, axes=(tuple(range(-1, -n - 1, -1)), tuple(range(n))))
        depends_on.append(Dependency(tensor=right, grad_fn=grad_fn_right))

    requires_grad = left.requires_grad or right.requires_grad

    return Tensor(data=data_mul, depends_on=depends_on, requires_grad=requires_grad)


class SGD:
    def __init__(self, parameters: list, lr: float = 1e-3) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self):
        self.parameters.data -= self.parameters.grad.data * self.lr

    def zero_grad(self):
        self.parameters.zero_grad_()


def mse_loss(inp: Tensor, target: Tensor) -> Tensor:
    # TODO
    return ((inp - target) ** 2).sum()

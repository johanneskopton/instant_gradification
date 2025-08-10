import numpy as np


class Node:
    def __init__(self):
        self.value = None
        self.grad = 0

    def back(self, error=None):
        self.grad = np.ones_like(self.value)
        self._clipped_back(error)

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def __sub__(self, other):
        return Sum(self, Product(V(-1), other))

    def __truediv__(self, other):
        return Product(self, Inv(other))

    def __matmul__(self, other):
        return MatProduct(self, other)

    def __str__(self):
        return f"Value({self.value})"

    def _clipped_back(self, error):
        self.grad = np.clip(self.grad, -1, 1)
        self._back(error)

    def _back(self, error):
        _ = error
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


class V(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self):
        return self.value

    def _back(self, error):
        self.value -= self.grad * error


class C(V):
    def _back(self, error):
        _ = error
        pass


class Sum(Node):
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b
        super().__init__()

    def __str__(self):
        return f"Sum({self.a}, {self.b})"

    def eval(self):
        self.value = self.a.eval() + self.b.eval()
        return self.value

    def _back(self, error):
        self.a.grad += self.grad
        self.b.grad += self.grad
        self.a._clipped_back(error)
        self.b._clipped_back(error)


class Product(Node):
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b
        super().__init__()

    def __str__(self):
        return f"Product({self.a}, {self.b})"

    def eval(self):
        self.value = self.a.eval() * self.b.eval()
        return self.value

    def _back(self, error):
        self.a.grad += self.grad * self.b.eval()
        self.b.grad += self.grad * self.a.eval()
        self.a._clipped_back(error)
        self.b._clipped_back(error)


class MatProduct(Node):
    def __init__(self, a: Node, b: Node):
        self.a = a  # Matrix
        self.b = b  # Vector
        super().__init__()

    def __str__(self):
        return f"MatProduct({self.a}, {self.b})"

    def eval(self):
        self.value = self.a.eval() @ self.b.eval()
        return self.value

    def _back(self, error):
        self.a.grad += np.expand_dims(self.grad, axis=1) * self.b.eval()
        self.b.grad += self.a.eval().T @ self.grad
        self.a._clipped_back(error)
        self.b._clipped_back(error)


class ReLU(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"Product({self.input})"

    def eval(self):
        self.value = np.maximum(self.input.eval(), 0)
        return self.value

    def _back(self, error):
        self.input.grad += (
            self.grad * np.minimum(np.maximum(self.input.eval(), 0), 0.001) / 0.001
        )
        self.input._clipped_back(error)


class Exp(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"Exp({self.input})"

    def eval(self):
        self.value = np.exp(self.input.eval())
        return self.value

    def _back(self, error):
        self.input.grad += self.grad * np.exp(self.input.eval())
        self.input._clipped_back(error)


class Inv(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"Inv({self.input})"

    def eval(self):
        self.value = 1 / self.input.eval()
        return self.value

    def _back(self, error):
        self.input.grad += self.grad * -1 / np.square(self.input.eval())
        self.input._clipped_back(error)


class ElSum(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"ElSum({self.input})"

    def eval(self):
        self.value = np.sum(self.input.eval())
        return self.value

    def _back(self, error):
        self.input.grad += self.grad
        self.input._clipped_back(error)


def sigmoid(x):
    return Inv(C(1) + Exp(C(-1) * x))

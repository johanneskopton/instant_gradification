import numpy as np
import torch


class Node:
    def __init__(self):
        self.value = None
        self.grad = 0

    def back(self):
        self.grad = np.ones_like(self.value)
        self._back()

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


class V(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self):
        return self.value

    def _back(self):
        pass


class C(V):
    def back(self):
        pass

    def _back(self):
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

    def _back(self):
        self.a.grad += self.grad
        self.b.grad += self.grad
        self.a._back()
        self.b._back()


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

    def _back(self):
        self.a.grad += self.grad * self.b.eval()
        self.b.grad += self.grad * self.a.eval()
        self.a._back()
        self.b._back()


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

    def _back(self):
        self.a.grad += np.expand_dims(self.grad, axis=1) * self.b.eval()
        self.b.grad += self.a.eval().T @ self.grad
        self.a._back()
        self.b._back()


class ReLU(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"Product({self.input})"

    def eval(self):
        self.value = np.maximum(self.input.eval(), 0)
        return self.value

    def _back(self):
        self.input.grad += (
            self.grad * np.minimum(np.maximum(self.input.eval(), 0), 0.001) / 0.001
        )
        self.input._back()


class Exp(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"Exp({self.input})"

    def eval(self):
        self.value = np.exp(self.input.eval())
        return self.value

    def _back(self):
        self.input.grad += self.grad * np.exp(self.input.eval())
        self.input._back()


class Inv(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"Inv({self.input})"

    def eval(self):
        self.value = 1 / self.input.eval()
        return self.value

    def _back(self):
        self.input.grad += self.grad * -1 / np.square(self.input.eval())
        self.input._back()


class ElSum(Node):
    def __init__(self, input: Node):
        self.input = input
        super().__init__()

    def __str__(self):
        return f"ElSum({self.input})"

    def eval(self):
        self.value = np.sum(self.input.eval())
        return self.value

    def _back(self):
        self.input.grad += self.grad
        self.input._back()


def sigmoid(x):
    return Inv(C(1) + Exp(C(-1) * x))


aa = np.array(
    [[6, -4, 0, 2, -8], [-8, 8, -4, 3, 5], [-2, 1, 4, -5, 7], [1, 0, -7, 4, 5]]
)
aa2 = np.array([[2, -2, 0, 2], [7, 0, -3, -5], [-7, 9, 2, -2], [-1, -9, 2, 3]])

bb = np.array([1, 2, -3, -4, 5])

A = V(aa)
A2 = V(aa2)
b = V(bb)
c = ReLU(A @ b)
c2 = sigmoid(A2 @ c)

c2.eval()

print(c2.value)

c2.back()

print(A.grad)
print(b.grad)


AA = torch.tensor(aa, dtype=float, requires_grad=True)
AA2 = torch.tensor(aa2, dtype=float, requires_grad=True)
BB = torch.tensor(bb, dtype=float, requires_grad=True)
CC = torch.nn.ReLU()(AA @ BB)
CC2 = torch.nn.Sigmoid()(AA2 @ CC)

CC2.sum().backward()

print(CC2)
print(AA.grad)
print(BB.grad)

import numpy as np


class Node:
    def __init__(self):
        self.value = None
        self.grad = 0
        self.children = []

    def back(self, error=None):
        self.grad = np.ones_like(self.value)
        self._back(error)

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

    def _back(self, error):
        # self.grad = np.clip(self.grad, -100, 100)
        self._apply_grad(error)
        for child in self.children:
            child._back(error)

    def _apply_grad(self, error):
        _ = error
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def reset_grad(self):
        self.grad = 0
        for child in self.children:
            child.reset_grad()


class V(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self):
        return self.value

    def _apply_grad(self, error):
        if error is not None:
            self.value -= self.grad * error


class C(V):
    def _back(self, error):
        _ = error
        pass


class Sum(Node):
    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.children = [a, b]

    def __str__(self):
        return f"Sum({self.children[0]}, {self.children[1]})"

    def eval(self):
        self.value = self.children[0].eval() + self.children[1].eval()
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += self.grad
        self.children[1].grad += self.grad


class Product(Node):
    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.children = [a, b]

    def __str__(self):
        return f"Product({self.children[0]}, {self.children[1]})"

    def eval(self):
        self.value = self.children[0].eval() * self.children[1].eval()
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += self.grad * self.children[1].eval()
        self.children[1].grad += self.grad * self.children[0].eval()


class MatProduct(Node):
    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.children = [a, b]

    def __str__(self):
        return f"MatProduct({self.children[0]}, {self.children[1]})"

    def eval(self):
        self.value = self.children[0].eval() @ self.children[1].eval()
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += (
            np.expand_dims(self.grad, axis=1) * self.children[1].eval()
        )
        self.children[1].grad += self.children[0].eval().T @ self.grad


class ReLU(Node):
    def __init__(self, input: Node):
        super().__init__()
        self.children = [input]

    def __str__(self):
        return f"Product({self.children[0]})"

    def eval(self):
        self.value = np.maximum(self.children[0].eval(), 0)
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += (
            self.grad
            * np.minimum(np.maximum(self.children[0].eval(), 0), 0.001)
            / 0.001
        )


class Exp(Node):
    def __init__(self, input: Node):
        super().__init__()
        self.children = [input]

    def __str__(self):
        return f"Exp({self.children[0]})"

    def eval(self):
        self.value = np.exp(self.children[0].eval())
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += self.grad * np.exp(self.children[0].eval())


class Inv(Node):
    def __init__(self, input: Node):
        super().__init__()
        self.children = [input]

    def __str__(self):
        return f"Inv({self.children[0]})"

    def eval(self):
        self.value = 1 / self.children[0].eval()
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += self.grad * -1 / np.square(self.children[0].eval())


class ElSum(Node):
    def __init__(self, input: Node):
        super().__init__()
        self.children = [input]

    def __str__(self):
        return f"ElSum({self.children[0]})"

    def eval(self):
        self.value = np.sum(self.children[0].eval())
        return self.value

    def _apply_grad(self, error):
        self.children[0].grad += self.grad


def sigmoid(x):
    return Inv(C(1) + Exp(C(-1) * x))

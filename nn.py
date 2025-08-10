from autograd import V, sigmoid, ReLU, C
import numpy as np
import copy


class Layer:
    def __init__(self, in_size, out_size, activation):
        self.W = V(np.random.normal(0, 0.3, [out_size, in_size]))
        self.b = V(np.zeros(out_size))
        self.activation = activation

    def forward(self, input):
        return self.activation(self.W @ input + self.b)


class NN:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                activation = sigmoid
            else:
                activation = sigmoid
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))

    def forward(self, input):
        result = copy.deepcopy(input)
        for layer in self.layers:
            result = layer.forward(result)
        return result


def MSE(errors):
    return np.mean(np.square(errors))

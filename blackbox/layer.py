import numpy as np


class Layer:
    def __init__(self) -> None:
        self.inputs = None
        self.outputs = None

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagation(self, derivative_outputs: np.ndarray,
                                   learning_rate: float) -> np.ndarray:

        raise NotImplementedError


class FullyConnected(Layer):
    def __init__(self, size_inputs: int, size_outputs: int) -> None:
        super().__init__()

        self.size_inputs = size_inputs
        self.size_outputs = size_outputs

        self.bias = np.random.rand(1, self.size_outputs) - 0.5
        self.weights = np.random.rand(size_inputs, size_outputs) - 0.5

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.bias

        return self.outputs

    def backward_propagation(self, derivative_outputs: np.ndarray,
                                   learning_rate: float) -> np.ndarray:

        derivative_inputs = np.dot(derivative_outputs, self.weights.T)
        derivative_weights = np.dot(self.inputs.T, derivative_outputs)

        # Update parameters by gradient descent
        self.bias -= learning_rate * derivative_outputs
        self.weights -= learning_rate * derivative_weights

        return derivative_inputs

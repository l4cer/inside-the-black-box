import numpy as np

from typing import Any, Tuple


class Layer:
    def __init__(self) -> None:
        self.inputs = None
        self.outputs = None

    def init_randomly(self) -> None:
        raise NotImplementedError

    def dimension_compatibility(self, shape_inputs: Any) -> Tuple[bool, Any]:
        raise NotImplementedError

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagation(self, derivative_outputs: np.ndarray,
                                   learning_rate: float) -> np.ndarray:

        raise NotImplementedError


class FullyConnected(Layer):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.init_randomly()

    def init_randomly(self) -> None:
        self.bias = np.random.rand(1, self.num_outputs) - 0.5
        self.weights = np.random.rand(self.num_inputs, self.num_outputs) - 0.5

    def dimension_compatibility(self, shape_inputs: Any) -> Tuple[bool, Any]:
        if shape_inputs[1] != np.shape(self.weights)[0]:
            return False, None

        if shape_inputs[0] != np.shape(self.bias)[0]:
            return False, None

        return True, np.shape(self.bias)

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = self.inputs @ self.weights + self.bias

        return self.outputs

    def backward_propagation(self, derivative_outputs: np.ndarray,
                                   learning_rate: float) -> np.ndarray:

        derivative_inputs = derivative_outputs @ self.weights.T
        derivative_weights = self.inputs.T @ derivative_outputs

        # Update parameters by gradient descent
        self.bias -= learning_rate * derivative_outputs
        self.weights -= learning_rate * derivative_weights

        return derivative_inputs

    def __repr__(self) -> str:
        return f"FullyConnected({self.num_inputs}, {self.num_outputs})"

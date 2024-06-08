import numpy as np

from blackbox.layer import Layer


class Activation(Layer):
    def __init__(self) -> None:
        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = self.activation_func(self.inputs)

        return self.outputs

    def backward_propagation(self, derivative_outputs: np.ndarray,
                                   learning_rate: float) -> np.ndarray:

        return self.activation_prime(self.inputs) * derivative_outputs


class ReLuActivation(Activation):
    def __init__(self) -> None:
        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0.0) * 1.0


class TanhActivation(Activation):
    def __init__(self) -> None:
        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(np.tanh(inputs), 2)


class SigmoidActivation(Activation):
    def __init__(self) -> None:
        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-inputs))

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs)

        return exp / np.power(1.0 - exp, 2)

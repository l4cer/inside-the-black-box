import numpy as np

from typing import Tuple, Union

from blackbox.layer import Layer

from blackbox.optimizer import Optimizer


class Activation(Layer):
    """
    Base class for all activation layers in a neural network.

    Attributes
    ----------
    inputs : np.ndarray
        Stores the input tensor received by the layer in forward propagation.
    outputs : np.ndarray
        Stores the output tensor generated by the layer in forward propagation.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the Activation class.
        """

        super().__init__()

    def dimension_compatibility(self, shape_inputs: Tuple[int, ...]) -> Union[Tuple[int, ...], None]:
        return shape_inputs

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        """
        Applies the activation function to the inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Layer input tensor.

        Returns
        -------
        outputs : np.ndarray
            Output tensor where each element is the result of applying the
            activation function to the corresponding element of the input tensor.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by subclasses.
        """

        raise NotImplementedError

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function.

        Parameters
        ----------
        inputs : np.ndarray
            Layer input tensor.

        Returns
        -------
        np.ndarray
            Output tensor where each element is the derivative of the activation
            function applied to the corresponding element of the input tensor.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by subclasses.
        """

        raise NotImplementedError

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = self.activation_func(self.inputs)

        return self.outputs

    def backward_propagation(self, gradient_outputs: np.ndarray,
                                   optimizer: Optimizer) -> np.ndarray:

        # Optimizer is not used because there is no "learnable" parameters
        return self.activation_prime(self.inputs) * gradient_outputs


class ReLU(Activation):
    """
    A ReLU (Rectified Linear Unit) activation layer in a neural network.

    The ReLU function is defined as f(x) = max(0, x).

    Attributes
    ----------
    inputs : np.ndarray
        Stores the input tensor received by the layer in forward propagation.
    outputs : np.ndarray
        Stores the output tensor generated by the layer in forward propagation.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the ReLUActivation class.
        """

        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 * (inputs > 0.0)


class Tanh(Activation):
    """
    A hyperbolic tangent activation layer in a neural network.

    The hyperbolic tangent function is defined as
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).

    Attributes
    ----------
    inputs : np.ndarray
        Stores the input tensor received by the layer in forward propagation.
    outputs : np.ndarray
        Stores the output tensor generated by the layer in forward propagation.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the TanhActivation class.
        """

        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(np.tanh(inputs), 2)


class Sigmoid(Activation):
    """
    A sigmoid activation layer in a neural network.

    The sigmoid function is defined as f(x) = 1 / (1 + exp(-x)).

    Attributes
    ----------
    inputs : np.ndarray
        Stores the input tensor received by the layer in forward propagation.
    outputs : np.ndarray
        Stores the output tensor generated by the layer in forward propagation.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the SigmoidActivation class.
        """

        super().__init__()

    def activation_func(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-inputs))

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs)

        return exp / np.power(1.0 - exp, 2)

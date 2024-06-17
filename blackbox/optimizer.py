import numpy as np


class Optimizer:
    """
    Base class for all iterative optimizers.

    Attributes
    ----------
    temp_data : Dict[int, Any]
        Associates a set of temporary data with each identifier,
        generalizing the object and enabling the implementation
        of algorithms that use terms other than the gradient.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the Optimizer class.
        """

        self.temp_data = {}

    def update(self, params: np.ndarray, gradient: np.ndarray) -> None:
        """
        Iteratively updates the learnable parameters in place.

        Parameters
        ----------
        params : np.ndarray
            The learnable parameters to be updated in place.
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by subclasses.
        """

        raise NotImplementedError


class GradientDescent(Optimizer):
    """
    Gradient descent optimizer with optional momentum and dampening.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate (default is 0.001).
    momentum : float, optional
        The momentum factor (default is 0).
    dampening : float, optional
        The dampening for momentum (default is 0).

    Attributes
    ----------
    learning_rate : float
        The learning rate.
    momentum : float
        The momentum factor.
    dampening : float
        The dampening for momentum.
    temp_data : Dict[int, Any]
        Associates a set of temporary data with each identifier,
        generalizing the object and enabling the implementation
        of algorithms that use terms other than the gradient.
    """

    def __init__(self, learning_rate: float = 0.001,
                       momentum: float = 0,
                       dampening: float = 0) -> None:
        """
        Initializes a new instance of the GradientDescent optimizer.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate (default is 0.001).
        momentum : float, optional
            The momentum factor (default is 0).
        dampening : float, optional
            The dampening for momentum (default is 0).
        """

        super().__init__()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening

    def update(self, params: np.ndarray, gradient: np.ndarray) -> None:
        identifier = id(params)

        try:
            # Unpacks already initialized temporary data
            prev_grad = self.temp_data[identifier]
            gradient = self.momentum * prev_grad + (1 - self.dampening) * gradient

        except KeyError:
            # Uninitialized temporary data for this identifier
            pass

        self.temp_data[identifier] = gradient

        params -= self.learning_rate * gradient

import numpy as np

from typing import Callable

from blackbox.layer import Layer


class Network:
    def __init__(self) -> None:
        self.layers = []

        self.loss_func = None
        self.loss_prime = None

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def use(self, loss_func: Callable[[np.ndarray], np.ndarray],
                  loss_prime: Callable[[np.ndarray], np.ndarray]) -> None:

        self.loss_func = loss_func
        self.loss_prime = loss_prime

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output

    def train(self, inputs_train: np.ndarray,
                    outputs_train: np.ndarray,
                    epochs: int,
                    learning_rate: float) -> None:

        for epoch in range(epochs):
            loss = 0.0

            for inputs, outputs in zip(inputs_train, outputs_train):
                prediction = self.predict(inputs)

                loss += self.loss_func(outputs, prediction)
                derivative = self.loss_prime(outputs, prediction)

                for layer in reversed(self.layers):
                    derivative = layer.backward_propagation(derivative, learning_rate)

            loss = loss / len(inputs_train)   # Mean loss

            print(f"Epoch {epoch+1}/{epochs}   loss={loss:.6f}")

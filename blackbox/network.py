import numpy as np

from typing import Any, Dict, Tuple, Callable, Generator

from blackbox.layer import Layer

from blackbox.dataset import Dataset


class DimensionMismatch(Exception):
    pass


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

    def dimension_compatibility(self, shape_inputs: Tuple[int, ...]) -> Tuple[int, ...]:
        for index, layer in enumerate(self.layers):
            shape_inputs = layer.dimension_compatibility(shape_inputs)

            if shape_inputs is None:
                raise DimensionMismatch(
                    f"{layer} at layer {index} is not " +
                    f"consistent with the previous layer")

        return shape_inputs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output

    def train(self, dataset: Dataset,
                    epochs: int,
                    learning_rate: float) -> Generator[Dict[str, Any], None, None]:

        for epoch in range(epochs):
            info = {
                "epoch": epoch,
                "epochs": epochs,
                "loss_test": 0.0,
                "loss_train": 0.0
            }

            # Network training
            for inputs, outputs in dataset.get_train_data():
                shape_outputs = self.dimension_compatibility(np.shape(inputs))

                if shape_outputs != np.shape(outputs):
                    raise DimensionMismatch(
                        f"network output {shape_outputs} and training output " +
                        f"{np.shape(outputs)} have different dimensions")

                prediction = self.predict(inputs)

                info["loss_train"] += self.loss_func(outputs, prediction)
                gradient = self.loss_prime(outputs, prediction)

                for layer in reversed(self.layers):
                    gradient = layer.backward_propagation(gradient, learning_rate)

            info["loss_train"] = info["loss_train"] / dataset.size_train

            # Network testing
            for inputs, outputs in dataset.get_test_data():
                shape_outputs = self.dimension_compatibility(np.shape(inputs))

                if shape_outputs != np.shape(outputs):
                    raise DimensionMismatch(
                        f"network output {shape_outputs} and testing output " +
                        f"{np.shape(outputs)} have different dimensions")

                prediction = self.predict(inputs)

                info["loss_test"] += self.loss_func(outputs, prediction)

            info["loss_test"] = info["loss_test"] / dataset.size_train

            yield info

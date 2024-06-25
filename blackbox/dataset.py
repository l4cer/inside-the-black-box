import numpy as np

from typing import Any, Generator, Tuple


class Dataset:
    def __init__(self) -> None:
        self.test = {"inputs": None, "outputs": None}
        self.train = {"inputs": None, "outputs": None}

    @property
    def size_test(self) -> int:
        return len(self.test["inputs"])

    @property
    def size_train(self) -> int:
        return len(self.train["inputs"])

    def get_test_data(self) -> Generator[Tuple[Any, Any], None, None]:
        dataset = self.test

        for inputs, outputs in zip(dataset["inputs"], dataset["outputs"]):
            yield inputs, outputs

    def get_train_data(self) -> Generator[Tuple[Any, Any], None, None]:
        dataset = self.train

        permutation = np.random.permutation(self.size_train)

        dataset["inputs"] = dataset["inputs"][permutation]
        dataset["outputs"] = dataset["outputs"][permutation]

        for inputs, outputs in zip(dataset["inputs"], dataset["outputs"]):
            yield inputs, outputs

import numpy as np

from blackbox.core import *


def loss_func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.power(y_true - y_pred, 2))


def loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / len(y_true)


def save() -> None:
    net = Network()

    net.add(FullyConnected(2, 3))
    net.add(TanhActivation())
    net.add(FullyConnected(3, 1))
    net.add(TanhActivation())

    net.use(loss_func, loss_prime)
    
    dataset = Dataset()

    dataset.test = {
        "inputs": np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]),
        "outputs": np.array([[[0]], [[1]], [[1]], [[0]]])
    }

    dataset.train = {
        "inputs": np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]),
        "outputs": np.array([[[0]], [[1]], [[1]], [[0]]])
    }

    for info in net.train(dataset, epochs=1000, learning_rate=0.1):
        epoch, epochs, loss_test, loss_train = info.values()

        print(f"Epoch {epoch+1}/{epochs}", end="   ")
        print(f"loss_train={loss_train:.6f}", end="   ")
        print(f"loss_test={loss_test:.6f}")

    save_network(net, "xor.pickle")


def load() -> None:
    net = load_network("xor.pickle")

    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    for x, y in zip(x_train, y_train):
        print(f"true: {y}   pred: {net.predict(x)}")


if __name__ == "__main__":
    load()

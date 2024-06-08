import numpy as np

from blackbox.layer import FullyConnected

from blackbox.activation import TanhActivation

from blackbox.network import Network


def loss_func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.power(y_true - y_pred, 2))


def loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / len(y_true)


def main() -> None:
    net = Network()

    net.add(FullyConnected(2, 3))
    net.add(TanhActivation())
    net.add(FullyConnected(3, 1))
    net.add(TanhActivation())

    net.use(loss_func, loss_prime)

    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    net.train(x_train, y_train, epochs=1000, learning_rate=0.1)

    for x, y in zip(x_train, y_train):
        print(f"true: {y}   pred: {net.predict(x)}")


if __name__ == "__main__":
    main()

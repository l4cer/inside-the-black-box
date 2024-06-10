import pickle

from blackbox.network import Network


def save_network(net: Network, filename: str) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_network(filename: str) -> Network:
    with open(filename, "rb") as handle:
        return pickle.load(handle)

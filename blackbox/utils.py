import pickle

from blackbox.network import Network


def save_network(net: Network, filename: str) -> None:
    """
    Saves the neural network to a pickle file.

    This function serializes the given network object
    and writes it to a file using the pickle module.

    Parameters
    ----------
    net : Network
        The neural network to be saved.
    filename : str
        The name of the pickle file where the network will be saved.
    """

    with open(filename, "wb") as handle:
        pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_network(filename: str) -> Network:
    """
    Loads a neural network from a pickle file.

    This function reads a serialized network object from a file
    using the pickle module and returns the Python net object.

    Parameters
    ----------
    filename : str
        The name of the pickle file where the network will be loaded.

    Returns
    -------
    net : Network
        The deserialized neural network object.
    """

    with open(filename, "rb") as handle:
        return pickle.load(handle)

import pickle

def Unpickle(file):
    """Load CIFAR-10 batch file."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict
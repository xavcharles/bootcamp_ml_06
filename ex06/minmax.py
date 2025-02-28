import numpy as np

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    if x.shape == (1, 1):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        return (x.flatten() - np.min(x.flatten())) / (np.max(x.flatten()) - np.min(x.flatten()))
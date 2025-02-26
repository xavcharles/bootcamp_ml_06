import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    col = np.array([1 for _ in range(len(x))])
    if (len(x.shape) == 1):
        res = np.empty((x.shape[0], 2))
        for i in range(len(x)):
            res[i] = [1, x[i]]
    elif (len(x.shape) == 2):
        res = np.empty((x.shape[0], x.shape[1] + 1))
        for i in range(x.shape[0]):
            res[i] = np.concatenate(([1], x[i]))
    return res

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a two-dimensional array of shape m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    y = add_intercept(x)
    return np.dot(y, theta)

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    nabla0 = np.sum(y_hat - y) / y.shape[0]
    nabla1 = np.dot((y_hat - y).flatten(), x.flatten()) / y.shape[0]
    return np.array(([[nabla0],[nabla1]]))
    
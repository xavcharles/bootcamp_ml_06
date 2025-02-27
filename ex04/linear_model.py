import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def add_intercept(self, x):
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

    def predict_(self, x):
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
        x_bis = self.add_intercept(x)
        y_hat = np.dot(x_bis, self.thetas)
        return y_hat

    def simple_gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.arrays, without any for loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be a numpy.array, a vector of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        x_prime = self.add_intercept(x)
        y_hat = self.predict_(x)
        return np.dot(x_prime.T, y_hat - y) / y.shape[0]

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        for i in range(self.max_iter):
            nabla = self.simple_gradient(x, y)
            self.thetas = self.thetas - self.alpha * nabla

    def loss_elem_(self, y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
        J_elem: numpy.array, a array of dimension (number of the training examples, 1).
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        J_elem = np.array([pow(y_hat[i] - y[i], 2) for i in range(y.shape[0])])
        return J_elem

    def mse_(self, y, y_hat):
        """
        Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
        Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        return (np.dot((y_hat - y).flatten(), (y_hat - y).flatten()) / (y.shape[0]))
    
    def plot_hypothesis(self, x, y):
        """Plot the data and prediction line from three non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, one-dimensional array of size m.
        y: has to be an numpy.ndarray, one-dimensional array of size m.
        theta: has to be an numpy.ndarray, one-dimensional array of size 2.
        Returns:
        Nothing.
        Raises:
        This function should not raise any Exception.
        """
        y_hat = self.predict_(x)
        # print(y_hat)
        plt.plot(x, y, label="Strue(pills)", color='blue', linestyle="None", marker="o", markersize=4)
        plt.plot(x, y_hat, label="Spredict(pills)", color='green', linewidth=1, linestyle='--', marker='x', markersize=4)
        # plt.title(f"Cost : {self.mse_(y, y_hat)}")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_loss_functions(self, x, y):
        # xx = np.linspace(self.thetas[1] - 2.3, self.thetas[1] + 2.3, 100)
        thetas = []
        mse = []
        max_theta = self.thetas[1][0] + 2.3
        self.thetas[1] = self.thetas[1] - 2.3
        while self.thetas[1][0] < max_theta:
            y_hat = self.predict_(x)
            mse.append(self.mse_(y, y_hat))
            self.thetas[1] += 0.046
            print(self.thetas[1])
            thetas.append(self.thetas[1][0])
            print(thetas)
        mse = np.array(mse)
        print(thetas)
        thetas = np.array(thetas)
        print(thetas)
        # y_hats = np.array()
        # mses = np.array([self.mse_])

        y_hat = self.predict_(x)
        plt.plot(thetas, mse, label="loss function", color='black', linewidth=1, linestyle='-')
        plt.grid(True)
        plt.legend()
        plt.show()


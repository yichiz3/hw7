"""Module containing classes for supervised linear regression models."""
import numpy as np
from scipy.linalg import lstsq

from statmodels.regression.utils.regularization import l1_regularization, l2_regularization,l1_l2_regularization
from statmodels.regression.utils.metrics import mean_squared_error, r2_score

class Regression():
    """
    Class representing our base regression model.  
    
    Models relationship between a dependant scaler variable y and independent
    variables X by optimizing a cost function with batch gradient descent.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.

    Attributes 
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the regression problem.
    """
    def __init__(self, n_iter=1000, lr=1e-1):
        self.n_iter = n_iter 
        self.lr = lr 

    def fit(self, X, y):
        """
        Fit linear model with batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples, 1)
            Target values. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Store number of samples and features in variables.
        n_samples, n_features = np.shape(X)
        self.training_errors = []

        # Randomly intialize weights using glorot uniform intializer.
        limit = np.sqrt(2 / n_features)
        self.coef_ = np.random.uniform(-limit, limit, (n_features,))

        # Batch gradient descent for number iterations = n_iter.
        for _ in range(self.n_iter):
            y_preds = X.dot(self.coef_)

            # Penalty term if regularized (don't include bias term).
            regularization = self.regularization(self.coef_[1:])

            # Calculate mse + penalty term if regularized.
            cost_function = mean_squared_error(y, y_preds) + regularization
            self.training_errors.append(cost_function) 

            # Regularization term of gradients (don't include bias term).
            gradient_reg = self.regularization.grad(self.coef_[1:])

            # Gradients of loss function.
            gradients = (2/n_samples) * X.T.dot(y_preds - y)
            gradients = gradients + gradient_reg

            # Update the weights.
            self.coef_ -= self.lr * gradients 

        return self 

    def predict(self, X):
        """
        Estimate target values using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Instances.

        Returns
        -------
        C : array of shape (n_samples,)
            Estimated targets per instance.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        return X.dot(self.coef_)

    def score(self, X, y):
        """
        Calculate the coefficient of determination, R^2 of the predictions.

            R^2 = 1 - SS_res / SS_tot

        where SS_res is the residual sum of squares and SS_tot is the total
        sum of squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for model to be scores against.
        y : array-like of shape (n_samples,).
            True values for test samples.

        Returns
        -------
        score : float
            R^2 calculated on test samples.
        """
        y_preds = self.predict(X)

        score = r2_score(y, y_preds)

        return score


class LinearRegression(Regression):
    """
    Class representing a linear regression model.

    Models relationship between target variable and attributes by computing 
    line that minimizes mean squared error.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.     
    solver : {'bgd', 'lstsq'}, default="bgd"
        Optimization method used to minimize mean squared error in training.

        'bgd' : 
            Batch gradient descent.

        'lstsq' : 
            Ordinary lease squares method using scipy.linalg.lstsq.

    Attributes 
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using ordinary least squares method
    or batch gradient descent.  See solver parameter above.
    """
    def __init__(self, n_iter=1000, lr=1e-1, solver='bgd'):
        self.solver = solver 

        # No regularization.
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iter=n_iter, lr=lr)

    def fit(self, X, y):
        """
        Fit linear regression model.

        If solver='bgd', model is trained using batch gradient descent. 
        If solver='lstsq' model is trained using ordinary least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples,)
            Target values. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # If solver is 'lstsq' use ordinary least squares optimization method.
        if self.solver == 'lstsq':

            # Insert X_0 = 1 for the bias term.
            X = np.insert(X, 0, 1, axis=1)
            
            # Scipy implementation of least squares.
            self.coef_, residues, rank, singular = lstsq(X, y)

            return self

        elif self.solver == 'bgd': 
            super(LinearRegression, self).fit(X, y)


class RidgeRegression(Regression):
    """
    Class representing a linear regression model with l2 regularization.

    Minimizes the cost fuction:

        J(w) = MSE(w) + alpha * 1/2 * ||w||^2

    where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.

    Attributes 
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using batch gradient descent at
    current version.
    """
    def __init__(self, n_iter=1000, lr=1e-1, alpha=1.0, solver='bgd'):
        self.alpha = alpha
        self.regularization = l2_regularization(alpha=self.alpha)
        super(RidgeRegression, self).__init__(n_iter=n_iter, lr=lr)


class LassoRegression(Regression):
    """
    Class representing a linear regression model with l1 regularization.

    Minimizes the cost fuction:

        J(w) = MSE(w) + alpha * ||w||

    where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-2
        Learning rate determining the size of steps in batch gradient descent.
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using batch gradient descent at
    current version.
    """
    def __init__(self, n_iter=1000, lr=1e-2, alpha=1.0):
        self.alpha = alpha
        self.regularization = l1_regularization(alpha=self.alpha)
        super(LassoRegression, self).__init__(n_iter=n_iter, lr=lr)


class ElasticNetRegression(Regression):
    """
    Class representing a linear regression model with a mix of l1 and l2 
    regularization.

    Minimizes the cost function:

        J(w) = MSE(w) + r * alpha * ||w|| + (1 - r) * alpha * 1/2 * ||w||^2

    where w is the vector of feature weights, r is the mix ratio, and alpha
    is the hyperparameter controlling how much regularization is done.

    Parameters
    ----------
    n_iter : float, default=1000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-2
        Learning rate determining the size of steps in batch gradient descent.
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.
    r : float, default=0.5
        Mix ratio determining the amount of l1 vs l2 regularization to add.  
        A value of 0 is equivalent to l2 regularization and a value of 1 is
        equivalent to l1 regularization.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the regression problem.

    Notes
    -----
    This class is capable of being trained using batch gradient descent at
    current version.
    """
    def __init__(self, n_iter=1000, lr=1e-2, alpha=1.0, r=0.5):
        self.alpha = alpha
        self.r = r 
        self.regularization = l1_l2_regularization(alpha=self.alpha, r=self.r)
        super(ElasticNetRegression, self).__init__(n_iter=n_iter, lr=lr)
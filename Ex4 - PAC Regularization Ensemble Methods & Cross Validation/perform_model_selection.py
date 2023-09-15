from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    # Randomly divide to train and test sets, with n_samples in the train set
    train_idx = np.random.choice(y.size, n_samples)
    test_idx = np.arange(y.size)[~np.isin(np.arange(y.size), train_idx)]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # we want cross validation to be chosen in a random manner. but, we also want consistency when checking the
    # performance of different hyperparameters (lambdas for Lasso and Ridge). so we randomize here:
    random_idx = np.random.choice(np.arange(y_train.size), y_train.size, replace=False)
    X_train, y_train = X_train[random_idx], y_train[random_idx]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    # after exploring best range, chosen range is [0.001, 0.5] for Ridge, [0.001, 2] for Lasso
    ridge_lambdas, lasso_lambdas = np.linspace(0.001, 0.5, n_evaluations), np.linspace(0.001, 2, n_evaluations)
    ridge_scores, lasso_scores = np.zeros((n_evaluations, 2)), np.zeros(
        (n_evaluations, 2))  # we get both train & validation errors

    for i in range(n_evaluations):
        ridge_scores[i] = cross_validate(RidgeRegression(ridge_lambdas[i]), X_train, y_train, mean_square_error)
        lasso_scores[i] = cross_validate(Lasso(lasso_lambdas[i], max_iter=5000), X_train, y_train, mean_square_error)
    fig = make_subplots(1, 2, subplot_titles=["Ridge Regression", r"Lasso Regression"], shared_xaxes=True) \
        .update_layout(title="Train and Validation Errors (averaged over the k-folds)", width=750, height=300,
                       margin=dict(t=60)).update_xaxes(title="λ - Regularization parameter") \
        .add_traces([go.Scatter(x=ridge_lambdas, y=ridge_scores[:, 0], name="Ridge Train Error"),
                     go.Scatter(x=ridge_lambdas, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                     go.Scatter(x=lasso_lambdas, y=lasso_scores[:, 0], name="Lasso Train Error"),
                     go.Scatter(x=lasso_lambdas, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    fig.show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model

    # Report the best hyperparameters according to the minimizer of the validation error (which simulate the test error)
    best_ridge_lambda = ridge_lambdas[np.argmin(ridge_scores[:, 1])]
    best_lasso_lambda = lasso_lambdas[np.argmin(lasso_scores[:, 1])]
    print(f"Best hyperparameter (regularization values):\n Ridge Regression: {best_ridge_lambda} \n Lasso Regression: "
          f"{best_lasso_lambda}")

    # Report the losses of best models
    ridge_loss = RidgeRegression(best_ridge_lambda).fit(X_train, y_train).loss(X_test, y_test)
    lasso_loss = mean_square_error(y_test, Lasso(best_lasso_lambda).fit(X_train, y_train).predict(X_test))
    ols_loss = LinearRegression().fit(X_train, y_train).loss(X_test, y_test)
    print(f"Mean Square Error for different Regression Models:\n Ridge Regression over test set: {ridge_loss}\n Lasso Regression: "
          f"{lasso_loss}\n Least Squares Regression: {ols_loss}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()

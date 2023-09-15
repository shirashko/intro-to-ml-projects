from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Randomly partition X to k=cv disjoint subsets, folds
    indices = np.arange(len(X))  # used to create an array of indices ranging from 0 to n_samples - 1
    folds = np.array_split(indices, cv)  # create list of numpy arrays, where each array represents a fold and contains
    # the indices of the samples assigned to that fold

    train_score, validation_score = 0, 0
    for fold_indices in folds:
        train_mask = ~np.isin(indices, fold_indices)  # creates a boolean mask where the samples that are not in the
        # current fold are marked as True, indicating they should be included in the training set.
        fitted_model = deepcopy(estimator).fit(X[train_mask], y[train_mask])  # Deepcopy for safely perform cross-
        # validation without modifying the original estimator or introducing any unwanted interactions between different
        # folds.

        # Calculate the loss of the model on the train set
        y_train_pred = fitted_model.predict(X[train_mask])
        train_score += scoring(y[train_mask], y_train_pred)  # callable function

        # Calculate the loss of the model on current fold functioning as a test set.
        y_val_pred = fitted_model.predict(X[fold_indices])
        validation_score += scoring(y[fold_indices], y_val_pred)  # callable function

    # can assume cv != 0
    return train_score / cv, validation_score / cv

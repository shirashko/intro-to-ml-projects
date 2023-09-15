from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


# CART - Classification And Regression Trees.
# A decision stump is a machine learning model consisting of a one-level decision tree (Decision tree of depth 1)

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        number_of_features = X.shape[1]
        best_err, best_thr = 1, 0
        # check for each feature what is the best threshold achived by it to choose the best one
        for j, sign in product(range(number_of_features), [-1, 1]):
            thr, err = self._find_threshold(X[:, j], y, sign)
            if err < best_err:
                best_err = err
                self.threshold_ = thr
                self.j_ = j
                self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        condition = X[:, self.j_] >= self.threshold_
        return np.where(condition, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_idx = np.argsort(values)
        values, labels = values[sorted_idx], labels[sorted_idx]
        loss_array = [np.sum(np.abs(labels)[np.sign(labels) != sign])]  # for threshold that tag all labels are as sign

        # logic: we start with loss of every sample is tagged as 1. each iteration, we change the tagging so the next
        # sqample in the array will be tagged as -sign, when the last iteration arrives, all the values in the array are
        # tagged as -1. the updating rule: take the current loss and check- if the values[idx] was misclassified until
        # now when we will change its tag to -sign it will be classified correctly, so no need to sum in the loss the
        # abs(label[idx]), meaning we need to subtract it from the prev loss, and this is the only change (all other
        # samples are tagged the same). On the other hand, if the change in the tag of values[idx] made the sample to
        # become misclassify, we need to add the abs(label[idx]) to the loss calculation.
        # the compact way for the update rule is in the loop
        for idx in range(len(values)):
            loss_array.append(loss_array[idx] + (sign * labels[idx]))
        min_loss_idx = np.argmin(loss_array)
        thr_err = loss_array[min_loss_idx]
        # -np.inf stands for -infinity and np.inf stands for +infinity. if best thr tag all labels as sign, take the
        # thr to be -infinity, and if best labling is to label all as -sign, take thr=infinity
        thr = np.concatenate([[-np.inf], values[1:], [np.inf]])[min_loss_idx]
        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

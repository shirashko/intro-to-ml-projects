from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit` // I fixed the documentation according to the forum
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, labels_count = np.unique(y, return_counts=True)
        y_mapped_to_range = np.searchsorted(self.classes_, y)  # check if I can assume y will contain only labels in
        # the range 0-number_of_classes
        train_size, num_of_classes = len(X), len(self.classes_)
        self.pi_ = labels_count / train_size
        self.mu_ = np.array([(np.mean(X[y == i], axis=0)) for i in self.classes_])
        centered_X = X - self.mu_[y_mapped_to_range]  # if y3=1 so sample 3 is from gaussian with mu1, so need to
        # calculate x3-mu1. self.mu_[y] is an array that contains the rows from self.mu_ corresponding to the indices
        # provided by y.
        P = centered_X.T @ centered_X  # projection matrix
        self.cov_ = P / (train_size - num_of_classes)  # we want unbiased estimator
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # for each sample, we want to find the most likely label. when calling likelihood function with the samples,
        # we get in return a matrix, in which each row correspond to one sample.
        most_likely_label = np.argmax(self.likelihood(X), axis=1)  # choose for each row the column index which maximize
        return self.classes_[most_likely_label]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        const = 1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * det(self.cov_))
        centered_X = np.tile(X, (len(self.mu_), 1)) - np.repeat(self.mu_, len(X), axis=0)
        exp_expression = np.exp(-0.5 * np.sum((centered_X @ self._cov_inv) * centered_X, axis=1))
        return const * exp_expression.reshape((len(X), len(self.classes_)), order="F") * self.pi_

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

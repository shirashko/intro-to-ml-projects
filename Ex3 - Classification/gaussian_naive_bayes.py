from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        train_size = len(X)
        self.classes_, labels_count = np.unique(y, return_counts=True)
        self.pi_ = labels_count / train_size
        # Notice that here we view the mean matrix as matrix which it's ij element in mu(i,j), which is the mean of
        # the (i,j) gaussian distribution. In contrast, in the LDA the i'th row of the mean matrix represents the (i)
        # gaussian distribution. although the point of view is different, in practical manner, we get the same
        # calculation because in the second one we average on each feature "separately"
        self.mu_ = np.array([np.mean(X[y == i], axis=0) for i in self.classes_])
        self.vars_ = np.array([np.var(X[y == i], axis=0, ddof=1) for i in self.classes_])  # we're using the average for
        # the calculation which can be concluded from the samples in X. The divisor used in the calculation is N - ddof,
        # so ddof should be 1 if we want unbiased estimator

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
        most_likely_label = np.argmax(self.likelihood(X), axis=1)  # returns numpy array with the column indices (which
        # correspond with the class indices) that maximize each sample (row)
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

        const = np.repeat(np.sqrt(2 * np.pi * self.vars_), len(X), axis=0)
        centered_X = np.tile(X, (len(self.mu_), 1)) - np.repeat(self.mu_, len(X), axis=0)
        exp_expression = (np.exp(centered_X ** 2 / np.repeat((- 2 * self.vars_), len(X), axis=0))) / const
        exp_expression = np.prod(exp_expression, axis=1)
        res = exp_expression.reshape((len(X), len(self.classes_)), order="F")
        return res * self.pi_


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
        return misclassification_error(y, self.predict(X))

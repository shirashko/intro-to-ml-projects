import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss
    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    Returns
    -------
    MSE of given predictions
    """
    # can assume validity of the input ? len(y) > 0?
    diff = y_true - y_pred
    return (diff ** 2).mean()


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss
    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not
    Returns
    -------
    Misclassification of given predictions
    """
    # miss classification error is the sum of examples which was misclassified (y_true ! y_pred).
    normalization_factor = 1
    if normalize:
        normalization_factor = len(y_pred)
    return np.sum(y_true != y_pred) / normalization_factor


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions
    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    Returns
    -------
    Accuracy of given predictions
    """
    # The Accuracy is the number of correct classification out of all predictions:
    return float(np.mean(y_true == y_pred))


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions
    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data
    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)
    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    raise NotImplementedError()

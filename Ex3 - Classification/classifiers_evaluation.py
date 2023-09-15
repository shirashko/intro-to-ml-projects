from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def cached_losses(fit: Perceptron, _, __):
            """
             adds the loss of the current state in the fitted algorithm.
             we don't need current sample and current response for our purposes, using only the perceptron instance
             to calculate the loss"""
            losses.append(fit.loss(X, y))

        Perceptron(callback=cached_losses).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(data=go.Scatter(x=list(range(len(losses))), y=losses, marker=dict(color="black")))
        fig.update_layout(title=f"The error of the Perceptron algorithm during training on {n} data"
                                f"over the course of the fitting process", xaxis_title="Iteration",
                          yaxis_title="Misclassification Error").show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        lda_prediction, gnb_prediction = lda.predict(X), gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Gaussian Naive Bayes predictions, Accuracy = "
                                                            f"{accuracy(y, gnb_prediction)}",
                                                            f"LDA predictions, Accuracy = {accuracy(y, lda_prediction)}"])

        fig.update_layout(title={ "text": f"Compare Gaussian Classifiers for {f[0:10]} data set", "x": 0.5, "xanchor":
            "center"}, margin={"t": 100}, showlegend=False)  # adding title in the middle and make margin so the title
        # be spaced from the rest

        # Add traces for data-points setting symbols and colors
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=gnb_prediction, symbol=y),),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=lda_prediction, symbol=y))],
                       rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers", marker=dict(color="black", symbol=
        "x", size=10)), go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers", marker=dict(color="black",
                                                        symbol="x", size=10))], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians

        # for lda we have the same covariance matrix for each class.
        # for gnb we have vars for each feature, which from it, we can derive the covariance matrix for each class.
        # The assumption when using the gaussian naive bayes is that the features are independant, meaning the
        # covariance of each pair of features equals zero.
        # So the covariance of label i samples is a diagonal matrix, which it's diagonl equals the row of gnb.var

        for i in range(len(lda.classes_)):
            fig.add_traces([get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    lowest_loss, argmin_eta, min_loss = np.inf, None, np.inf
    for module, L in [(L1, "L1 norm"), (L2, "squared L2 norm")]:
        fig = go.Figure(
            layout=dict(title=f"Convergence rate for {L} function for different learning rates",
                        xaxis_title='iteration', yaxis_title='f'))
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            minimizer = GradientDescent(learning_rate=FixedLR(eta), callback=callback, out_type="best").fit(
                module(init.copy()), X=None, y=None)
            plot_descent_path(module, np.array(weights), title=f"The descent path achieved with {L} "
                                                               f"and learning rate: {eta}").show()

            fig.add_trace(go.Scatter(x=np.arange(len(values)), y=values, name=f"{L} - learning rate {eta}"))

            # Find the lowest loss achieved when minimizing each of the modules
            min_loss = module(minimizer).compute_output()
            if min_loss < lowest_loss:
                lowest_loss = min_loss
                argmin_eta = eta

        fig.show()
        print(f"Module {L} achieves lowest loss: {np.round(lowest_loss, 5)} with learning rate: {argmin_eta}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    # our classes work with arrays and not with data frames
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    y_proba = LogisticRegression().fit(X_train, y_train).predict_proba(X_train)

    # Find the best threshold (alpha) & print the test error achieved by it
    fpr, tpr, thresholds = roc_curve(y_train, y_proba)  # roc_curve function consider each y_proba value as
    # a threshold and returns all the thresholds & fpr & tpr received by it

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_threshold = thresholds[np.argmax(tpr - fpr)]
    test_error_with_alpha = LogisticRegression(alpha=best_threshold).fit(X_train, y_train).loss(X_test, y_test)
    print(f"The optimal threshold for TP & FP rations is: {np.round(best_threshold, 5)}. using this threshold we "
          f"achieve test error: {np.round(test_error_with_alpha, 5)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    for penalty in ["l1", "l2"]:
        test_errors = []
        for lam in lambdas:
            model = LogisticRegression(solver=solver, penalty=penalty, lam=lam)
            _, val_score = cross_validate(model, X_train, y_train, misclassification_error)
            test_errors.append(val_score)

        # find the best lambda when using this penalty & the loss achieved by it
        best_lambda = lambdas[np.argmin(test_errors)]
        loss = LogisticRegression(solver=solver, penalty=penalty, lam=best_lambda). \
            fit(X_train, y_train).loss(X_test, y_test)
        print(
            f"selected lambda for {penalty} regularization term is: {best_lambda}, getting test error of: {np.round(loss, 5)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()

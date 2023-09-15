from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    ug = UnivariateGaussian()
    true_expectation = 10
    true_var = 1
    X = np.random.normal(true_expectation, true_var, 1000)
    ug.fit(X)
    expectation, variance = ug.mu_, ug.var_
    print(f"({round(expectation, 3)}, {round(variance, 3)})")

    # Question 2 - Empirically showing sample mean is consistent
    dist_of_sample_sizes = []
    sample_sizes = list(range(10, 1001, 10))
    for size in sample_sizes:
        ug.fit(X[0:size])
        dist_of_sample_sizes.append(abs(ug.mu_ - true_expectation))
    plt.scatter(sample_sizes, dist_of_sample_sizes, s=0.8)
    plt.xlabel("sample size")
    plt.ylabel("distance between estimated- and true value of mean")
    plt.title("Question 2- deviation of estimated expectation from true expectation \n as function of sample size"
              "for samples from N(10,1) distribution")
    plt.show()

    # question 3
    plt.scatter(X, ug.pdf(X), s=0.5)
    plt.xlabel("x")
    plt.ylabel("pdf(x)")
    plt.title("Question 3- PDF values of samples from N(10,1) distribution")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)  # m=1000 rows, d=4 columns
    ug = MultivariateGaussian().fit(X)
    estimated_mu, estimated_cov = ug.mu_, ug.cov_
    print(np.round(estimated_mu, 3))
    print(np.round(estimated_cov, 3))

    # Question 5 - Likelihood evaluation
    vals = np.linspace(-10, 10, 200)
    f1_f3_pairs = [(i, j) for i in vals for j in vals]
    log_likelihood = []
    for pair in f1_f3_pairs:
        f1, f3 = pair[0], pair[1]
        m = np.array([f1, 0, f3, 0])
        log_likelihood.append(ug.log_likelihood(m, cov, X))

    max_ = np.max(log_likelihood)
    maximize_pair = f1_f3_pairs[log_likelihood.index(max_)]

    log_likelihood = np.flip(np.array(log_likelihood).reshape(200, 200), 0)

    go.Figure(go.Heatmap(x=vals, y=vals, z=log_likelihood), layout=dict(template="simple_white",
                          title="Question 5- Log likelihood of Multivariate Gaussian as \nfunction of mean's features "
                                "1,3", xaxis_title="feature 3 value", yaxis_title="feature 1 value")).show()

    # Question 6 - Maximum likelihood
    print("features 1,3 that maximize the log-likelihood are: ")
    f1, f3 = round(maximize_pair[0], 3), round(maximize_pair[1], 3)
    print("f1=", f1, ",f3=", f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

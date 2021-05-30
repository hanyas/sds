import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import ARDRegression, LinearRegression

# Parameters of the example
np.random.seed(0)
n_samples, n_features = 100, 100
# Create Gaussian data
X = np.random.randn(n_samples, n_features)
# Create weights with a precision lambda_ of 4.
lambda_ = 4.
w = np.zeros(n_features)
# Only keep 10 weights of interest
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# Create noise with a precision alpha of 50.
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
# Create the target<
y = np.dot(X, w) + noise

clf = ARDRegression(fit_intercept=False, n_iter=1000)
clf.fit(X, y)

ols = LinearRegression(fit_intercept=False)
ols.fit(X, y)

from copy import deepcopy

from sds.distributions.lingauss import IndependentLinearGaussianWithKnownPrecision
from sds.distributions.lingauss import IndependentLinearGaussianWithKnownMean
from sds.distributions.gaussian import GaussianWithPrecision
from sds.distributions.gaussian import GaussianWithKnownMeanAndDiagonalPrecision
from sds.distributions.gamma import Gamma

likelihood_precision_prior = Gamma(dim=1, alphas=np.array([1e-16]),
                                   betas=np.array([1e-16]))

parameter_precision_prior = Gamma(dim=n_features, alphas=1e-16 * np.ones((n_features, )),
                                  betas=1e-16 * np.ones((n_features, )))

likelihood_precision_posterior = deepcopy(likelihood_precision_prior)
parameter_precision_posterior = deepcopy(parameter_precision_prior)
parameter_posterior = None

for _ in range(100):
    # parameter posterior
    alpha = parameter_precision_posterior.mean()
    parameter_prior = GaussianWithPrecision(dim=n_features,
                                            mu=np.zeros((n_features, )),
                                            lmbda=alpha * np.eye(n_features))
    parameter_posterior = deepcopy(parameter_prior)

    beta = likelihood_precision_posterior.mean()
    likelihood_known_precision = IndependentLinearGaussianWithKnownPrecision(input_dim=n_features,
                                                                             lmbda=beta * np.ones((1, )),
                                                                             affine=False)

    stats = likelihood_known_precision.statistics(X, y)
    parameter_posterior.nat_param = parameter_prior.nat_param + stats

    # likelihood precision posterior
    param = parameter_posterior.mean()
    likelihood_known_mean = IndependentLinearGaussianWithKnownMean(input_dim=n_features,
                                                                   A=param, affine=False)

    stats = likelihood_known_mean.statistics(X, y)
    likelihood_precision_posterior.nat_param = likelihood_precision_prior.nat_param + stats

    # parameter precision posterior
    parameter_likelihood = GaussianWithKnownMeanAndDiagonalPrecision(dim=n_features)

    param = parameter_posterior.mean()
    stats = parameter_likelihood.statistics(param)
    parameter_precision_posterior.nat_param = parameter_precision_prior.nat_param + stats

our_ard = parameter_posterior.mode()

from sds.distributions.composite import MatrixNormalGamma
from sds.distributions.lingauss import LinearGaussianWithDiagonalPrecision

M = np.zeros((1, n_features))
K = 1e-16 * np.eye(n_features)
alphas = 1e-16 * np.ones((1, ))
betas = 1e-16 * np.ones((1, ))

prior = MatrixNormalGamma(input_dim=n_features,
                          output_dim=1,
                          M=M, K=K, alphas=alphas, betas=betas)

posterior = deepcopy(prior)
likelihood = LinearGaussianWithDiagonalPrecision(input_dim=n_features,
                                                 output_dim=1,
                                                 affine=False)

stats = likelihood.statistics(X, np.atleast_2d(y).T)
posterior.nat_param = prior.nat_param + stats
our_ols = posterior.mode()[0]

plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
plt.plot(clf.coef_, color='darkblue', linestyle='-', linewidth=2, label="Sklearn ARD")
plt.plot(our_ard, color='red', linestyle='-', linewidth=2, label="Our ARD")
# plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2, label="Sklearn OLS")
# plt.plot(our_ols.flatten(), color='cyan', linestyle='-', linewidth=2, label="Our OLS")

plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc=1)

plt.show()

import numpy as np
import numpy.random as npr

from scipy.special import gammaln
from sds.utils.general import Statistics as Stats


class Gamma:
    # In comparison to a Wishart distribution
    # alpha = nu / 2.
    # beta = 1. / (2. * psi)

    def __init__(self, dim, alphas, betas):
        self.dim = dim

        self.alphas = alphas  # shape
        self.betas = betas  # rate

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        alphas = params[0] - 1
        betas = - params[1]
        return Stats([alphas, betas])

    @staticmethod
    def nat_to_std(natparam):
        alphas = natparam[0] + 1
        betas = - natparam[1]
        return alphas, betas

    def mean(self):
        return self.alphas / self.betas

    def mode(self):
        assert np.all(self.alphas >= 1.)
        return (self.alphas - 1.) / self.betas

    def rvs(self):
        # numpy uses a different parameterization
        return npr.gamma(self.alphas, 1. / self.betas, size=(self.dim,))

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def log_likelihood(self, x):
        loglik = np.sum((self.alphas - 1.) * np.log(x) - self.betas * x)
        return - self.log_partition() + self.log_base() + loglik

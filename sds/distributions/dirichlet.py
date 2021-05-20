import numpy as np
import numpy.random as npr

from scipy.special import gammaln


class Dirichlet:

    def __init__(self, dim=None, alphas=None):
        self.dim = dim
        self.alphas = alphas

    @property
    def params(self):
        return self.alphas

    @params.setter
    def params(self, values):
        self.alphas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        alphas = params - 1.
        return alphas

    @staticmethod
    def nat_to_std(natparam):
        alphas = natparam + 1.
        return alphas

    def mean(self):
        return self.alphas / np.sum(self.alphas)

    def mode(self):
        assert np.all(self.alphas > 1.), "Make sure alphas > 1."
        return (self.alphas - 1.) / (np.sum(self.alphas) - self.dim)

    def rvs(self):
        return npr.dirichlet(self.alphas)

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            logx = np.log(data)
            return logx
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data, weights = data[idx], weights[idx]

            logx = np.einsum('n,nk->nk', weights, np.log(data))
            return logx
        else:
            return list(map(self.weighted_statistics, data, weights))

    def log_partition(self):
        return np.sum(gammaln(self.alphas)) - gammaln(np.sum(self.alphas))

    def log_likelihood(self, x):
        loglik = np.sum((self.alphas - 1.) * np.log(x))
        return - self.log_partition() + self.log_base() + loglik

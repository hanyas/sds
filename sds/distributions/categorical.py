import numpy as np
import numpy.random as npr


class Categorical:

    def __init__(self, dim=None, pi=None):
        self.dim = dim
        self.pi = pi

        if dim is not None and pi is None:
            self.pi = 1. / self.dim * np.ones((self.dim, ))

    @property
    def params(self):
        return self.pi

    @params.setter
    def params(self, values):
        self.pi = values

    @property
    def nb_params(self):
        return len(self.pi) - 1

    def mean(self):
        raise NotImplementedError

    def mode(self):
        return np.argmax(self.pi)

    def rvs(self):
        return npr.choice(a=self.dim, p=self.pi)

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            return np.bincount(data, minlength=self.dim)
        else:
            return sum(list(map(self.statistics, data)))

    def weighted_statistics(self, data, weights):
        if isinstance(weights, np.ndarray):
            assert weights.ndim in (1, 2)
            if data is None or weights.ndim == 2:
                return np.sum(np.atleast_2d(weights), axis=0)
            else:
                return np.bincount(data, weights, minlength=self.dim)
        else:
            data = data if data else [None] * len(weights)
            return sum(list(map(self.weighted_statistics, data, weights)))

    def log_partition(self):
        raise NotImplementedError

    def log_likelihood(self, x):
        bads = np.isnan(x)
        loglik = np.zeros_like(x, dtype=np.double)
        err = np.seterr(invalid='ignore', divide='ignore')
        loglik[~bads] = np.log(self.pi)[list(x[~bads])]  # log(0) can happen, no warning
        np.seterr(**err)
        return loglik

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        counts = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)
        self.pi = counts / counts.sum()

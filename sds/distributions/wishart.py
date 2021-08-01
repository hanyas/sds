import numpy as np
import numpy.random as npr

from scipy.special import multigammaln
from sds.utils.general import Statistics as Stats


class Wishart:

    def __init__(self, dim, psi=None, nu=None):
        self.dim = dim

        self.nu = nu

        self._psi = psi
        self._psi_chol = None

    @property
    def params(self):
        return self.psi, self.nu

    @params.setter
    def params(self, values):
        self.psi, self.nu = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        psi = - 0.5 * np.linalg.inv(params[0])
        nu = 0.5 * (params[1] - psi.shape[0] - 1)
        return Stats([psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        psi = - 0.5 * np.linalg.inv(natparam[0])
        nu = 2. * natparam[1] + psi.shape[0] + 1
        return psi, nu

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value
        self._psi_chol = None

    @property
    def psi_chol(self):
        if self._psi_chol is None:
            self._psi_chol = np.linalg.cholesky(self.psi)
        return self._psi_chol

    def mean(self):
        return self.nu * self.psi

    def mode(self):
        assert self.nu >= (self.dim + 1)
        return (self.nu - self.dim - 1) * self.psi

    # copied from scipy
    def rvs(self):
        # Random normal variates for off-diagonal elements
        n_tril = self.dim * (self.dim - 1) // 2
        covariances = npr.normal(size=n_tril).reshape((n_tril,))

        # Random chi-square variates for diagonal elements
        variances = (np.r_[[npr.chisquare(self.nu - (i + 1) + 1, size=1)**0.5
                            for i in range(self.dim)]].reshape((self.dim,)).T)

        A = np.zeros((self.dim, self.dim))

        # Input the covariances
        tril_idx = np.tril_indices(self.dim, k=-1)
        A[tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(self.dim)
        A[diag_idx] = variances

        T = np.dot(self.psi_chol, A)
        return np.dot(T, T.T)

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               + self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def log_likelihood(self, x):
        log_lik = 0.5 * (self.nu - self.dim - 1) * np.linalg.slogdet(x)[1]\
                 - 0.5 * np.trace(np.linalg.solve(self.psi, x))
        return - self.log_partition() + self.log_base() + log_lik

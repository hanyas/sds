from autograd import numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import logsumexp

import scipy as sc
from scipy import linalg, stats

from scipy.stats import multivariate_normal as mvn
from sds.stats import multivariate_normal_logpdf


class CategoricalInitState:

    def __init__(self, nb_states, prior, reg=1e-128):
        self.nb_states = nb_states

        self.prior = prior
        self.reg = reg

        self.logpi = - np.log(self.nb_states) * np.ones(self.nb_states)

    @property
    def params(self):
        return self.logpi

    @params.setter
    def params(self, value):
        self.logpi = value

    @property
    def pi(self):
        return np.exp(self.logpi - logsumexp(self.logpi))

    def initialize(self, x, u):
        pass

    def sample(self):
        return npr.choice(self.nb_states, p=self.pi)

    def maximum(self):
        return np.argmax(self.pi)

    def log_init(self):
        return self.logpi - logsumexp(self.logpi)

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def permute(self, perm):
        self.logpi = self.logpi[perm]

    def mstep(self, gamma, **kwargs):
        _pi = sum([_w[0, :] for _w in gamma]) + self.reg
        self.logpi = np.log(_pi / sum(_pi))


class GaussianInitObservation:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-128):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.mu = npr.randn(self.dm_obs)

        if self.prior:
            _cov = sc.stats.invwishart.rvs(prior['nu0'], prior['psi0'] * np.eye(dm_obs))
            self._sqrt_cov = sc.linalg.cholesky(_cov * np.eye(dm_obs))
        else:
            self._sqrt_cov = npr.randn(self.dm_obs, self.dm_obs)

    @property
    def params(self):
        return self.mu, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_cov = value

    def mean(self):
        return self.mu

    @property
    def cov(self):
        return self._sqrt_cov @ self._sqrt_cov.T

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, stoch=True):
        if stoch:
            _x = mvn(mean=self.mu, cov=self.cov).rvs()
            return np.atleast_1d(_x)
        else:
            return self.mean()

    def initialize(self, x, u, **kwargs):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([multivariate_normal_logpdf(_x[0, :], self.mu, self.cov)
                                       for _ in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, **kwargs):
        _x0 = np.vstack([_x[0, :] for _x in x])
        self.mu = np.mean(_x0, axis=0)
        self.cov = np.cov(m=_x0, rowvar=False, bias=True)

    def smooth(self, gamma, x, u):
        mean = []
        for _ in range(len(x)):
            mean.append(self.mu)
        return mean


class GaussianInitControl:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-128):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.mu = npr.randn(self.dm_act)

        if self.prior:
            _cov = sc.stats.invwishart.rvs(prior['nu0'], prior['psi0'] * np.eye(dm_act))
            self._sqrt_cov = sc.linalg.cholesky(_cov * np.eye(dm_act))
        else:
            self._sqrt_cov = npr.randn(self.dm_act, self.dm_act)

    @property
    def params(self):
        return self.mu, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_cov = value

    def mean(self):
        return self.mu

    @property
    def cov(self):
        return self._sqrt_cov @ self._sqrt_cov.T

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_act))

    def sample(self, stoch=True):
        if stoch:
            _u = mvn(mean=self.mu, cov=self.cov).rvs()
            return np.atleast_1d(_u)
        else:
            return self.mean()

    def initialize(self, x, u, **kwargs):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _u in u:
            _loglik = np.column_stack([multivariate_normal_logpdf(_u[0, :], self.mu, self.cov)
                                       for _ in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, **kwargs):
        _u0 = np.vstack([_u[0, :] for _u in u])
        self.mu = np.mean(_u0, axis=0)
        self.cov = np.cov(m=_u0, rowvar=False, bias=True)

    def smooth(self, gamma, x, u):
        mean = []
        for _ in range(len(u)):
            mean.append(self.mu)
        return mean

import numpy as np
import numpy.random as npr

import torch

import scipy as sc
from scipy import stats

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as invw

from sds_torch.stats import multivariate_normal_logpdf as lg_mvn

from sds_torch.utils import random_rotation
from sds_torch.utils import linear_regression


class GaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-32):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.mu = torch.rand(self.nb_states, self.dm_obs)

        # self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

        self._sqrt_cov = torch.zeros((self.nb_states, self.dm_obs, self.dm_obs), dtype=torch.float64)
        for k in range(self.nb_states):
            _cov = torch.from_numpy(sc.stats.invwishart.rvs(self.dm_obs + 1, np.eye(self.dm_obs)))
            self._sqrt_cov[k, ...] = torch.cholesky(_cov * np.eye(self.dm_obs))

    @property
    def params(self):
        return self.mu, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_cov = value

    def mean(self, z, x=None, u=None):
        return self.mu[z, :]

    @property
    def cov(self):
        return torch.matmul(self._sqrt_cov, self._sqrt_cov.permute(0, 2, 1))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = torch.cholesky(value + self.reg * torch.eye(self.dm_obs, dtype=torch.float64))

    def sample(self, z, x=None, u=None):
        _x = mvn(mean=self.mean(z), cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def initialize(self, x, u, **kwargs):
        from sklearn.cluster import KMeans
        _obs = np.concatenate(x)
        km = KMeans(self.nb_states).fit(_obs)

        self.mu = torch.from_numpy(km.cluster_centers_)
        self.cov = torch.from_numpy(np.array([np.cov(_obs[km.labels_ == k].T)
                             for k in range(self.nb_states)]))

    def permute(self, perm):
        self.mu = self.mu[perm]
        self._sqrt_cov = self._sqrt_cov[perm]

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x in x:
            _loglik = torch.stack([torch.distributions.MultivariateNormal(self.mean(k), self.cov[k]).log_prob(_x)
                                       for k in range(self.nb_states)], dim=1)
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, weights=None, **kwargs):
        _J = torch.zeros((self.nb_states, self.dm_obs), dtype=torch.float64)
        _h = torch.zeros((self.nb_states, self.dm_obs), dtype=torch.float64)
        for _x, _w in zip(x, gamma):
            _J += torch.sum(_w[:, :, None], dim=0)
            _h += torch.sum(_w[:, :, None] * _x[:, None, :], dim=0)

        self.mu = _h / _J

        sqerr = torch.zeros((self.nb_states, self.dm_obs, self.dm_obs), dtype=torch.float64)
        weight = self.reg * torch.ones((self.nb_states, ), dtype=torch.float64)
        for _x, _w in zip(x, gamma):
            resid = _x[:, None, :] - self.mu
            sqerr += torch.sum(_w[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], dim=0)
            weight += torch.sum(_w, dim=0)

        self.cov = sqerr / weight[:, None, None]

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            mean.append(_gamma.dot(self.mu))
        return mean


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-32):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act


        self.prior = prior
        self.reg = reg

        self._sqrt_cov = torch.zeros((self.nb_states, self.dm_obs, self.dm_obs), dtype=torch.float64)

        self.A = torch.zeros((self.nb_states, self.dm_obs, self.dm_obs), dtype=torch.float64)
        self.B = torch.zeros((self.nb_states, self.dm_obs, self.dm_act), dtype=torch.float64)
        self.c = torch.zeros((self.nb_states, self.dm_obs), dtype=torch.float64)

        # for k in range(self.nb_states):
        #     self._sqrt_cov[k, ...] = npr.randn(self.dm_obs, self.dm_obs)
        #     self.A[k, ...] = .95 * random_rotation(self.dm_obs)
        #     self.B[k, ...] = npr.randn(self.dm_obs, self.dm_act)
        #     self.c[k, :] = npr.randn(self.dm_obs)

        for k in range(self.nb_states):
            _cov = torch.from_numpy(sc.stats.invwishart.rvs(self.dm_obs + 1, np.eye(self.dm_obs)))
            self._sqrt_cov[k, ...] = torch.cholesky(_cov * torch.eye(self.dm_obs, dtype=torch.float64))
            self.A[k, ...] = torch.from_numpy(sc.stats.matrix_normal.rvs(mean=None, rowcov=_cov.numpy(), colcov=_cov.numpy()))
            self.B[k, ...] = torch.from_numpy(sc.stats.matrix_normal.rvs(mean=None, rowcov=_cov.numpy(), colcov=_cov.numpy())[:, [0]])
            self.c[k, ...] = torch.from_numpy(sc.stats.matrix_normal.rvs(mean=None, rowcov=_cov.numpy(), colcov=_cov.numpy())[:, 0])

    @property
    def params(self):
        return self.A, self.B, self.c, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.A, self.B, self.c, self._sqrt_cov = value

    def mean(self, z, x, u):
        # Einsum throws error if action dimension is 0
        return torch.einsum('kh,...h->...k', self.A[z, ...], x) +\
               (torch.einsum('kh,...h->...k', self.B[z, ...], u) if u.shape[1] else 0.) + self.c[z, :]

    @property
    def cov(self):
        return torch.matmul(self._sqrt_cov, self._sqrt_cov.permute(0, 2, 1))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = torch.cholesky(value + self.reg * torch.eye(self.dm_obs, dtype=torch.float64))

    def sample(self, z, x, u):
        _x = mvn(self.mean(z, x, u), cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def reset(self):
        self._sqrt_cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))

        self.A = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        self.B = np.zeros((self.nb_states, self.dm_obs, self.dm_act))
        self.c = np.zeros((self.nb_states, self.dm_obs))

        # for k in range(self.nb_states):
        #     self._sqrt_cov[k, ...] = npr.randn(self.dm_obs, self.dm_obs)
        #     self.A[k, ...] = .95 * random_rotation(self.dm_obs)
        #     self.B[k, ...] = npr.randn(self.dm_obs, self.dm_act)
        #     self.c[k, :] = npr.randn(self.dm_obs)

        for k in range(self.nb_states):
            _cov = torch.from_numpy(sc.stats.invwishart.rvs(self.dm_obs + 1, 1. * np.eye(self.dm_obs)))
            self._sqrt_cov[k, ...] = torch.cholesky(_cov * np.eye(self.dm_obs, dtype=torch.float64))
            self.A[k, ...] = torch.from_numpy(sc.stats.matrix_normal.rvs(mean=None, rowcov=_cov.numpy(), colcov=_cov.numpy()))
            self.B[k, ...] = torch.from_numpy(sc.stats.matrix_normal.rvs(mean=None, rowcov=_cov.numpy(), colcov=_cov.numpy())[:, [0]])
            self.c[k, ...] = torch.from_numpy(sc.stats.matrix_normal.rvs(mean=None, rowcov=_cov.numpy(), colcov=_cov.numpy())[:, 0])

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(x), np.vstack(u))))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [torch.from_numpy(z[:-1]) for z in zs]
        else:
            zs = [torch.from_numpy(npr.choice(self.nb_states, size=T - 1)) for T in Ts]

        _cov = torch.zeros((self.nb_states, self.dm_obs, self.dm_obs), dtype=torch.float64)
        for k in range(self.nb_states):
            ts = [torch.where(z == k)[0] for z in zs]
            xs = [torch.cat((_x[t, :], _u[t, :]), dim=1) for t, _x, _u in zip(ts, x, u)]
            ys = [_x[t + 1, :] for t, _x in zip(ts, x)]

            coef_, intercept_, sigma = linear_regression(torch.cat(xs), torch.cat(ys),
                                                         weights=None, fit_intercept=True,
                                                         **self.prior)
            self.A[k, ...] = coef_[:, :self.dm_obs]
            self.B[k, ...] = coef_[:, self.dm_obs:]
            self.c[k, :] = intercept_
            _cov[k, ...] = sigma

        self.cov = _cov

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        if self.prior:
            for k in range(self.nb_states):
                coef_ = np.column_stack((self.A[k, ...], self.B[k, ...], self.c[k, ...])).flatten()
                lp += mvn(mean=self.prior['mu0'] * np.ones((coef_.shape[0], )),
                          cov=self.prior['sigma0'] * np.eye(coef_.shape[0])).logpdf(coef_)\
                      + invw(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_obs)).logpdf(self.cov[k, ...])
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = torch.stack([torch.distributions.MultivariateNormal(self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act]), self.cov[k]).log_prob(_x[1:, :])
                                       for k in range(self.nb_states)], dim=1)
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, weights=None, use_prior=False):
        aux = []
        if weights:
            for _w, _gamma in zip(weights, gamma):
               aux.append(_w[:, None] * _gamma)
            gamma = aux

        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            xs.append(torch.cat((_x[:-1, :], _u[:-1, :self.dm_act], torch.ones((_x.shape[0] - 1, 1), dtype=torch.float64)), dim=1))
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _cov = torch.zeros((self.nb_states, self.dm_obs, self.dm_obs), dtype=torch.float64)
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=torch.cat(xs), ys=torch.cat(ys),
                                             weights=torch.cat(ws)[:, k], fit_intercept=False,
                                             **self.prior if use_prior else {})

            self.A[k, ...] = coef_[:, :self.dm_obs]
            self.B[k, ...] = coef_[:, self.dm_obs:self.dm_obs + self.dm_act]
            self.c[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
        #         self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
        #         self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
        #         _cov[k] = _cov[i]

        self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x) - 1, self.nb_states, self.dm_obs))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act])
            mean.append(np.einsum('nk,nkl->nl', _gamma[1:, ...], _mu))
        return mean

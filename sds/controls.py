import autograd.numpy as np
import autograd.numpy.random as npr

import scipy as sc
from scipy import stats

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as invw

from sds.stats import multivariate_normal_logpdf

from sds.utils import linear_regression

from autograd.tracer import getval


class LinearGaussianControl:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-16):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.K = npr.randn(self.nb_states, self.dm_act, self.dm_obs)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        self._sqrt_cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        if self.prior:
            for k in range(self.nb_states):
                _cov = sc.stats.invwishart.rvs(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_act))
                self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.dm_act))
        else:
            self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

    @property
    def params(self):
        return self.K, self.kff, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.K, self.kff, self._sqrt_cov = value

    def mean(self, z, x, u):
        return np.einsum('kh,...h->...k', self.K[z, ...], x) + self.kff[z, ...]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_act))

    def sample(self, z, x, u, stoch=True):
        if stoch:
            _u = mvn(mean=self.mean(z, x, u), cov=self.cov[z, ...]).rvs()
            return np.atleast_1d(_u)
        else:
            return self.mean(z, x, u)

    def reinit(self):
        self.K = npr.randn(self.nb_states, self.dm_act, self.dm_obs)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        self._sqrt_cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        if self.prior:
            for k in range(self.nb_states):
                _cov = sc.stats.invwishart.rvs(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_act))
                self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.dm_act))
        else:
            self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit((np.vstack(x)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=T) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [_x[t, :] for t, _x in zip(ts, x)]
            ys = [_u[t, :] for t, _u in zip(ts, u)]

            coef_, intercept_, sigma = linear_regression(np.vstack(xs), np.vstack(ys),
                                                         weights=None, fit_intercept=True,
                                                         **self.prior)
            self.K[k, ...] = coef_[:, :self.dm_obs]
            self.kff[k, :] = intercept_
            _cov[k, ...] = sigma

        self.cov = _cov

    def permute(self, perm):
        self.K = self.K[perm, ...]
        self.kff = self.kff[perm, ...]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        if self.prior:
            for k in range(self.nb_states):
                coef_ = np.column_stack((self.K[k, ...], self.kff[k, ...])).flatten()
                lp += mvn(mean=self.prior['mu0'] * np.ones((coef_.shape[0], )),
                          cov=self.prior['sigma0'] * np.eye(coef_.shape[0])).logpdf(getval(coef_))\
                      + invw(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_act)).logpdf(getval(self.cov[k, ...]))
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_u[1:], self.mean(k, x=_x[1:], u=_u[1:]), self.cov[k])
                                       for k in range(self.nb_states)])
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
            xs.append(np.hstack((_x, np.ones((_x.shape[0], 1)))))
            ys.append(_u)
            ws.append(_w)

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False,
                                             **self.prior if use_prior else {})

            self.K[k, ...] = coef_[:, :self.dm_obs]
            self.kff[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.K[k] = self.K[i] + 0.01 * npr.randn(*self.K[i].shape)
        #         self.kff[k] = self.kff[i] + 0.01 * npr.randn(*self.kff[i].shape)
        #         _cov[k] = _cov[i]

        self.cov = _cov

    # def mstep(self, gamma, x, u, weights=None, reg=1e-16):
    #     aux = []
    #     if weights:
    #         for _w, _gamma in zip(weights, gamma):
    #            aux.append(_w[:, None] * _gamma)
    #         gamma = aux
    #
    #     xs, ys, ws = [], [], []
    #     for _x, _u, _w in zip(x, u, gamma):
    #         xs.append(np.hstack((_x, np.ones((_x.shape[0], 1)))))
    #         ys.append(_u)
    #         ws.append(_w)
    #
    #     _J_diag = np.concatenate((reg * np.ones(self.dm_obs), reg * np.ones(1)))
    #     _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
    #     _h = np.zeros((self.nb_states, self.dm_obs + 1, self.dm_act))
    #
    #     # solving p = (xT w x)^-1 xT w y
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         for k in range(self.nb_states):
    #             wx = _x * _w[:, k:k + 1]
    #             _J[k] += np.dot(wx.T, _x)
    #             _h[k] += np.dot(wx.T, _y)
    #
    #     mu = np.linalg.solve(_J, _h)
    #     self.K = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2)
    #     self.kff = mu[:, -1, :]
    #
    #     sqerr = np.zeros((self.nb_states, self.dm_act, self.dm_act))
    #     weight = reg * np.ones(self.nb_states)
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         yhat = np.matmul(_x[None, :, :], mu)
    #         resid = _y[None, :, :] - yhat
    #         sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
    #         weight += np.sum(_w, axis=0)
    #
    #     _cov = sqerr / weight[:, None, None]
    #
    #     # usage = sum([_gamma.sum(0) for _gamma in gamma])
    #     # unused = np.where(usage < 1)[0]
    #     # used = np.where(usage > 1)[0]
    #     # if len(unused) > 0:
    #     #     for k in unused:
    #     #         i = npr.choice(used)
    #     #         self.K[k] = self.K[i] + 0.01 * npr.randn(*self.K[i].shape)
    #     #         self.kff[k] = self.kff[i] + 0.01 * npr.randn(*self.kff[i].shape)
    #     #         _cov[k] = _cov[i]
    #
    #     self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x), self.nb_states, self.dm_act))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x)
            mean.append(np.einsum('nk,nkl->nl', _gamma, _mu))
        return mean


class AutoregRessiveLinearGaussianControl:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-16):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.Kx = npr.randn(self.nb_states, self.dm_act, self.dm_obs)
        self.Ku = npr.randn(self.nb_states, self.dm_act, self.dm_act)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        self._sqrt_cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        if self.prior:
            for k in range(self.nb_states):
                _cov = sc.stats.invwishart.rvs(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_act))
                self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.dm_act))
        else:
            self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

    @property
    def params(self):
        return self.Kx, self.Ku, self.kff, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.Kx, self.Ku, self.kff, self._sqrt_cov = value

    def mean(self, z, x, u):
        return np.einsum('kh,...h->...k', self.Kx[z, ...], x) \
               + np.einsum('kh,...h->...k', self.Ku[z, ...], u) + self.kff[z, ...]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_act))

    def sample(self, z, x, u, stoch=True):
        if stoch:
            _u = mvn(mean=self.mean(z, x, u), cov=self.cov[z, ...]).rvs()
            return np.atleast_1d(_u)
        else:
            return self.mean(z, x, u)

    def reinit(self):
        self.Kx = npr.randn(self.nb_states, self.dm_act, self.dm_obs)
        self.Ku = npr.randn(self.nb_states, self.dm_act, self.dm_act)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        self._sqrt_cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        if self.prior:
            for k in range(self.nb_states):
                _cov = sc.stats.invwishart.rvs(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_act))
                self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.dm_act))
        else:
            self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

    def initialize(self, x, u, **kwargs):
        pass

    def permute(self, perm):
        self.Kx = self.Kx[perm, ...]
        self.Ku = self.Ku[perm, ...]
        self.kff = self.kff[perm, ...]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_u[1:], self.mean(k, x=_x[:-1], u=_u[:-1]), self.cov[k])
                                       for k in range(self.nb_states)])
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
            xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_u[1:])
            ws.append(_w[1:])

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False,
                                             **self.prior if use_prior else {})

            self.Kx[k, ...] = coef_[:, :self.dm_obs]
            self.Ku[k, ...] = coef_[:, self.dm_obs:self.dm_obs + self.dm_act]
            self.kff[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.Kx[k] = self.Kx[i] + 0.01 * npr.randn(*self.Kx[i].shape)
        #         self.Ku[k] = self.Ku[i] + 0.01 * npr.randn(*self.ku[i].shape)
        #         self.kff[k] = self.kff[i] + 0.01 * npr.randn(*self.kff[i].shape)
        #         _cov[k] = _cov[i]

        self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x), self.nb_states, self.dm_act))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x, _u)
            mean.append(np.einsum('nk,nkl->nl', _gamma, _mu))
        return mean

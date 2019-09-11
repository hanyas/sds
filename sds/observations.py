import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.stats import multivariate_normal as mvn

from sds.utils import random_rotation
from sds.utils import linear_regression

from sds.stats import multivariate_normal_logpdf


class GaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, reg=1e-8):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act
        self.reg = reg

        self.mu = npr.randn(self.nb_states, self.dm_obs)
        self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

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
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, z, x=None, u=None, stoch=True):
        if stoch:
            return mvn(mean=self.mean(z), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z)

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)
        if kmeans:
            from sklearn.cluster import KMeans
            _obs = np.concatenate(x)
            km = KMeans(self.nb_states).fit(_obs)

            self.mu = km.cluster_centers_
            self.cov = np.array([np.cov(_obs[km.labels_ == k].T)
                             for k in range(self.nb_states)])
        else:
            _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
            for k in range(self.nb_states):
                self.mu[k, :] = np.mean(np.vstack([_x[0, :] for _x in x]), axis=0)
                _cov[k, ...] = np.cov(np.vstack([_x[0, :] for _x in x]), rowvar=False)
            self.cov = _cov

    def permute(self, perm):
        self.mu = self.mu[perm]
        self._sqrt_cov = self._sqrt_cov[perm]

    def log_prior(self):
        return 0.0

    def log_likelihood(self, x, u):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([multivariate_normal_logpdf(_x, self.mean(k), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u):
        _J = np.zeros((self.nb_states, self.dm_obs))
        _h = np.zeros((self.nb_states, self.dm_obs))
        for _x, _w in zip(x, gamma):
            _J += np.sum(_w[:, :, None], axis=0)
            _h += np.sum(_w[:, :, None] * _x[:, None, :], axis=0)

        self.mu = _h / _J

        sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        weight = np.zeros((self.nb_states, ))
        for _x, _w in zip(x, gamma):
            resid = _x[:, None, :] - self.mu
            sqerr += np.sum(_w[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(_w, axis=0)

        self.cov = sqerr / weight[:, None, None]

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            mean.append(_gamma.dot(self.mu))
        return mean

class LinearGaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, reg=1e-8):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act
        self.reg = reg

        self.K = npr.randn(self.nb_states, self.dm_act, self.dm_obs)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

    @property
    def params(self):
        return self.K, self.kff, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.K, self.kff, self._sqrt_cov = value

    def mean(self, z, x):
        return np.einsum('kh,...h->...k', self.K[z, ...], x) + self.kff[z, ...]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_act))

    def sample(self, z, x, stoch=True):
        if stoch:
            return mvn(mean=self.mean(z, x), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z, x)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_u.shape[0] for _u in u]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states, random_state=1)
            km.fit((np.vstack(u)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [_x[t, :] for t, _x in zip(ts, x)]
            ys = [_u[t, :] for t, _u in zip(ts, u)]

            coef_, intercept_, sigma = linear_regression(xs, ys)
            self.K[k, ...] = coef_[:, :self.dm_act]
            self.kff[k, :] = intercept_
            _cov[k, ...] = sigma

        self.cov = _cov

    def permute(self, perm):
        self.K = self.K[perm, ...]
        self.kff = self.kff[perm, ...]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        return 0.0

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_u,  self.mean(k, x=_x), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            xs.append(np.hstack((_x, np.ones((_x.shape[0], 1)))))
            ys.append(_u)
            ws.append(_w)

        _J_diag = np.concatenate((self.reg * np.ones(self.dm_obs), self.reg * np.ones(1)))
        _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
        _h = np.zeros((self.nb_states, self.dm_obs + 1, self.dm_act))

        # solving p = (xT w x)^-1 xT w y
        for _x, _y, _w in zip(xs, ys, ws):
            for k in range(self.nb_states):
                wx = _x * _w[:, k:k + 1]
                _J[k] += np.dot(wx.T, _x)
                _h[k] += np.dot(wx.T, _y)

        mu = np.linalg.solve(_J, _h)
        self.K = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2)
        self.kff = mu[:, -1, :]

        sqerr = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        weight = self.reg * np.ones(self.nb_states)
        for _x, _y, _w in zip(xs, ys, ws):
            yhat = np.matmul(_x[None, :, :], mu)
            resid = _y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
            weight += np.sum(_w, axis=0)

        _cov = sqerr / weight[:, None, None]

        usage = sum([_gamma.sum(0) for _gamma in gamma])
        unused = np.where(usage < 1)[0]
        used = np.where(usage > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                self.K[k] = self.K[i] + 0.01 * npr.randn(*self.K[i].shape)
                self.kff[k] = self.kff[i] + 0.01 * npr.randn(*self.kff[i].shape)
                _cov[k] = _cov[i]

        self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x), self.nb_states, self.dm_act))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x)
            mean.append(np.einsum('nk,nkl->nl', _gamma, _mu))
        return mean


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act=0, reg=1e-8):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act
        self.reg = reg

        self.A = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        self.B = np.zeros((self.nb_states, self.dm_obs, self.dm_act))
        self.c = np.zeros((self.nb_states, self.dm_obs))

        for k in range(self.nb_states):
            self.A[k, ...] = .95 * random_rotation(self.dm_obs)
            self.B[k, ...] = npr.randn(self.dm_obs, self.dm_act)
            self.c[k, :] = npr.randn(self.dm_obs)

        self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

    @property
    def params(self):
        return self.A, self.B, self.c, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.A, self.B, self.c, self._sqrt_cov = value

    def mean(self, z, x, u):
        return np.einsum('kh,...h->...k', self.A[z, ...], x) +\
               np.einsum('kh,...h->...k', self.B[z, ...], u) + self.c[z, :]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, z, x, u, stoch=True):
        if stoch:
            return mvn(self.mean(z, x, u), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z, x, u)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states, random_state=1)
            km.fit((np.vstack(x)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [np.hstack((_x[t, :], _u[t, :])) for t, _x, _u in zip(ts, x, u)]
            ys = [_x[t + 1, :] for t, _x in zip(ts, x)]

            coef_, intercept_, sigma = linear_regression(xs, ys)
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
        return 0.0

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_x[1:, :], self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act]), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _J_diag = np.concatenate((self.reg * np.ones(self.dm_obs),
                                  self.reg * np.ones(self.dm_act),
                                  self.reg * np.ones(1)))
        _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
        _h = np.zeros((self.nb_states, self.dm_obs + self.dm_act + 1, self.dm_obs))

        for _x, _y, _w in zip(xs, ys, ws):
            for k in range(self.nb_states):
                wx = _x * _w[:, k:k + 1]
                _J[k] += np.dot(wx.T, _x)
                _h[k] += np.dot(wx.T, _y)

        mu = np.linalg.solve(_J, _h)
        self.A = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2)
        self.B = np.swapaxes(mu[:, self.dm_obs:self.dm_obs + self.dm_act, :], 1, 2)
        self.c = mu[:, -1, :]

        sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        weight = self.reg * np.ones(self.nb_states)
        for _x, _y, _w in zip(xs, ys, ws):
            yhat = np.matmul(_x[None, :, :], mu)
            resid = _y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
            weight += np.sum(_w, axis=0)

        _cov = sqerr / weight[:, None, None]

        usage = sum([_gamma.sum(0) for _gamma in gamma])
        unused = np.where(usage < 1)[0]
        used = np.where(usage > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
                self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
                self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
                _cov[k] = _cov[i]

        self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x) - 1, self.nb_states, self.dm_obs))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act])
            mean.append(np.einsum('nk,nkl->nl', _gamma[1:, ...], _mu))
        return mean

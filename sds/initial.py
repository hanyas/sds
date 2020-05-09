import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

import scipy as sc
from scipy import linalg, stats, special
from scipy.stats import multivariate_normal as mvn

from sds.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils import linear_regression

from sklearn.preprocessing import PolynomialFeatures


class CategoricalInitState:

    def __init__(self, nb_states, prior, reg=1e-8):
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

    def initialize(self):
        pass

    def sample(self):
        return npr.choice(self.nb_states, p=self.pi)

    def likeliest(self):
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

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1e-8):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.mu = npr.randn(self.nb_states, self.dm_obs)

        # self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

        self._sqrt_cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            _cov = sc.stats.invwishart.rvs(self.dm_obs + 1, np.eye(self.dm_obs))
            self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.dm_obs))

    @property
    def params(self):
        return self.mu, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_cov = value

    def mean(self, z):
        return self.mu[z, :]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, z):
        _x = mvn(mean=self.mean(z), cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_x)

    def initialize(self, x):
        from sklearn.cluster import KMeans
        _obs = np.concatenate(x)
        km = KMeans(self.nb_states).fit(_obs)

        self.mu = km.cluster_centers_
        self.cov = np.array([np.cov(_obs[km.labels_ == k].T)
                             for k in range(self.nb_states)])

    def permute(self, perm):
        self.mu = self.mu[perm, ...]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def log_likelihood(self, x):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([lg_mvn(_x[0], self.mean(k), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, weights=None, **kwargs):
        aux = []
        if weights:
            for _w, _gamma in zip(weights, gamma):
               aux.append(_w[:, None] * _gamma)
            gamma = aux

        _J = self.reg * np.ones((self.nb_states, self.dm_obs))
        _h = np.zeros((self.nb_states, self.dm_obs))
        for _x, _w in zip(x, gamma):
            _J += _w[0, :, None]
            _h += _w[0, :, None] * _x[0, None, :]

        self.mu = _h / _J

        sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        weight = self.reg * np.ones((self.nb_states, ))
        for _x, _w in zip(x, gamma):
            resid = _x[0, None, :] - self.mu
            sqerr += _w[0, :, None, None] * resid[:, None, :] * resid[:, :, None]
            weight += _w[0, ...]

        self.cov = sqerr / weight[:, None, None]

    def smooth(self, gamma, x):
        mean = []
        for _x, _gamma in zip(x, gamma):
            mean.append(np.einsum('k,kl->l', _gamma[0, ...], self.mu))
        return mean


class GaussianInitControl:

    def __init__(self, nb_states, dm_obs, dm_act,
                 prior, lags=1, degree=1, reg=1e-8):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.lags = lags

        self.prior = prior
        self.reg = reg

        self.degree = degree
        self.dm_feat = int(sc.special.comb(self.degree + self.dm_obs, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.K = npr.randn(self.nb_states, self.dm_act, self.dm_feat)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        # self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

        self._sqrt_cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            _cov = sc.stats.invwishart.rvs(self.dm_act + 1, np.eye(self.dm_act))
            self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.dm_act))

    @property
    def params(self):
        return self.K, self.kff, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.K, self.kff, self._sqrt_cov = value

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x)).squeeze()
        return feat

    def mean(self, z, x):
        feat = self.featurize(x)
        return np.einsum('kh,...h->...k', self.K[z, ...], feat) + self.kff[z, ...]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_act))

    def sample(self, z, x):
        _u = mvn(mean=self.mean(z, x), cov=self.cov[z, ...]).rvs()
        return np.atleast_1d(_u)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        feat = [self.featurize(_x) for _x in x]
        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(feat), np.vstack(u))))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=T) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [_feat[t, :] for t, _feat in zip(ts, feat)]
            ys = [_u[t, :] for t, _u in zip(ts, u)]

            coef_, intercept_, sigma = linear_regression(np.vstack(xs), np.vstack(ys),
                                                         weights=None, fit_intercept=True,
                                                         **self.prior)
            self.K[k, ...] = coef_[:, :self.dm_feat]
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
            pass
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([lg_mvn(_u[:self.lags], self.mean(k, x=_x[:self.lags]), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, weights=None, **kwargs):
        aux = []
        if weights:
            for _w, _gamma in zip(weights, gamma):
               aux.append(_w[:, None] * _gamma)
            gamma = aux

        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            _feat = self.featurize(_x)
            xs.append(np.hstack((_feat[:self.lags], np.ones((_feat[:self.lags].shape[0], 1)))))
            ys.append(_u[:self.lags])
            ws.append(_w[:self.lags])

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False)

            self.K[k, ...] = coef_[:, :self.dm_feat]
            self.kff[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        self.cov = _cov

    def smooth(self, gamma, x):
        mean = []
        for _x, _gamma in zip(x, gamma):
            _mu = np.zeros((self.nb_states, self.dm_act))
            for k in range(self.nb_states):
                _mu[k, :] = self.mean(k, _x)
            mean.append(np.einsum('k,kl->l', _gamma[self.lags], _mu))
        return mean

import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats
from scipy import special
from scipy.stats import multivariate_normal as mvn

from sds.utils.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils.general import linear_regression, one_hot, arstack
from sds.utils.decorate import ensure_args_are_viable

from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision

from sklearn.preprocessing import PolynomialFeatures

import copy


class LinearGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 degree=1, **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.degree = degree
        self.feat_dim = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.K = npr.randn(self.nb_states, self.act_dim, self.feat_dim)
        self.kff = npr.randn(self.nb_states, self.act_dim)
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.act_dim, self.act_dim)

    @property
    def sigma(self):
        return np.matmul(self._sigma_chol, np.swapaxes(self._sigma_chol, -1, -2))

    @sigma.setter
    def sigma(self, value):
        self._sigma_chol = np.linalg.cholesky(value + 1e-8 * np.eye(self.act_dim))

    @property
    def params(self):
        return self.K, self.kff, self._sigma_chol

    @params.setter
    def params(self, value):
        self.K, self.kff, self._sigma_chol = value

    def permute(self, perm):
        self.K = self.K[perm, ...]
        self.kff = self.kff[perm, ...]
        self._sigma_chol = self._sigma_chol[perm, ...]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        f = [self.featurize(_x) for _x in x]
        t = [_x.shape[0] for _x in x]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(f))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = [npr.choice(self.nb_states, size=_t) for _t in t]

        z = [one_hot(_z, self.nb_states) for _z in z]
        self.mstep(z, x, u)

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat)

    def mean(self, z, x):
        feat = self.featurize(x)
        return np.einsum('kh,...h->...k', self.K[z, ...], feat) + self.kff[z, ...]

    def sample(self, z, x):
        u = mvn(mean=self.mean(z, x), cov=self.sigma[z, ...]).rvs()
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            loglik = np.zeros((x.shape[0], self.nb_states))
            for k in range(self.nb_states):
                loglik[:, k] = lg_mvn(u, self.mean(k, x), self.sigma[k])
            return loglik
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        fs, us, ws = [], [], []
        for _x, _u, _w in zip(x, u, p):
            fs.append(self.featurize(_x))
            us.append(_u)
            ws.append(_w)

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(fs), ys=np.vstack(us),
                                                       weights=np.vstack(ws)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)
            self.K[k, ...] = coef
            self.kff[k, ...] = intercept
            _sigma[k, ...] = sigma

        self.sigma = _sigma

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x)
            return np.einsum('nk,nkl->nl', p, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianLinearGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 prior, degree=1, likelihood=None):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.degree = degree

        self.feat_dim = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.input_dim = self.feat_dim + 1
        self.output_dim = self.act_dim

        self.prior = prior
        self.posterior = copy.deepcopy(prior)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbdas = self.prior.rvs()
            self.likelihood = StackedLinearGaussiansWithPrecision(size=self.nb_states,
                                                                  column_dim=self.input_dim,
                                                                  row_dim=self.output_dim,
                                                                  As=As, lmbdas=lmbdas, affine=True)

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]
        self.likelihood.lambdas = self.likelihood.lambdas[perm]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', False)

        f = [self.featurize(_x) for _x in x]
        t = [_x.shape[0] for _x in x]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(f))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = [npr.choice(self.nb_states, size=_t) for _t in t]

        z = [one_hot(_z, self.nb_states) for _z in z]

        stats = self.likelihood.weighted_statistics(f, u, z)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat)

    def mean(self, z, x):
        feat = self.featurize(x)
        return self.likelihood.dists[z].mean(feat)

    def sample(self, z, x):
        feat = self.featurize(x)
        u = self.likelihood.dists[z].rvs(feat)
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            f = self.featurize(x)
            return self.likelihood.log_likelihood(f, u)
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        f = [self.featurize(_x) for _x in x]

        method = kwargs.get('method', 'direct')
        if method == 'direct':
            stats = self.likelihood.weighted_statistics(f, u, p)
            self.posterior.nat_param = self.prior.nat_param + stats
        elif method == 'sgd':
            from sds.utils.general import batches

            lr = kwargs.get('lr', 1e-3)
            nb_iter = kwargs.get('nb_iter', 100)
            batch_size = kwargs.get('batch_size', 64)

            f, u, p = list(map(np.vstack, (f, u, p)))

            set_size = len(f)
            prob = float(batch_size / set_size)
            for _ in range(nb_iter):
                for batch in batches(batch_size, set_size):
                    stats = self.likelihood.weighted_statistics(f[batch], u[batch], p[batch])
                    self.posterior.nat_param = (1. - lr) * self.posterior.nat_param\
                                               + lr * (self.prior.nat_param + 1. / prob * stats)
        else:
            raise NotImplementedError

        self.prior.nat_param = (1. - 1e-3) * self.prior.nat_param\
                               + 1e-3 * self.posterior.nat_param

        self.likelihood.params = self.posterior.mode()

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x)
            return np.einsum('nk,nkl->nl', p, mu)
        else:
            return list(map(self.smooth, p, x, u))


class AutorRegressiveLinearGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags=1, degree=1, reg=1e-8):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.reg = reg

        self.degree = degree
        self.feat_dim = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.K = npr.randn(self.nb_states, self.act_dim, self.feat_dim * (self.nb_lags + 1))
        self.kff = npr.randn(self.nb_states, self.act_dim)
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.act_dim, self.act_dim)

    @property
    def sigma(self):
        return np.matmul(self._sigma_chol, np.swapaxes(self._sigma_chol, -1, -2))

    @sigma.setter
    def sigma(self, value):
        self._sigma_chol = np.linalg.cholesky(value + self.reg * np.eye(self.act_dim))

    @property
    def params(self):
        return self.K, self.kff, self._sigma_chol

    @params.setter
    def params(self, value):
        self.K, self.kff, self._sigma_chol = value

    def permute(self, perm):
        self.K = self.K[perm, ...]
        self.kff = self.kff[perm, ...]
        self._sigma_chol = self._sigma_chol[perm, ...]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        ts = [_x.shape[0] for _x in x]
        fs = [self.featurize(_x) for _x in x]

        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(fs), np.vstack(u))))
            zs = np.split(km.labels_, np.cumsum(ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=t) for t in ts]

        zs = [one_hot(z, self.nb_states) for z in zs]
        self.mstep(zs, x, u)

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat)

    def mean(self, z, x):
        xr = np.reshape(x, (-1, self.obs_dim * (self.nb_lags + 1)))
        feat = self.featurize(xr)
        return np.einsum('kh,...h->...k', self.K[z, ...], feat) + self.kff[z, ...]

    def sample(self, z, x):
        u = mvn(mean=self.mean(z, x), cov=self.sigma[z, ...]).rvs()
        return np.atleast_1d(u)

    def log_likelihood(self, x, u):
        xr, ur = [], []
        for _x, _u in zip(x, u):
            xr.append(arstack(_x, self.nb_lags ))
            ur.append(_u[self.nb_lags:])

        loglik = []
        for _xr, _ur in zip(xr, ur):
            _loglik = np.zeros((_xr.shape[0], self.nb_states))
            for k in range(self.nb_states):
                _mu = self.mean(k, _xr)
                _loglik[:, k] = lg_mvn(_ur, _mu, self.sigma[k])
            loglik.append(_loglik)

        return loglik

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        fs, us, ws = [], [], []
        for _x, _u, _w in zip(x, u, p):
            _xr = arstack(_x, self.nb_lags)
            fs.append(self.featurize(_xr))
            us.append(_u[self.nb_lags:])
            ws.append(_w[self.nb_lags:])

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(fs), ys=np.vstack(us),
                                                       weights=np.vstack(ws)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.K[k, ...] = coef
            self.kff[k, ...] = intercept
            _sigma[k, ...] = sigma

        self.sigma = _sigma

    def smooth(self, p, x, u):
        xr, ur, pr = [], [], []
        for _x, _u, _pr in zip(x, u, p):
            xr.append(arstack(_x, self.nb_lags))
            ur.append(_u[self.nb_lags:])
            pr.append(_pr[self.nb_lags:])

        mean = []
        for _xr, _ur, _pr in zip(xr, ur, pr):
            _mu = np.zeros((len(_xr), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _xr)
            mean.append(np.einsum('nk,nkl->nl', _pr, _mu))

        return mean

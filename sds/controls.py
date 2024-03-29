import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats
from scipy.special import comb
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart

from sds.utils.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils.general import linear_regression, one_hot, arstack
from sds.utils.decorate import ensure_args_are_viable

from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision
from sds.distributions.composite import StackedMultiOutputLinearGaussianWithAutomaticRelevance

from sklearn.preprocessing import PolynomialFeatures

from functools import partial
from operator import mul
import copy


class GaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 degree=1, **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.degree = degree
        self.feat_dim = int(comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        # self.K = npr.randn(self.nb_states, self.act_dim, self.feat_dim)
        # self.kff = npr.randn(self.nb_states, self.act_dim)
        # self._sigma_chol = 5. * npr.randn(self.nb_states, self.act_dim, self.act_dim)

        self.K = np.zeros((self.nb_states, self.act_dim, self.feat_dim))
        self.kff = np.zeros((self.nb_states, self.act_dim))
        self._sigma_chol = np.zeros((self.nb_states, self.act_dim, self.act_dim))

        for k in range(self.nb_states):
            _sigma = invwishart.rvs(self.act_dim + 1, np.eye(self.act_dim))
            self._sigma_chol[k] = np.linalg.cholesky(_sigma * np.eye(self.act_dim))
            self.K[k] = mvn.rvs(mean=None, cov=1e2 * _sigma, size=(self.feat_dim, )).T
            self.kff[k] = mvn.rvs(mean=None, cov=1e2 * _sigma, size=(1, ))

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
        self.K = self.K[perm]
        self.kff = self.kff[perm]
        self._sigma_chol = self._sigma_chol[perm]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        f = list(map(self.featurize, x))
        t = list(map(len, x))
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(f))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = list(map(partial(npr.choice, self.nb_states), t))

        z = list(map(partial(one_hot, self.nb_states), z))
        self.mstep(z, x, u)

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x):
        feat = self.featurize(x)
        u = np.einsum('kh,...h->...k', self.K[z], feat) + self.kff[z]
        return np.atleast_1d(u)

    def sample(self, z, x):
        u = mvn(mean=self.mean(z, x), cov=self.sigma[z]).rvs()
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            log_lik = np.zeros((x.shape[0], self.nb_states))
            for k in range(self.nb_states):
                log_lik[:, k] = lg_mvn(u, self.mean(k, x), self.sigma[k])
            return log_lik
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        f = [self.featurize(_x) for _x in x]

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(f), ys=np.vstack(u),
                                                       weights=np.vstack(p)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)
            self.K[k, ...] = coef
            self.kff[k, ...] = intercept
            _sigma[k, ...] = sigma

        self.sigma = _sigma

    def weighted_mstep(self, p, x, u, w, **kwargs):
        assert isinstance(w, list)
        wp = list(map(mul, w, p))
        self.mstep(wp, x, u, **kwargs)

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x)
            return np.einsum('nk,nkl->nl', p, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianGaussianControl:

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

        f = list(map(self.featurize, x))
        t = list(map(len, x))
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(f))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = list(map(partial(npr.choice, self.nb_states), t))

        z = list(map(partial(one_hot, self.nb_states), z))

        stats = self.likelihood.weighted_statistics(f, u, z)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x):
        feat = self.featurize(x)
        u = self.likelihood.dists[z].mean(feat)
        return np.atleast_1d(u)

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
        f = list(map(self.featurize, x))

        stats = self.likelihood.weighted_statistics(f, u, p)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.mode()

    def weighted_mstep(self, p, x, u, w, **kwargs):
        assert isinstance(w, list)
        wp = list(map(mul, w, p))
        self.mstep(wp, x, u, **kwargs)

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x)
            return np.einsum('nk,nkl->nl', p, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianGaussianControlWithAutomaticRelevance\
            (StackedMultiOutputLinearGaussianWithAutomaticRelevance):

    def __init__(self, nb_states, obs_dim, act_dim, prior, degree=1):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.degree = degree

        self.feat_dim = int(comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.input_dim = self.feat_dim + 1
        self.output_dim = self.act_dim

        likelihood_precision_prior = prior['likelihood_precision_prior']
        parameter_precision_prior = prior['parameter_precision_prior']

        super().__init__(self.nb_states,
                         self.input_dim,
                         self.output_dim,
                         likelihood_precision_prior,
                         parameter_precision_prior)

    def permute(self, perm):
        self.As = self.As[perm]
        self.lmbdas = self.lmbdas[perm]

    def initialize(self, x, u, **kwargs):
        pass

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x):
        feat = self.featurize(x)
        u = super().mean(z, feat)
        return np.atleast_1d(u)

    def sample(self, z, x):
        feat = self.featurize(x)
        u = self.rvs(z, feat)
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            f = self.featurize(x)
            return super().log_likelihood(f, u)
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        f = list(map(self.featurize, x))
        fs, us, ps = list(map(np.vstack, (f, u, p)))
        self.em(fs, us, ps, **kwargs)

    def weighted_mstep(self, p, x, u, w, **kwargs):
        assert isinstance(w, list)
        wp = list(map(mul, w, p))
        self.mstep(wp, x, u, **kwargs)

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x)
            return np.einsum('nk,nkl->nl', p, mu)
        else:
            return list(map(self.smooth, p, x, u))


class AutorRegressiveGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags=1, degree=1, **kwargs):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.degree = degree
        self.feat_dim = int(comb(self.degree + (self.obs_dim * (self.nb_lags + 1)), self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        # self.K = npr.randn(self.nb_states, self.act_dim, self.feat_dim)
        # self.kff = npr.randn(self.nb_states, self.act_dim)
        # self._sigma_chol = 5. * npr.randn(self.nb_states, self.act_dim, self.act_dim)

        self.K = np.zeros((self.nb_states, self.act_dim, self.feat_dim))
        self.kff = np.zeros((self.nb_states, self.act_dim))
        self._sigma_chol = np.zeros((self.nb_states, self.act_dim, self.act_dim))

        for k in range(self.nb_states):
            _sigma = invwishart.rvs(self.act_dim + 1, np.eye(self.act_dim))
            self._sigma_chol[k] = np.linalg.cholesky(_sigma * np.eye(self.act_dim))
            self.K[k] = mvn.rvs(mean=None, cov=1e2 * _sigma, size=(self.feat_dim, )).T
            self.kff[k] = mvn.rvs(mean=None, cov=1e2 * _sigma, size=(1, ))

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
        self.K = self.K[perm]
        self.kff = self.kff[perm]
        self._sigma_chol = self._sigma_chol[perm]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        xr, ur = [], []
        for _x, _u in zip(x, u):
            xr.append(arstack(_x, self.nb_lags + 1))
            ur.append(_u[self.nb_lags:])
        fr = list(map(self.featurize, xr))

        t = list(map(len, fr))
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(fr))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = list(map(partial(npr.choice, self.nb_states), t))

        z = list(map(partial(one_hot, self.nb_states), z))

        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.obs_dim + 1)

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(fr), ys=np.vstack(ur),
                                                       weights=np.vstack(z)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.K[k] = coef
            self.kff[k] = intercept
            _sigma[k] = sigma

        self.sigma = _sigma

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags + 1), axis=0) if ar else x
        feat = self.featurize(xr)
        u = np.einsum('kh,...h->...k', self.K[z], feat) + self.kff[z]
        return np.atleast_1d(u)

    def sample(self, z, x):
        u = mvn(mean=self.mean(z, x), cov=self.sigma[z]).rvs()
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            xr = arstack(x, self.nb_lags + 1)
            ur = u[self.nb_lags:]

            log_lik = np.zeros((ur.shape[0], self.nb_states))
            for k in range(self.nb_states):
                log_lik[:, k] = lg_mvn(ur, self.mean(k, xr), self.sigma[k])
            return log_lik
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        xr, ur, wr = [], [], []
        for _x, _u, _w in zip(x, u, p):
            xr.append(arstack(_x, self.nb_lags + 1))
            ur.append(_u[self.nb_lags:])
            wr.append(_w[self.nb_lags:])
        fr = list(map(self.featurize, xr))

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(fr), ys=np.vstack(ur),
                                                       weights=np.vstack(wr)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.K[k] = coef
            self.kff[k] = intercept
            _sigma[k] = sigma

        self.sigma = _sigma

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            xr = arstack(x, self.nb_lags + 1)
            ur = u[self.nb_lags:]
            pr = p[self.nb_lags:]

            mu = np.zeros((len(ur), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, xr)
            return np.einsum('nk,nkl->nl', pr, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianAutorRegressiveGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, degree=1, likelihood=None):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.degree = degree
        self.feat_dim = int(comb(self.degree + (self.obs_dim * (self.nb_lags + 1)), self.degree)) - 1
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
        self.likelihood.lmbdas = self.likelihood.lmbdas[perm]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', False)

        xr, ur = [], []
        for _x, _u in zip(x, u):
            xr.append(arstack(_x, self.nb_lags + 1))
            ur.append(_u[self.nb_lags:])
        fr = list(map(self.featurize, xr))

        t = list(map(len, fr))
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(fr))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = list(map(partial(npr.choice, self.nb_states), t))

        z = list(map(partial(one_hot, self.nb_states), z))

        stats = self.likelihood.weighted_statistics(fr, ur, z)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags + 1), axis=0) if ar else x
        fr = self.featurize(xr)
        u = self.likelihood.dists[z].mean(fr)
        return np.atleast_1d(u)

    def sample(self, z, x, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags + 1), axis=0) if ar else x
        fr = self.featurize(xr)
        u = self.likelihood.dists[z].rvs(fr)
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            xr = arstack(x, self.nb_lags + 1)
            ur = u[self.nb_lags:]
            fr = self.featurize(xr)
            return self.likelihood.log_likelihood(fr, ur)
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        xr, ur, wr = [], [], []
        for _x, _u, _w in zip(x, u, p):
            xr.append(arstack(_x, self.nb_lags + 1))
            ur.append(_u[self.nb_lags:])
            wr.append(_w[self.nb_lags:])
        fr = list(map(self.featurize, xr))

        stats = self.likelihood.weighted_statistics(fr, ur, wr)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.mode()

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            xr = arstack(x, self.nb_lags + 1)
            ur = u[self.nb_lags:]
            pr = p[self.nb_lags:]

            mu = np.zeros((len(ur), self.nb_states, self.obs_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, xr)
            return np.einsum('nk,nkl->nl', pr, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianAutoRegressiveGaussianControlWithAutomaticRelevance\
            (StackedMultiOutputLinearGaussianWithAutomaticRelevance):

    def __init__(self, nb_states, obs_dim, act_dim, nb_lags, prior, degree=1):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.degree = degree
        self.feat_dim = int(comb(self.degree + (self.obs_dim * (self.nb_lags + 1)), self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.input_dim = self.feat_dim + 1
        self.output_dim = self.act_dim

        likelihood_precision_prior = prior['likelihood_precision_prior']
        parameter_precision_prior = prior['parameter_precision_prior']

        super().__init__(self.nb_states,
                         self.input_dim,
                         self.output_dim,
                         likelihood_precision_prior,
                         parameter_precision_prior)

    def permute(self, perm):
        self.As = self.As[perm]
        self.lmbdas = self.lmbdas[perm]

    def initialize(self, x, u, **kwargs):
        pass

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags + 1), axis=0) if ar else x
        fr = self.featurize(xr)
        u = super().mean(z, fr)
        return np.atleast_1d(u)

    def sample(self, z, x, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags + 1), axis=0) if ar else x
        fr = self.featurize(xr)
        u = self.rvs(z, fr)
        return np.atleast_1d(u)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            xr = arstack(x, self.nb_lags + 1)
            ur = u[self.nb_lags:]
            fr = self.featurize(xr)
            return super().log_likelihood(fr, ur)
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        xr, ur, wr = [], [], []
        for _x, _u, _w in zip(x, u, p):
            xr.append(arstack(_x, self.nb_lags + 1))
            ur.append(_u[self.nb_lags:])
            wr.append(_w[self.nb_lags:])
        fr = list(map(self.featurize, xr))

        fr, ur, wr = list(map(np.vstack, (fr, ur, wr)))
        self.em(fr, ur, wr, **kwargs)

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            xr = arstack(x, self.nb_lags + 1)
            ur = u[self.nb_lags:]
            pr = p[self.nb_lags:]

            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, xr)
            return np.einsum('nk,nkl->nl', pr, mu)
        else:
            return list(map(self.smooth, p, x, u))

import numpy as np
import numpy.random as npr

import scipy as sc
from scipy.stats import multivariate_normal as mvn

from sds.utils.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils.general import linear_regression, one_hot, arstack
from sds.utils.decorate import ensure_args_are_viable

from sds.distributions.gaussian import StackedGaussiansWithPrecision

from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision
from sds.distributions.lingauss import StackedLinearGaussiansWithDiagonalPrecision

from sds.distributions.lingauss import TiedLinearGaussiansWithPrecision
from sds.distributions.lingauss import TiedLinearGaussiansWithDiagonalPrecision

import copy


class GaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim, **kwargs):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.mu = npr.randn(self.nb_states, self.obs_dim)
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.obs_dim, self.obs_dim)

    @property
    def sigma(self):
        return np.matmul(self._sigma_chol, np.swapaxes(self._sigma_chol, -1, -2))

    @sigma.setter
    def sigma(self, value):
        self._sigma_chol = np.linalg.cholesky(value + 1e-8 * np.eye(self.obs_dim))

    @property
    def params(self):
        return self.mu, self._sigma_chol

    @params.setter
    def params(self, value):
        self.mu, self._sigma_chol = value

    def permute(self, perm):
        self.mu = self.mu[perm]
        self._sigma_chol = self._sigma_chol[perm]

    def initialize(self, x, u=None, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        t = [_x.shape[0] for _x in x]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(x))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = [npr.choice(self.nb_states, size=_t) for _t in t]

        z = [one_hot(_z, self.nb_states) for _z in z]
        self.mstep(z, x, u)

    def mean(self, z, x=None, u=None):
        return self.mu[z]

    def sample(self, z, x=None, u=None):
        x = mvn(mean=self.mean(z), cov=self.sigma[z]).rvs()
        return np.atleast_1d(x)

    @ensure_args_are_viable
    def log_likelihood(self, x, u=None):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            loglik = np.zeros((x.shape[0], self.nb_states))
            for k in range(self.nb_states):
                loglik[:, k] = lg_mvn(x, self.mean(k), self.sigma[k])
            return loglik
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        J = np.zeros((self.nb_states, self.obs_dim))
        h = np.zeros((self.nb_states, self.obs_dim))
        for _x, _p in zip(x, p):
            J += np.sum(_p[:, :, None], axis=0)
            h += np.sum(_p[:, :, None] * _x[:, None, :], axis=0)

        self.mu = h / J

        sqerr = np.zeros((self.nb_states, self.obs_dim, self.obs_dim))
        norm = np.zeros((self.nb_states, ))
        for _x, _p in zip(x, p):
            resid = _x[:, None, :] - self.mu
            sqerr += np.sum(_p[:, :, None, None] * resid[:, :, None, :]
                            * resid[:, :, :, None], axis=0)
            norm += np.sum(_p, axis=0)

        self.sigma = sqerr / norm[:, None, None]

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            return p.dot(self.mu)
        else:
            return list(map(self.smooth, p, x, u))


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim, nb_lags=1, **kwargs):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.A = npr.randn(self.nb_states, self.obs_dim, self.obs_dim * self.nb_lags)
        self.B = npr.randn(self.nb_states, self.obs_dim, self.act_dim)
        self.c = npr.randn(self.nb_states, self.obs_dim, )
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.obs_dim, self.obs_dim)

    @property
    def sigma(self):
        return np.matmul(self._sigma_chol, np.swapaxes(self._sigma_chol, -1, -2))

    @sigma.setter
    def sigma(self, value):
        self._sigma_chol = np.linalg.cholesky(value + 1e-8 * np.eye(self.obs_dim))

    @property
    def params(self):
        return self.A, self.B, self.c, self._sigma_chol

    @params.setter
    def params(self, value):
        self.A, self.B, self.c, self._sigma_chol = value

    def permute(self, perm):
        self.A = self.A[perm]
        self.B = self.B[perm]
        self.c = self.c[perm]
        self._sigma_chol = self._sigma_chol[perm]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        xr, ur, xn = [], [], []
        for _x, _u in zip(x, u):
            xr.append(arstack(_x, self.nb_lags)[:-1])
            ur.append(_u[self.nb_lags - 1:-1])
            xn.append(_x[self.nb_lags:])
        xu = list(map(np.hstack, zip(xr, ur)))

        t = [_xu.shape[0] for _xu in xu]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(xu))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = [npr.choice(self.nb_states, size=_t) for _t in t]

        z = [one_hot(_z, self.nb_states) for _z in z]

        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.obs_dim + 1)

        _sigma = np.zeros((self.nb_states, self.obs_dim, self.obs_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(xu), ys=np.vstack(xn),
                                                       weights=np.vstack(z)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.A[k] = coef[:, :self.obs_dim * self.nb_lags]
            self.B[k] = coef[:, self.obs_dim * self.nb_lags:]
            self.c[k] = intercept
            _sigma[k] = sigma

        self.sigma = _sigma

    def mean(self, z, x, u, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags), axis=0) if ar else x
        return np.einsum('kh,...h->...k', self.A[z], xr) +\
               np.einsum('kh,...h->...k', self.B[z], u) + self.c[z, :]

    def sample(self, z, x, u, ar=False):
        xn = mvn(self.mean(z, x, u, ar), cov=self.sigma[z]).rvs()
        return np.atleast_1d(xn)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            xr = arstack(x, self.nb_lags)[:-1]
            ur = u[self.nb_lags - 1:-1]
            xn = x[self.nb_lags:]

            loglik = np.zeros((xr.shape[0], self.nb_states))
            for k in range(self.nb_states):
                mu = self.mean(k, xr, ur)
                loglik[:, k] = lg_mvn(xn, mu, self.sigma[k])
            return loglik
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.obs_dim + 1)

        xr, ur, xn, w = [], [], [], []
        for _x, _u, _w in zip(x, u, p):
            xr.append(arstack(_x, self.nb_lags)[:-1])
            ur.append(_u[self.nb_lags - 1:-1])
            xn.append(_x[self.nb_lags:])
            w.append(_w[self.nb_lags:])
        xu = list(map(np.hstack, zip(xr, ur)))

        _sigma = np.zeros((self.nb_states, self.obs_dim, self.obs_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(xu), ys=np.vstack(xn),
                                                       weights=np.vstack(w)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.A[k] = coef[:, :self.obs_dim * self.nb_lags]
            self.B[k] = coef[:, self.obs_dim * self.nb_lags:]
            self.c[k] = intercept
            _sigma[k] = sigma

        self.sigma = _sigma

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            xr = arstack(x, self.nb_lags)[:-1]
            ur = u[self.nb_lags - 1:-1]
            pr = p[self.nb_lags:]

            mu = np.zeros((len(xr), self.nb_states, self.obs_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, xr, ur)
            return np.einsum('nk,nkl->nl', pr, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianGaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim,
                 prior, likelihood=None):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Normal-Wishart conjugate
        self.prior = prior

        # Normal-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mus, lmbdas = self.prior.rvs()
            self.likelihood = StackedGaussiansWithPrecision(size=self.nb_states,
                                                            dim=self.obs_dim,
                                                            mus=mus, lmbdas=lmbdas)

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def permute(self, perm):
        self.likelihood.mus = self.likelihood.mus[perm]
        self.likelihood.lmbdas = self.likelihood.lmbdas[perm]

    def initialize(self, x, u=None, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        t = [_x.shape[0] for _x in x]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(x))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = [npr.choice(self.nb_states, size=_t) for _t in t]

        z = [one_hot(_z, self.nb_states) for _z in z]

        stats = self.likelihood.weighted_statistics(x, z)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def mean(self, z, x=None, u=None):
        return self.likelihood.dists[z].mean()

    def sample(self, z, x=None, u=None):
        x = self.likelihood.dists[z].rvs()
        return np.atleast_1d(x)

    @ensure_args_are_viable
    def log_likelihood(self, x, u=None):
        return self.likelihood.log_likelihood(x)

    def mstep(self, p, x, u, **kwargs):
        stats = self.likelihood.weighted_statistics(x, p)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.mode()

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            return p.dot(self.likelihood.mus)
        else:
            return list(map(self.smooth, p, x, u))


class _BayesianAutoRegressiveObservationBase:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.input_dim = self.obs_dim * self.nb_lags\
                         + self.act_dim + 1
        self.output_dim = self.obs_dim

        self.prior = prior
        self.posterior = copy.deepcopy(prior)
        self.likelihood = likelihood

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        xr, ur, xn = [], [], []
        for _x, _u in zip(x, u):
            xr.append(arstack(_x, self.nb_lags)[:-1])
            ur.append(_u[self.nb_lags - 1:-1])
            xn.append(_x[self.nb_lags:])
        xu = list(map(np.hstack, zip(xr, ur)))

        t = [_xu.shape[0] for _xu in xu]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(xu))
            z = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z = [npr.choice(self.nb_states, size=_t) for _t in t]

        z = [one_hot(_z, self.nb_states) for _z in z]

        stats = self.likelihood.weighted_statistics(xu, xn, z)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def permute(self, perm):
        raise NotImplementedError

    def mean(self, z, x, u, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags), axis=0) if ar else x
        xu = np.hstack((xr, u))
        return self.likelihood.dists[z].mean(xu)

    def sample(self, z, x, u, ar=False):
        xr = np.squeeze(arstack(x, self.nb_lags), axis=0) if ar else x
        xu = np.hstack((xr, u))
        xn = self.likelihood.dists[z].rvs(xu)
        return np.atleast_1d(xn)

    @ensure_args_are_viable
    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            xr = arstack(x, self.nb_lags)[:-1]
            ur = u[self.nb_lags - 1:-1]
            xn = x[self.nb_lags:]
            xu = np.hstack((xr, ur))
            return self.likelihood.log_likelihood(xu, xn)
        else:
            def inner(x, u):
                return self.log_likelihood.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):

        xr, ur, xn, w = [], [], [], []
        for _x, _u, _w in zip(x, u, p):
            xr.append(arstack(_x, self.nb_lags)[:-1])
            ur.append(_u[self.nb_lags - 1:-1])
            xn.append(_x[self.nb_lags:])
            w.append(_w[self.nb_lags:])
        xu = list(map(np.hstack, zip(xr, ur)))

        method = kwargs.get('method', 'direct')
        if method == 'direct':
            stats = self.likelihood.weighted_statistics(xu, xn, w)
            self.posterior.nat_param = self.prior.nat_param + stats
        elif method == 'sgd':
            from sds.utils.general import batches

            lr = kwargs.get('lr', 1e-3)
            nb_iter = kwargs.get('nb_iter', 100)
            batch_size = kwargs.get('batch_size', 64)

            xu, xn, w = list(map(np.vstack, (xu, xn, w)))

            set_size = len(xu)
            prob = float(batch_size / set_size)
            for _ in range(nb_iter):
                for batch in batches(batch_size, set_size):
                    stats = self.likelihood.weighted_statistics(xu[batch], xn[batch], w[batch])
                    self.posterior.nat_param = (1. - lr) * self.posterior.nat_param\
                                               + lr * (self.prior.nat_param + 1. / prob * stats)
        else:
            raise NotImplementedError

        # self.prior.nat_param = (1. - 1e-3) * self.prior.nat_param\
        #                        + 1e-3 * self.posterior.nat_param

        self.likelihood.params = self.posterior.mode()

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            xr = arstack(x, self.nb_lags)[:-1]
            ur = u[self.nb_lags - 1:-1]
            pr = p[self.nb_lags:]

            mu = np.zeros((len(xr), self.nb_states, self.obs_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, xr, ur)
            return np.einsum('nk,nkl->nl', pr, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianAutoRegressiveGaussianObservation(_BayesianAutoRegressiveObservationBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-6 * np.eye(input_dim)
    # psi = 1e8 * np.eye(obs_dim) / (obs_dim + 1)
    # nu = (obs_dim + 1) + obs_dim + 1
    #
    # from sds.distributions.composite import StackedMatrixNormalWishart
    # prior = StackedMatrixNormalWishart(nb_states, input_dim, output_dim,
    #                                    Ms=np.array([M for _ in range(nb_states)]),
    #                                    Ks=np.array([K for _ in range(nb_states)]),
    #                                    psis=np.array([psi for _ in range(nb_states)]),
    #                                    nus=np.array([nu for _ in range(nb_states)]))

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianAutoRegressiveGaussianObservation, self).__init__(nb_states, obs_dim, act_dim,
                                                                        nb_lags, prior, likelihood)
        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbdas = self.prior.rvs()
            self.likelihood = StackedLinearGaussiansWithPrecision(size=self.nb_states,
                                                                  column_dim=self.input_dim,
                                                                  row_dim=self.output_dim,
                                                                  As=As, lmbdas=lmbdas, affine=True)

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]
        self.likelihood.lmbdas = self.likelihood.lmbdas[perm]


class BayesianAutoRegressiveTiedGaussianObservation(_BayesianAutoRegressiveObservationBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-64 * np.eye(input_dim)
    # psi = 1e16 * np.eye(obs_dim) / (obs_dim + 1)
    # nu = (obs_dim + 1) + obs_dim + 1
    #
    # from sds.distributions.composite import TiedMatrixNormalWishart
    # prior = TiedMatrixNormalWishart(nb_states, input_dim, output_dim,
    #                                 Ms=np.array([M for _ in range(nb_states)]),
    #                                 Ks=np.array([K for _ in range(nb_states)]),
    #                                 psi=psi, nu=nu)

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianAutoRegressiveTiedGaussianObservation, self).__init__(nb_states, obs_dim, act_dim,
                                                                            nb_lags, prior, likelihood)

        # Tied Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbda = self.prior.rvs()
            self.likelihood = TiedLinearGaussiansWithPrecision(size=self.nb_states,
                                                               column_dim=self.input_dim,
                                                               row_dim=self.output_dim,
                                                               As=As, lmbda=lmbda, affine=True)

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]


class BayesianAutoRegressiveDiagonalGaussianObservation(_BayesianAutoRegressiveObservationBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-64 * np.eye(input_dim)
    # alpha = ((obs_dim + 1) + obs_dim + 1) / 2. * np.ones((obs_dim,))
    # beta = 1. / (2. * 1e16 * np.ones((obs_dim,)) / (obs_dim + 1))
    #
    # from sds.distributions.composite import StackedMatrixNormalGamma
    # prior = StackedMatrixNormalGamma(nb_states, input_dim, output_dim,
    #                                  Ms=np.array([M for _ in range(nb_states)]),
    #                                  Ks=np.array([K for _ in range(nb_states)]),
    #                                  alphas=np.array([alpha for _ in range(nb_states)]),
    #                                  betas=np.array([beta for _ in range(nb_states)]))

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianAutoRegressiveDiagonalGaussianObservation, self).__init__(nb_states, obs_dim, act_dim,
                                                                                nb_lags, prior, likelihood)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbdas_diag = self.prior.rvs()
            self.likelihood = StackedLinearGaussiansWithDiagonalPrecision(size=self.nb_states,
                                                                          column_dim=self.input_dim,
                                                                          row_dim=self.output_dim,
                                                                          As=As, lmbdas_diag=lmbdas_diag,
                                                                          affine=True)

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]
        self.likelihood.lmbdas_diag = self.likelihood.lmbdas_diag[perm]


class BayesianAutoRegressiveTiedDiagonalGaussianObservation(_BayesianAutoRegressiveObservationBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-64 * np.eye(input_dim)
    # alpha = ((obs_dim + 1) + obs_dim + 1) / 2. * np.ones((obs_dim,))
    # beta = 1. / (2. * 1e8 * np.ones((obs_dim,)) / (obs_dim + 1))
    #
    # from sds.distributions.composite import TiedMatrixNormalGamma
    # prior = TiedMatrixNormalGamma(nb_states, input_dim, output_dim,
    #                               Ms=np.array([M for _ in range(nb_states)]),
    #                               Ks=np.array([K for _ in range(nb_states)]),
    #                               alphas=alpha, betas=beta)

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianAutoRegressiveTiedDiagonalGaussianObservation, self).__init__(nb_states, obs_dim, act_dim,
                                                                                    nb_lags, prior, likelihood)

        # Tied Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbda_diag = self.prior.rvs()
            self.likelihood = TiedLinearGaussiansWithDiagonalPrecision(size=self.nb_states,
                                                                       column_dim=self.input_dim,
                                                                       row_dim=self.output_dim,
                                                                       As=As, lmbda_diag=lmbda_diag,
                                                                       affine=True)

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]

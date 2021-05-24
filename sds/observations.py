import numpy as np
import numpy.random as npr

from scipy.stats import multivariate_normal as mvn

from sds.utils.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils.general import linear_regression, one_hot, arstack

from sds.distributions.gaussian import StackedGaussiansWithPrecision

from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision
from sds.distributions.lingauss import StackedLinearGaussiansWithDiagonalPrecision

from sds.distributions.lingauss import TiedLinearGaussiansWithPrecision
from sds.distributions.lingauss import TiedLinearGaussiansWithDiagonalPrecision

import copy


class GaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim, reg=1e-8):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.reg = reg

        self.mu = npr.randn(self.nb_states, self.obs_dim)
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.obs_dim, self.obs_dim)

    @property
    def sigma(self):
        return np.matmul(self._sigma_chol, np.swapaxes(self._sigma_chol, -1, -2))

    @sigma.setter
    def sigma(self, value):
        self._sigma_chol = np.linalg.cholesky(value + self.reg * np.eye(self.obs_dim))

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

        ts = [_x.shape[0] for _x in x]

        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(x), np.vstack(u))))
            zs = np.split(km.labels_, np.cumsum(ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=t) for t in ts]

        zs = [one_hot(z, self.nb_states) for z in zs]
        self.mstep(zs, x, u)

    def mean(self, z, x=None, u=None):
        return self.mu[z, :]

    def sample(self, z, x=None, u=None):
        y = mvn(mean=self.mean(z), cov=self.sigma[z, ...]).rvs()
        return np.atleast_1d(y)

    def log_likelihood(self, x, u=None):
        loglik = []
        for _x in x:
            _loglik = np.zeros((_x.shape[0], self.nb_states))
            for k in range(self.nb_states):
                _loglik[:, k] = lg_mvn(_x, self.mean(k), self.sigma[k])
            loglik.append(_loglik)
        return loglik

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
            sqerr += np.sum(_p[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            norm += np.sum(_p, axis=0)

        self.sigma = sqerr / norm[:, None, None]

    def smooth(self, p, x, u):
        mean = []
        for _x, _u, _p in zip(x, u, p):
            mean.append(_p.dot(self.mu))
        return mean


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim, nb_lags=1, reg=1e-8):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.reg = reg

        self.A = npr.randn(self.nb_states, self.obs_dim, self.obs_dim * self.nb_lags)
        self.B = npr.randn(self.nb_states, self.obs_dim, self.act_dim)
        self.c = npr.randn(self.nb_states, self.obs_dim, )
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.obs_dim, self.obs_dim)

    @property
    def sigma(self):
        return np.matmul(self._sigma_chol, np.swapaxes(self._sigma_chol, -1, -2))

    @sigma.setter
    def sigma(self, value):
        self._sigma_chol = np.linalg.cholesky(value + self.reg * np.eye(self.obs_dim))

    @property
    def params(self):
        return self.A, self.B, self.c, self._sigma_chol

    @params.setter
    def params(self, value):
        self.A, self.B, self.c, self._sigma_chol = value

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self._sigma_chol = self._sigma_chol[perm, ...]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        xu, xn = [], []
        for _x, _u in zip(x, u):
            _xr = arstack(_x[:-1], self.nb_lags)
            _ur = _u[self.nb_lags - 1:-1]
            xu.append(np.hstack((_xr, _ur)))
            xn.append(_x[self.nb_lags:])

        ts = [_xu.shape[0] for _xu in xu]

        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(xu))
            zs = np.split(km.labels_, np.cumsum(ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=t) for t in ts]

        zs = [one_hot(z, self.nb_states) for z in zs]

        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.obs_dim + 1)

        _sigma = np.zeros((self.nb_states, self.obs_dim, self.obs_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(xu), ys=np.vstack(xn),
                                                       weights=np.vstack(zs)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.A[k, ...] = coef[:, :self.obs_dim * self.nb_lags]
            self.B[k, ...] = coef[:, self.obs_dim * self.nb_lags:]
            self.c[k, ...] = intercept
            _sigma[k, ...] = sigma

        self.sigma = _sigma

    def mean(self, z, x, u):
        xr = np.squeeze(np.reshape(x, (-1, self.obs_dim * self.nb_lags)))
        return np.einsum('kh,...h->...k', self.A[z, ...], xr) +\
               np.einsum('kh,...h->...k', self.B[z, ...], u) + self.c[z, :]

    def sample(self, z, x, u):
        xn = mvn(self.mean(z, x, u), cov=self.sigma[z, ...]).rvs()
        return np.atleast_1d(xn)

    def reset(self):
        self.A = npr.randn(self.nb_states, self.obs_dim, self.obs_dim * self.nb_lags)
        self.B = npr.randn(self.nb_states, self.obs_dim, self.act_dim)
        self.c = npr.randn(self.nb_states, self.obs_dim, )
        self._sigma_chol = 5. * npr.randn(self.nb_states, self.obs_dim, self.obs_dim)

    def log_likelihood(self, x, u):
        xr, ur, xn = [], [], []
        for _x, _u in zip(x, u):
            xr.append(arstack(_x[:-1], self.nb_lags))
            ur.append(_u[self.nb_lags - 1:-1])
            xn.append(_x[self.nb_lags:])

        loglik = []
        for _xr, _ur, _xn in zip(xr, ur, xn):
            _loglik = np.zeros((_xr.shape[0], self.nb_states))
            for k in range(self.nb_states):
                _mu = self.mean(k, _xr, _ur)
                _loglik[:, k] = lg_mvn(_xn, _mu, self.sigma[k])
            loglik.append(_loglik)

        return loglik

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.obs_dim + 1)

        xu, xn, ws = [], [], []
        for _x, _u, _w in zip(x, u, p):
            _xr = arstack(_x[:-1], self.nb_lags)
            _ur = _u[self.nb_lags - 1:-1]
            xu.append(np.hstack((_xr, _ur)))
            xn.append(_x[self.nb_lags:])
            ws.append(_w[self.nb_lags:])

        _sigma = np.zeros((self.nb_states, self.obs_dim, self.obs_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(xu), ys=np.vstack(xn),
                                                       weights=np.vstack(ws)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

            self.A[k, ...] = coef[:, :self.obs_dim * self.nb_lags]
            self.B[k, ...] = coef[:, self.obs_dim * self.nb_lags:]
            self.c[k, ...] = intercept
            _sigma[k, ...] = sigma

        self.sigma = _sigma

    def smooth(self, p, x, u):
        xr, ur, pr = [], [], []
        for _x, _u, _p in zip(x, u, p):
            xr.append(arstack(_x[:-1], self.nb_lags))
            ur.append(_u[self.nb_lags - 1:-1])
            pr.append(_p[self.nb_lags:])

        mean = []
        for _xr, _ur, _pr in zip(xr, ur, pr):
            _mu = np.zeros((len(_xr), self.nb_states, self.obs_dim))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _xr, _ur)
            mean.append(np.einsum('nk,nkl->nl', _pr, _mu))

        return mean


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
        stats = self.likelihood.statistics(x)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def mean(self, z, x=None, u=None):
        return self.likelihood.dists[z].mean()

    def sample(self, z, x=None, u=None):
        y = self.likelihood.dists[z].rvs()
        return np.atleast_1d(y)

    def log_likelihood(self, x, u=None):
        return self.likelihood.log_likelihood(x)

    def mstep(self, p, x, u, **kwargs):
        stats = self.likelihood.weighted_statistics(x, p)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

        # self.prior.nat_param = (1. - 0.001) * self.prior.nat_param\
        #                        + 0.001 * self.posterior.nat_param

    def smooth(self, p, x, u):
        mean = []
        for _x, _u, _p in zip(x, u, p):
            mean.append(_p.dot(self.likelihood.mus))
        return mean


class _BayesianAutoRegressiveObservationBase:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):

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

        xu, xn = [], []
        for _x, _u in zip(x, u):
            _xr = arstack(_x[:-1], self.nb_lags)
            _ur = _u[self.nb_lags - 1:-1]
            xu.append(np.hstack((_xr, _ur)))
            xn.append(_x[self.nb_lags:])

        ts = [_xu.shape[0] for _xu in xu]

        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(xu))
            zs = np.split(km.labels_, np.cumsum(ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=t) for t in ts]

        zs = [one_hot(z, self.nb_states) for z in zs]

        stats = self.likelihood.weighted_statistics(xu, xn, zs)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def permute(self, perm):
        raise NotImplementedError

    def mean(self, z, x, u):
        xr = np.squeeze(np.reshape(x, (-1, self.obs_dim * self.nb_lags)))
        xu = np.hstack((xr, u))
        return self.likelihood.dists[z].mean(xu)

    def sample(self, z, x, u):
        xr = np.squeeze(np.reshape(x, (-1, self.obs_dim * self.nb_lags)))
        xu = np.hstack((x, u))
        xn = self.likelihood.dists[z].rvs(xu)
        return np.atleast_1d(xn)

    def log_likelihood(self, x, u):
        xu, xn = [], []
        for _x, _u in zip(x, u):
            _xr = arstack(_x[:-1], self.nb_lags)
            _ur = _u[self.nb_lags - 1:-1]
            xu.append(np.hstack((_xr, _ur)))
            xn.append(_x[self.nb_lags:])

        return self.likelihood.log_likelihood(xu, xn)

    def mstep(self, p, x, u, **kwargs):
        xu, xn, ws = [], [], []
        for _x, _u, _w in zip(x, u, p):
            _xr = arstack(_x[:-1], self.nb_lags)
            _ur = _u[self.nb_lags - 1:-1]
            xu.append(np.hstack((_xr, _ur)))
            xn.append(_x[self.nb_lags:])
            ws.append(_w[self.nb_lags:])

        stats = self.likelihood.weighted_statistics(xu, xn, ws)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

        # self.prior.nat_param = (1. - 0.001) * self.prior.nat_param\
        #                        + 0.001 * self.posterior.nat_param

    def smooth(self, p, x, u):
        xr, ur, pr = [], [], []
        for _x, _u, _p in zip(x, u, p):
            xr.append(arstack(_x[:-1], self.nb_lags))
            ur.append(_u[self.nb_lags - 1:-1])
            pr.append(_p[self.nb_lags:])

        mean = []
        for _xr, _ur, _pr in zip(xr, ur, pr):
            _mu = np.zeros((len(_xr), self.nb_states, self.obs_dim))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _xr, _ur)
            mean.append(np.einsum('nk,nkl->nl', _pr, _mu))

        return mean


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
                                                                  input_dim=self.input_dim,
                                                                  output_dim=self.output_dim,
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
                                                               input_dim=self.input_dim,
                                                               output_dim=self.output_dim,
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
                                                                          input_dim=self.input_dim,
                                                                          output_dim=self.output_dim,
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
                                                                       input_dim=self.input_dim,
                                                                       output_dim=self.output_dim,
                                                                       As=As, lmbda_diag=lmbda_diag,
                                                                       affine=True)

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]

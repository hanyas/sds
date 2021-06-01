import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn

from sds.utils.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils.general import linear_regression, one_hot
from sds.utils.decorate import parse_init_values

from sds.distributions.categorical import Categorical
from sds.distributions.gaussian import StackedGaussiansWithPrecision
from sds.distributions.gaussian import StackedGaussiansWithDiagonalPrecision
from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision

from sklearn.preprocessing import PolynomialFeatures

import copy


class InitCategoricalState:

    def __init__(self, nb_states, **kwargs):
        self.nb_states = nb_states
        self.pi = 1. / self.nb_states * np.ones(self.nb_states)

    @property
    def params(self):
        return self.pi

    @params.setter
    def params(self, value):
        self.pi = value

    def permute(self, perm):
        self.pi = self.pi[perm]

    def initialize(self):
        pass

    def likeliest(self):
        return np.argmax(self.pi)

    def sample(self):
        return npr.choice(self.nb_states, p=self.pi)

    def log_init(self):
        return np.log(self.pi)

    def mstep(self, p, **kwargs):
        eps = kwargs.get('eps', 1e-8)

        pi = sum([_p[0, :] for _p in p]) + eps
        self.pi = pi / sum(pi)


class InitGaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim, nb_lags=1, **kwargs):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

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

    def initialize(self, x):
        x0 = np.vstack([_x[:self.nb_lags] for _x in x])
        self.mu = np.array([np.mean(x0, axis=0) for k in range(self.nb_states)])
        self.sigma = np.array([np.cov(x0, rowvar=False) for k in range(self.nb_states)])

    def mean(self, z):
        return self.mu[z]

    def sample(self, z):
        x = mvn(mean=self.mean(z), cov=self.sigma[z]).rvs()
        return np.atleast_1d(x)

    def log_likelihood(self, x):
        if isinstance(x, np.ndarray):
            x0 = x[:self.nb_lags]
            loglik = np.zeros((x0.shape[0], self.nb_states))
            for k in range(self.nb_states):
                loglik[:, k] = lg_mvn(x0, self.mean(k), self.sigma[k])
            return loglik
        else:
            return list(map(self.log_likelihood, x))

    def mstep(self, p, x, **kwargs):
        x0, p0 = [], []
        for _x, _p in zip(x, p):
            x0.append(_x[:self.nb_lags])
            p0.append(_p[:self.nb_lags])

        J = np.zeros((self.nb_states, self.obs_dim))
        h = np.zeros((self.nb_states, self.obs_dim))
        for _x, _p in zip(x0, p0):
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

    def smooth(self, p, x):
        if all(isinstance(i, np.ndarray) for i in [p, x]):
            p0 = p[:self.nb_lags]
            return p0.dot(self.mu)
        else:
            return list(map(self.smooth, p, x))


class InitGaussianControl:

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

        self.K = npr.randn(self.nb_states, self.act_dim, self.feat_dim)
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
        self.K = self.K[perm]
        self.kff = self.kff[perm]
        self._sigma_chol = self._sigma_chol[perm]

    @parse_init_values
    def initialize(self, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        xs = np.concatenate(x)
        us = np.concatenate(u)
        fs = self.featurize(xs)

        K, kff, sigma = linear_regression(fs, us, weights=None, fit_intercept=True,
                                          mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

        self.K = np.array([K for _ in range(self.nb_states)])
        self.kff = np.array([kff for _ in range(self.nb_states)])
        self.sigma = np.array([sigma for _ in range(self.nb_states)])

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat)

    def mean(self, z, x):
        feat = self.featurize(x)
        return np.einsum('kh,...h->...k', self.K[z], feat) + self.kff[z]

    def sample(self, z, x):
        u = mvn(mean=self.mean(z, x), cov=self.sigma[z]).rvs()
        return np.atleast_1d(u)

    @parse_init_values
    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.zeros((_x.shape[0], self.nb_states))
            for k in range(self.nb_states):
                _loglik[:, k] = lg_mvn(_u, self.mean(k, _x), self.sigma[k])
            loglik.append(_loglik)

        return loglik

    @parse_init_values
    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        feats = [self.featurize(_x) for _x in x]

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(feats), ys=np.vstack(u),
                                                       weights=np.vstack(p)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)
            self.K[k] = coef
            self.kff[k] = intercept
            _sigma[k] = sigma

        self.sigma = _sigma

    @parse_init_values
    def smooth(self, p, x):
        mean = []
        for _x, _p in zip(x, p):
            _mu = np.zeros((len(_x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x)
            mean.append(np.einsum('nk,nkl->nl', _p, _mu))

        return mean


class BayesianInitCategoricalState:

    def __init__(self, nb_states, prior, likelihood=None):
        self.nb_states = nb_states

        # Dirichlet prior
        self.prior = prior

        # Dirichlet posterior
        self.posterior = copy.deepcopy(prior)

        # Categorical likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            pi = self.prior.rvs()
            self.likelihood = Categorical(dim=nb_states, pi=pi)

    @property
    def params(self):
        return self.likelihood.pi

    @params.setter
    def params(self, value):
        self.likelihood.pi = value

    def permute(self, perm):
        self.likelihood.pi = self.likelihood.pi[perm]

    def initialize(self):
        pass

    def likeliest(self):
        return np.argmax(self.likelihood.pi)

    def sample(self):
        return npr.choice(self.nb_states, p=self.likelihood.pi)

    def log_init(self):
        return np.log(self.likelihood.pi)

    def mstep(self, p, **kwargs):
        stats = self.likelihood.weighted_statistics(None, p)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.mode()


class _BayesianInitGaussianObservationBase:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.prior = prior
        self.posterior = copy.deepcopy(prior)
        self.likelihood = likelihood

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def permute(self, perm):
        raise NotImplementedError

    def initialize(self, x, **kwargs):
        kmeans = kwargs.get('kmeans', True)

        x0 = [_x[:self.nb_lags] for _x in x]
        ts = [_x0.shape[0] for _x0 in x0]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(x0))
            zs = np.split(km.labels_, np.cumsum(ts)[:-1])
        else:
            zs = [npr.choice(self.nb_states, size=t) for t in ts]

        z0 = [one_hot(_z[:self.nb_lags], self.nb_states) for _z in zs]
        stats = self.likelihood.weighted_statistics(x0, z0)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def mean(self, z):
        return self.likelihood.dists[z].mean()

    def sample(self, z):
        x = self.likelihood.dists[z].rvs()
        return np.atleast_1d(x)

    def log_likelihood(self, x):
        if isinstance(x, np.ndarray):
            x0 = x[:self.nb_lags]
            return self.likelihood.log_likelihood(x0)
        else:
            return list(map(self.log_likelihood, x))

    def mstep(self, p, x, **kwargs):
        x0, p0 = [], []
        for _x, _p in zip(x, p):
            x0.append(_x[:self.nb_lags])
            p0.append(_p[:self.nb_lags])

        method = kwargs.get('method', 'direct')
        if method == 'direct':
            stats = self.likelihood.weighted_statistics(x0, p0)
            self.posterior.nat_param = self.prior.nat_param + stats
        elif method == 'sgd':
            lr = kwargs.get('lr', 1e-2)
            nb_iter = kwargs.get('nb_iter', 1)

            x0, p0 = list(map(np.vstack, (x0, p0)))
            for _ in range(nb_iter):
                stats = self.likelihood.weighted_statistics(x0, p0)
                self.posterior.nat_param = (1. - lr) * self.posterior.nat_param\
                                           + lr * (self.prior.nat_param + stats)

        self.likelihood.params = self.posterior.mode()

    def smooth(self, p, x):
        if all(isinstance(i, np.ndarray) for i in [p, x]):
            p0 = p[:self.nb_lags]
            return p0.dot(self.likelihood.mus)
        else:
            return list(map(self.smooth, p, x))


class BayesianInitGaussianObservation(_BayesianInitGaussianObservationBase):

    # mu = np.zeros((obs_dim,))
    # kappa = 1e-64
    # psi = 1e8 * np.eye(obs_dim) / (obs_dim + 1)
    # nu = (obs_dim + 1) + obs_dim + 1
    #
    # from sds.distributions.composite import StackedNormalWishart
    # prior = StackedNormalWishart(nb_states, obs_dim,
    #                              mus=np.array([mu for _ in range(nb_states)]),
    #                              kappas=np.array([kappa for _ in range(nb_states)]),
    #                              psis=np.array([psi for _ in range(nb_states)]),
    #                              nus=np.array([nu for _ in range(nb_states)]))

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianInitGaussianObservation, self).__init__(nb_states, obs_dim, act_dim,
                                                              nb_lags, prior, likelihood)

        # Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mus, lmbdas = self.prior.rvs()
            self.likelihood = StackedGaussiansWithPrecision(size=self.nb_states,
                                                            dim=self.obs_dim,
                                                            mus=mus, lmbdas=lmbdas)

    def permute(self, perm):
        self.likelihood.mus = self.likelihood.mus[perm]
        self.likelihood.lmbdas = self.likelihood.lmbdas[perm]


class BayesianInitDiagonalGaussianObservation(_BayesianInitGaussianObservationBase):

    # mu = np.zeros((obs_dim,))
    # kappa = 1e-64 * np.ones((obs_dim,))
    # alpha = ((obs_dim + 1) + obs_dim + 1) / 2. * np.ones((obs_dim,))
    # beta = 1. / (2. * 1e8 * np.ones((obs_dim,)) / (obs_dim + 1))
    #
    # from sds.distributions.composite import StackedNormalGamma
    # prior = StackedNormalGamma(nb_states, obs_dim,
    #                            mus=np.array([mu for _ in range(nb_states)]),
    #                            kappas=np.array([kappa for _ in range(nb_states)]),
    #                            alphas=np.array([alpha for _ in range(nb_states)]),
    #                            betas=np.array([beta for _ in range(nb_states)]))

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianInitDiagonalGaussianObservation, self).__init__(nb_states, obs_dim, act_dim,
                                                                      nb_lags, prior, likelihood)

        # Diagonal Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mus, lmbdas_diag = self.prior.rvs()
            self.likelihood = StackedGaussiansWithDiagonalPrecision(size=self.nb_states,
                                                                    dim=self.obs_dim,
                                                                    mus=mus, lmbdas_diag=lmbdas_diag)

    def permute(self, perm):
        self.likelihood.mus = self.likelihood.mus[perm]
        self.likelihood.lmbdas_diag = self.likelihood.lmbdas_diag[perm]


class BayesianInitGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, degree, prior, likelihood=None):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.degree = degree

        self.feat_dim = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.prior = prior
        self.posterior = copy.deepcopy(prior)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbdas = self.prior.rvs()
            self.likelihood = StackedLinearGaussiansWithPrecision(size=self.nb_states, column_dim=self.feat_dim, row_dim=self.act_dim, As=As, lmbdas=lmbdas, affine=True)

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def permute(self, perm):
        self.likelihood.As = self.likelihood.As[perm]
        self.likelihood.lmbdas = self.likelihood.lmbdas[perm]

    @parse_init_values
    def initialize(self, x, u, **kwargs):
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

        stats = self.likelihood.weighted_statistics(x, u, zs)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat)

    def mean(self, z, x):
        feat = self.featurize(x)
        return self.likelihood.dists[z].mean(feat)

    def sample(self, z, x):
        u = self.likelihood.dists[z].rvs(x)
        return np.atleast_1d(u)

    @parse_init_values
    def log_likelihood(self, x, u):
        return self.likelihood.log_likelihood(x, u)

    @parse_init_values
    def mstep(self, p, x, u, **kwargs):
        stats = self.likelihood.weighted_statistics(x, u, p)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.mode()

    @parse_init_values
    def smooth(self, p, x):
        mean = []
        for _x, _p in zip(x, p):
            _mu = np.zeros((len(_x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                _mu[:, k, :] = np.mean(k, _x)
            mean.append(np.einsum('nk,nkl->nk', _p, _mu))

        return mean

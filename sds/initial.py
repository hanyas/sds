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

from sklearn.preprocessing import PolynomialFeatures

import copy


class InitCategoricalState:

    def __init__(self, nb_states, reg=1e-8):
        self.nb_states = nb_states

        self.reg = reg
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
        pi = sum([_p[0, :] for _p in p]) + self.reg
        self.pi = pi / sum(pi)


class InitGaussianObservation:

    def __init__(self, nb_states, obs_dim, act_dim, nb_lags=1, reg=1e-8):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

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
        self.mu = self.mu[perm, ...]
        self._sigma_chol = self._sigma_chol[perm, ...]

    @parse_init_values
    def initialize(self, x):
        obs = np.concatenate(x)
        self.mu = np.array([np.mean(obs, axis=0)
                            for k in range(self.nb_states)])
        self.sigma = np.array([np.cov(obs, rowvar=False)
                               for k in range(self.nb_states)])

    def mean(self, z):
        return self.mu[z, :]

    def sample(self, z):
        y = mvn(mean=self.mean(z), cov=self.sigma[z, ...]).rvs()
        return np.atleast_1d(y)

    @parse_init_values
    def log_likelihood(self, x):
        loglik = []
        for _x in x:
            _loglik = np.zeros((_x.shape[0], self.nb_states))
            for k in range(self.nb_states):
                _loglik[:, k] = lg_mvn(_x, self.mean(k), self.sigma[k])
            loglik.append(_loglik)
        return loglik

    @parse_init_values
    def mstep(self, p, x, **kwargs):
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

    @parse_init_values
    def smooth(self, p, x):
        mean = []
        for _x, _p in zip(x, p):
            mean.append(_p.dot(self.mu))
        return mean


class InitGaussianControl:

    def __init__(self, nb_states, obs_dim, act_dim,
                 prior, lags=1, degree=1, reg=1e-8):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.lags = lags

        self.prior = prior
        self.reg = reg

        self.degree = degree
        self.dm_feat = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.K = npr.randn(self.nb_states, self.act_dim, self.dm_feat)
        self.kff = npr.randn(self.nb_states, self.act_dim)

        # self._sqrt_cov = npr.randn(self.nb_states, self.act_dim, self.act_dim)

        self._sqrt_cov = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            _cov = sc.stats.invwishart.rvs(self.act_dim + 1, np.eye(self.act_dim))
            self._sqrt_cov[k, ...] = np.linalg.cholesky(_cov * np.eye(self.act_dim))

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
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.act_dim))

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

        _cov = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [_feat[t, :] for t, _feat in zip(ts, feat)]
            ys = [_u[t, :] for t, _u in zip(ts, u)]

            coef, intercept, sigma = linear_regression(np.vstack(xs), np.vstack(ys),
                                                       weights=None, fit_intercept=True,
                                                       **self.prior)
            self.K[k, ...] = coef[:, :self.dm_feat]
            self.kff[k, :] = intercept
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

    def mstep(self, p, x, u, **kwargs):

        xs, ys, ws = [], [], []
        for _x, _u, _p in zip(x, u, p):
            _feat = self.featurize(_x)
            xs.append(np.hstack((_feat[:self.lags], np.ones((_feat[:self.lags].shape[0], 1)))))
            ys.append(_u[:self.lags])
            ws.append(_p[:self.lags])

        _cov = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False)

            self.K[k, ...] = coef[:, :self.dm_feat]
            self.kff[k, ...] = coef[:, -1]
            _cov[k, ...] = sigma

        self.cov = _cov

    def smooth(self, p, x):
        mean = []
        for _x, _p in zip(x, p):
            mu = np.zeros((self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[k, :] = self.mean(k, _x)
            mean.append(np.einsum('k,kl->l', _p[self.lags], mu))
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

        self.prior.nat_param = (1. - 0.01) * self.prior.nat_param\
                               + 0.01 * self.posterior.nat_param


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

    @parse_init_values
    def initialize(self, x, **kwargs):
        T = [_x.shape[0] for _x in x]
        z = [one_hot(npr.choice(self.nb_states, size=t),
                     self.nb_states) for t in T]

        stats = self.likelihood.weighted_statistics(x, z)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def mean(self, z, x=None):
        return self.likelihood.dists[z].mean()

    def sample(self, z, x=None):
        y = self.likelihood.dists[z].rvs()
        return np.atleast_1d(y)

    @parse_init_values
    def log_likelihood(self, x):
        return self.likelihood.log_likelihood(x)

    @parse_init_values
    def mstep(self, p, x, **kwargs):
        stats = self.likelihood.statistics(x) if p is None\
            else self.likelihood.weighted_statistics(x, p)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    @parse_init_values
    def smooth(self, p, x):
        mean = []
        for _x, _p in zip(x, p):
            mean.append(_p.dot(self.likelihood.mus))
        return mean


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

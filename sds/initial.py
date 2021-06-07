import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart

from sds.utils.stats import multivariate_normal_logpdf as lg_mvn
from sds.utils.general import linear_regression, one_hot

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

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        # self.mu = npr.randn(self.nb_states, self.obs_dim)
        # self._sigma_chol = 5. * npr.randn(self.nb_states, self.obs_dim, self.obs_dim)

        self.mu = np.zeros((self.nb_states, self.obs_dim))
        self._sigma_chol = np.zeros((self.nb_states, self.obs_dim, self.obs_dim))

        for k in range(self.nb_states):
            _sigma = invwishart.rvs(self.obs_dim + 1, np.eye(self.obs_dim))
            self._sigma_chol[k] = np.linalg.cholesky(_sigma * np.eye(self.obs_dim))
            self.mu[k] = mvn.rvs(mean=None, cov=1e2 * _sigma, size=(1, ))

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

    def initialize(self, x, **kwargs):
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
        for _x, _p in zip(x0, p0):
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
                 nb_lags=1, degree=1, **kwargs):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.degree = degree
        self.feat_dim = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
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
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        x0 = np.vstack([_x[:self.nb_lags] for _x in x])
        u0 = np.vstack([_u[:self.nb_lags] for _u in u])
        f0 = self.featurize(x0)

        K, kff, sigma = linear_regression(f0, u0, weights=None, fit_intercept=True,
                                          mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)

        self.K = np.array([K for _ in range(self.nb_states)])
        self.kff = np.array([kff for _ in range(self.nb_states)])
        self.sigma = np.array([sigma for _ in range(self.nb_states)])

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

    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray):
            x0 = x[:self.nb_lags]
            u0 = u[:self.nb_lags]
            loglik = np.zeros((u0.shape[0], self.nb_states))
            for k in range(self.nb_states):
                loglik[:, k] = lg_mvn(u0, self.mean(k, x0), self.sigma[k])
            return loglik
        else:
            return list(map(self.log_likelihood, x, u))

    def mstep(self, p, x, u, **kwargs):
        mu0 = kwargs.get('mu0', 0.)
        sigma0 = kwargs.get('sigma0', 1e64)
        psi0 = kwargs.get('psi0', 1.)
        nu0 = kwargs.get('nu0', self.act_dim + 1)

        x0, u0, p0 = [], [], []
        for _x, _u, _p in zip(x, u, p):
            x0.append(_x[:self.nb_lags])
            u0.append(_u[:self.nb_lags])
            p0.append(_p[:self.nb_lags])
        f0 = [self.featurize(_x0) for _x0 in x0]

        _sigma = np.zeros((self.nb_states, self.act_dim, self.act_dim))
        for k in range(self.nb_states):
            coef, intercept, sigma = linear_regression(Xs=np.vstack(f0), ys=np.vstack(u0),
                                                       weights=np.vstack(p0)[:, k], fit_intercept=True,
                                                       mu0=mu0, sigma0=sigma0, psi0=psi0, nu0=nu0)
            self.K[k] = coef
            self.kff[k] = intercept
            _sigma[k] = sigma

        self.sigma = _sigma

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            x0 = x[:self.nb_lags]
            u0 = u[:self.nb_lags]
            p0 = p[:self.nb_lags]

            mu = np.zeros((len(u0), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x0)
            return np.einsum('nk,nkl->nl', p, mu)
        else:
            return list(map(self.smooth, p, x, u))


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

        assert nb_lags > 0

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
        t = [_x0.shape[0] for _x0 in x0]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(x0))
            z0 = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z0 = [npr.choice(self.nb_states, size=_t) for _t in t]

        z0 = [one_hot(_z, self.nb_states) for _z in z0]

        stats = self.likelihood.weighted_statistics(x0, z0)
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.rvs()

    def mean(self, z):
        x = self.likelihood.dists[z].mean()
        return np.atleast_1d(x)

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

        self.prior.nat_param = (1. - 1e-1) * self.prior.nat_param\
                               + 1e-1 * self.posterior.nat_param

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
                 nb_lags, prior, degree=1, likelihood=None):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

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
        self.likelihood.lmbdas = self.likelihood.lmbdas[perm]

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', False)

        x0, u0 = [], []
        for _x, _u in zip(x, u):
            x0.append(_x[:self.nb_lags])
            u0.append(_u[:self.nb_lags])
        f0 = [self.featurize(_x0) for _x0 in x0]

        t = [_f0.shape[0] for _f0 in f0]
        if kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.vstack(f0))
            z0 = np.split(km.labels_, np.cumsum(t)[:-1])
        else:
            z0 = [npr.choice(self.nb_states, size=_t) for _t in t]

        z0 = [one_hot(_z, self.nb_states) for _z in z0]

        stats = self.likelihood.weighted_statistics(f0, u0, z0)
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

    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray):
            x0 = x[:self.nb_lags]
            u0 = u[:self.nb_lags]
            f0 = self.featurize(x0)
            return self.likelihood.log_likelihood(f0, u0)
        else:
            return list(map(self.log_likelihood, x, u))

    def mstep(self, p, x, u, **kwargs):
        x0, u0, p0 = [], [], []
        for _x, _u, _p in zip(x, u, p):
            x0.append(_x[:self.nb_lags])
            u0.append(_u[:self.nb_lags])
            p0.append(_p[:self.nb_lags])
        f0 = [self.featurize(_x0) for _x0 in x0]

        method = kwargs.get('method', 'direct')
        if method == 'direct':
            stats = self.likelihood.weighted_statistics(f0, u0, p0)
            self.posterior.nat_param = self.prior.nat_param + stats
        elif method == 'sgd':
            lr = kwargs.get('lr', 1e-2)
            nb_iter = kwargs.get('nb_iter', 1)

            f0, u0, p0 = list(map(np.vstack, (f0, u0, p0)))
            for _ in range(nb_iter):
                stats = self.likelihood.weighted_statistics(f0, u0, p0)
                self.posterior.nat_param = (1. - lr) * self.posterior.nat_param\
                                           + lr * (self.prior.nat_param + stats)

        # self.prior.nat_param = (1. - 1e-3) * self.prior.nat_param\
        #                        + 1e-3 * self.posterior.nat_param

        self.likelihood.params = self.posterior.mode()

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            x0 = x[:self.nb_lags]
            u0 = u[:self.nb_lags]
            p0 = p[:self.nb_lags]

            mu = np.zeros((len(u0), self.nb_states, self.obs_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x0)
            return np.einsum('nk,nkl->nl', p0, mu)
        else:
            return list(map(self.smooth, p, x, u))


class BayesianInitGaussianControlWithAutomaticRelevance:

    def __init__(self, nb_states, obs_dim, act_dim,
                 nb_lags, prior, degree=1):

        assert nb_lags > 0

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.degree = degree
        self.feat_dim = int(sc.special.comb(self.degree + self.obs_dim, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.input_dim = self.feat_dim + 1
        self.output_dim = self.act_dim

        likelihood_precision_prior = prior['likelihood_precision_prior']
        parameter_precision_prior = prior['parameter_precision_prior']

        from sds.distributions.composite import StackedMultiOutputLinearGaussianWithAutomaticRelevance
        self.object = StackedMultiOutputLinearGaussianWithAutomaticRelevance(self.nb_states,
                                                                             self.input_dim,
                                                                             self.output_dim,
                                                                             likelihood_precision_prior,
                                                                             parameter_precision_prior)

    @property
    def params(self):
        return self.object.params

    @params.setter
    def params(self, values):
        self.object.params = values

    def permute(self, perm):
        self.object.As = self.object.As[perm]
        self.object.lmbdas = self.object.lmbdas[perm]

    def initialize(self, x, u, **kwargs):
        pass

    def featurize(self, x):
        feat = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(feat) if x.ndim == 1\
               else np.reshape(feat, (x.shape[0], -1))

    def mean(self, z, x):
        feat = self.featurize(x)
        u = self.object.mean(z, feat)
        return np.atleast_1d(u)

    def sample(self, z, x):
        feat = self.featurize(x)
        u = self.object.rvs(z, feat)
        return np.atleast_1d(u)

    def log_likelihood(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            x0 = x[:self.nb_lags]
            u0 = u[:self.nb_lags]
            f0 = self.featurize(x0)
            return self.object.log_likelihood(f0, u0)
        else:
            def inner(x, u):
                return self.log_likelihood(x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        x0, u0, p0 = [], [], []
        for _x, _u, _p in zip(x, u, p):
            x0.append(_x[:self.nb_lags])
            u0.append(_u[:self.nb_lags])
            p0.append(_p[:self.nb_lags])
        f0 = [self.featurize(_x0) for _x0 in x0]

        f0, u0, p0 = list(map(np.vstack, (f0, u0, p0)))
        self.object.em(f0, u0, p0, **kwargs)

    def smooth(self, p, x, u):
        if all(isinstance(i, np.ndarray) for i in [p, x, u]):
            x0 = x[:self.nb_lags]
            u0 = u[:self.nb_lags]
            p0 = p[:self.nb_lags]

            mu = np.zeros((len(x), self.nb_states, self.act_dim))
            for k in range(self.nb_states):
                mu[:, k, :] = self.mean(k, x0)
            return np.einsum('nk,nkl->nl', p0, mu)
        else:
            return list(map(self.smooth, p, x, u))

from abc import ABC

import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import linalg
from scipy.special import digamma

from sds.distributions.gaussian import GaussianWithPrecision
from sds.distributions.gaussian import GaussianWithDiagonalPrecision

from sds.distributions.matrix import MatrixNormalWithPrecision
from sds.distributions.matrix import MatrixNormalWithDiagonalPrecision

from sds.distributions.lingauss import LinearGaussianWithPrecision

from sds.distributions.lingauss import SingleOutputLinearGaussianWithKnownPrecision
from sds.distributions.lingauss import SingleOutputLinearGaussianWithKnownMean
from sds.distributions.gaussian import GaussianWithKnownMeanAndDiagonalPrecision

from sds.distributions.wishart import Wishart
from sds.distributions.gamma import Gamma

from sds.utils.general import Statistics as Stats

from functools import partial
from copy import deepcopy


class NormalWishart:

    def __init__(self, dim, mu=None, kappa=None,
                 psi=None, nu=None):

        self.dim = dim

        self.gaussian = GaussianWithPrecision(dim=dim, mu=mu)
        self.wishart = Wishart(dim=dim, psi=psi, nu=nu)
        self.kappa = kappa

    @property
    def params(self):
        return self.gaussian.mu, self.kappa, self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappa, self.wishart.psi, self.wishart.nu = values

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [mu.T @ lmbda,
        #          -0.5 * lmbda @ (mu @ mu.T),
        #          -0.5 * lmbda,
        #          0.5 * logdet(lmbda)]
        #
        # nats = [kappa * m,
        #         kappa,
        #         psi^-1 + kappa * (m @ m.T),
        #         nu - d]

        a = params[1] * params[0]
        b = params[1]
        c = np.linalg.inv(params[2]) + params[1] * np.outer(params[0], params[0])
        d = params[3] - params[2].shape[0]
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappa = natparam[1]
        psi = np.linalg.inv(natparam[2] - kappa * np.outer(mu, mu))
        nu = natparam[3] + natparam[2].shape[0]
        return mu, kappa, psi, nu

    def mean(self):
        return self.gaussian.mean(), self.wishart.mean()

    def mode(self):
        mu = self.gaussian.mode()
        lmbda = (self.wishart.nu - self.dim) * self.wishart.psi
        return mu, lmbda

    def rvs(self):
        lmbda = self.wishart.rvs()
        self.gaussian.lmbda = self.kappa * lmbda
        mu = self.gaussian.rvs()
        return mu, lmbda

    @property
    def base(self):
        return self.gaussian.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, kappa, psi, nu = self.params
        return - 0.5 * self.dim * np.log(kappa)\
               + Wishart(dim=self.dim, psi=psi, nu=nu).log_partition()

    def log_likelihood(self, x):
        mu, lmbda = x
        return GaussianWithPrecision(dim=self.dim, mu=self.gaussian.mu,
                                     lmbda=self.kappa * lmbda).log_likelihood(mu) \
               + self.wishart.log_likelihood(lmbda)

    def log_likelihood_grad(self, x):
        mu, lmbda = x
        a = lmbda @ (mu - self.gaussian.mu)
        b = 0.5 * (self.dim / self.kappa - (mu - self.gaussian.mu).T @ lmbda @ (mu - self.gaussian.mu))
        c = 0.5 * ((np.linalg.inv(self.wishart.psi) @ lmbda @ np.linalg.inv(self.wishart.psi)).T
                   - self.wishart.nu * np.linalg.inv(self.wishart.psi).T)
        d = 0.5 * (np.linalg.slogdet(lmbda)[1] - self.dim * np.log(2.)
                   - np.linalg.slogdet(self.wishart.psi)[1] - digamma(self.wishart.nu / 2.))
        return a, b, c, d


class StackedNormalWisharts:

    def __init__(self, size, dim,
                 mus=None, kappas=None,
                 psis=None, nus=None):

        self.size = size
        self.dim = dim

        mus = [None] * self.size if mus is None else mus
        kappas = [None] * self.size if kappas is None else kappas
        psis = [None] * self.size if psis is None else psis
        nus = [None] * self.size if nus is None else nus
        self.dists = [NormalWishart(dim, mus[k], kappas[k],
                                    psis[k], nus[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.mus, self.kappas, self.psis, self.nus

    @params.setter
    def params(self, values):
        self.mus, self.kappas, self.psis, self.nus = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def mus(self):
        return np.array([dist.gaussian.mu for dist in self.dists])

    @mus.setter
    def mus(self, value):
        for k, dist in enumerate(self.dists):
            dist.gaussian.mu = value[k, ...]

    @property
    def kappas(self):
        return np.array([dist.kappa for dist in self.dists])

    @kappas.setter
    def kappas(self, value):
        for k, dist in enumerate(self.dists):
            dist.kappa = value[k, ...]

    @property
    def psis(self):
        return np.array([dist.wishart.psi for dist in self.dists])

    @psis.setter
    def psis(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.psi = value[k, ...]

    @property
    def nus(self):
        return np.array([dist.wishart.nu for dist in self.dists])

    @nus.setter
    def nus(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.nu = value[k, ...]

    def mean(self):
        zipped = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def mode(self):
        zipped = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def rvs(self):
        zipped = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])

    def log_likelihood_grad(self, x):
        grad_list = [dist.log_likelihood_grad(_x)
                     for dist, _x in zip(self.dists, list(zip(*x)))]
        grad_stack = tuple(map(partial(np.stack, axis=0), zip(*grad_list)))
        return grad_stack


class TiedNormalWisharts(StackedNormalWisharts):

    def __init_(self, size, dim,
                 mus=None, kappas=None,
                 psis=None, nus=None):

        super(TiedNormalWisharts, self).__init__(size, dim,
                                                 mus, kappas,
                                                 psis, nus)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        mus = np.einsum('k,kd->kd', 1. / natparam[1], natparam[0])
        kappas = natparam[1]
        psi = np.linalg.inv(np.mean(natparam[2] - np.einsum('k,kd,kl->kdl', kappas, mus, mus), axis=0))
        nu = np.mean(natparam[3] + self.dim)

        psis = np.array(self.size * [psi])
        nus = np.array(self.size * [nu])
        return mus, kappas, psis, nus


class NormalGamma:

    def __init__(self, dim, mu=None, kappas=None,
                 alphas=None, betas=None):

        self.dim = dim

        self.gaussian = GaussianWithDiagonalPrecision(dim=dim, mu=mu)
        self.gamma = Gamma(dim=dim, alphas=alphas, betas=betas)
        self.kappas = kappas

    @property
    def params(self):
        return self.gaussian.mu, self.kappas, self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappas, self.gamma.alphas, self.gamma.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [mu * lmbda_diag,
        #          -0.5 * lmbda_diag * mu * mu,
        #          0.5 * log(lmbda_diag),
        #          -0.5 * lmbda_diag]
        #
        # nats = [kappa * m,
        #         kappa,
        #         2. * alpha - 1.,
        #         2. * beta + kappa * m * m]

        a = params[1] * params[0]
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + params[1] * params[0]**2
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappas = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - kappas * mu**2)
        return mu, kappas, alphas, betas

    def mean(self):
        return self.gaussian.mean(), self.gamma.mean()

    def mode(self):
        mu = self.gaussian.mode()
        lmbda_diag = (self.gamma.alphas - 1. / 2.) / self.gamma.betas
        return mu, lmbda_diag

    def rvs(self):
        lmbda_diag = self.gamma.rvs()
        self.gaussian.lmbda_diag = self.kappas * lmbda_diag
        mu = self.gaussian.rvs()
        return mu, lmbda_diag

    @property
    def base(self):
        return self.gaussian.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu, kappas, alphas, betas = self.params
        return - 0.5 * np.sum(np.log(kappas))\
               + Gamma(dim=self.dim, alphas=alphas, betas=betas).log_partition()

    def log_likelihood(self, x):
        mu, lmbda_diag = x
        return GaussianWithDiagonalPrecision(dim=self.dim, mu=self.gaussian.mu,
                                             lmbda_diag=self.kappas * lmbda_diag).log_likelihood(mu)\
               + self.gamma.log_likelihood(lmbda_diag)


class StackedNormalGammas:

    def __init__(self, size, dim,
                 mus=None, kappas=None,
                 alphas=None, betas=None):

        self.size = size
        self.dim = dim

        mus = [None] * self.size if mus is None else mus
        kappas = [None] * self.size if kappas is None else kappas
        alphas = [None] * self.size if alphas is None else alphas
        betas = [None] * self.size if betas is None else betas

        self.dists = [NormalGamma(dim, mus[k], kappas[k],
                                  alphas[k], betas[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.mus, self.kappas, self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.mus, self.kappas, self.alphas, self.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def mus(self):
        return np.array([dist.gaussian.mu for dist in self.dists])

    @mus.setter
    def mus(self, value):
        for k, dist in enumerate(self.dists):
            dist.gaussian.mu = value[k, ...]

    @property
    def kappas(self):
        return np.array([dist.kappas for dist in self.dists])

    @kappas.setter
    def kappas(self, value):
        for k, dist in enumerate(self.dists):
            dist.kappas = value[k, ...]

    @property
    def alphas(self):
        return np.array([dist.gamma.alphas for dist in self.dists])

    @alphas.setter
    def alphas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.alphas = value[k, ...]

    @property
    def betas(self):
        return np.array([dist.gamma.betas for dist in self.dists])

    @betas.setter
    def betas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.betas = value[k, ...]

    def mean(self):
        zipped = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def mode(self):
        zipped = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def rvs(self):
        zipped = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])


class TiedNormalGammas(StackedNormalGammas):

    def __init_(self, size, dim,
                 mus=None, kappas=None,
                 alphas=None, betas=None):

        super(TiedNormalGammas, self).__init__(size, dim,
                                               mus, kappas,
                                               alphas, betas)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        mus = np.einsum('kd,kd->kd', 1. / natparam[1], natparam[0])
        kappas = natparam[1]
        alphas = np.mean(0.5 * (natparam[2] + 1.), axis=0)
        betas = np.mean(0.5 * (natparam[3] - kappas * mus**2), axis=0)

        alphas = np.array(self.size * [alphas])
        betas = np.array(self.size * [betas])
        return mus, kappas, alphas, betas


class MatrixNormalWishart:

    def __init__(self, column_dim, row_dim,
                 M=None, K=None, psi=None, nu=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.matnorm = MatrixNormalWithPrecision(column_dim, row_dim, M=M, K=K)
        self.wishart = Wishart(dim=row_dim, psi=psi, nu=nu)

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K, self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K, self.wishart.psi, self.wishart.nu = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        # stats = [A.T @ V,
        #          -0.5 * A.T @ V @ A,
        #          -0.5 * V,
        #          0.5 * log_det(V)]
        #
        # nats = [M @ K,
        #         K,
        #         psi^-1 + M @ K @ M.T,
        #         nu - d - 1. + l]

        a = params[0] @ params[1]
        b = params[1]
        c = np.linalg.inv(params[2]) + params[0] @ params[1] @ params[0].T
        d = params[3] - self.row_dim - 1. + self.column_dim
        return Stats([a, b, c, d])

    def nat_to_std(self, natparam):
        M = natparam[0] @ np.linalg.inv(natparam[1])
        K = natparam[1]
        psi = np.linalg.inv(natparam[2] - M @ K @ M.T)
        nu = natparam[3] + self.row_dim + 1. - self.column_dim
        return M, K, psi, nu

    def mean(self):
        return self.matnorm.mean(), self.wishart.mean()

    def mode(self):
        A = self.matnorm.mode()
        lmbda = (self.wishart.nu - self.row_dim) * self.wishart.psi
        return A, lmbda

    def rvs(self, size=1):
        lmbda = self.wishart.rvs()
        self.matnorm.V = lmbda
        A = self.matnorm.rvs()
        return A, lmbda

    @property
    def base(self):
        return self.matnorm.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, K, psi, nu = self.params
        return - 0.5 * self.row_dim * np.linalg.slogdet(K)[1]\
               + Wishart(dim=self.row_dim, psi=psi, nu=nu).log_partition()

    def log_likelihood(self, x):
        A, lmbda = x
        return MatrixNormalWithPrecision(column_dim=self.column_dim,
                                         row_dim=self.row_dim,
                                         M=self.matnorm.M, V=lmbda,
                                         K=self.matnorm.K).log_likelihood(A)\
               + self.wishart.log_likelihood(lmbda)

    def log_likelihood_grad(self, x):
        A, lmbda = x
        a = 0.5 * (lmbda @ A @ self.matnorm.K + (self.matnorm.K @ A.T @ lmbda).T) \
            - 0.5 * (lmbda @ self.matnorm.M @ self.matnorm.K + lmbda.T @ self.matnorm.M @ self.matnorm.K.T)
        b = 0.5 * (self.row_dim * np.linalg.inv(self.matnorm.K).T
                   - ((A - self.matnorm.M).T @ lmbda @ (A - self.matnorm.M)).T)
        c = 0.5 * ((np.linalg.inv(self.wishart.psi) @ lmbda @ np.linalg.inv(self.wishart.psi)).T
                   - self.wishart.nu * np.linalg.inv(self.wishart.psi).T)
        d = 0.5 * (np.linalg.slogdet(lmbda)[1] - self.row_dim * np.log(2.)
                   - np.linalg.slogdet(self.wishart.psi)[1] - digamma(self.wishart.nu / 2.))
        return a, b, c, d


class StackedMatrixNormalWisharts:

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Ks=None, psis=None, nus=None):

        self.size = size

        self.column_dim = column_dim
        self.row_dim = row_dim

        Ms = [None] * self.size if Ms is None else Ms
        Ks = [None] * self.size if Ks is None else Ks
        psis = [None] * self.size if psis is None else psis
        nus = [None] * self.size if nus is None else nus

        self.dists = [MatrixNormalWishart(column_dim, row_dim,
                                          Ms[k], Ks[k], psis[k], nus[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.Ms, self.Ks, self.psis, self.nus

    @params.setter
    def params(self, values):
        self.Ms, self.Ks, self.psis, self.nus = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def Ms(self):
        return np.array([dist.matnorm.M for dist in self.dists])

    @Ms.setter
    def Ms(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.M = value[k, ...]

    @property
    def Ks(self):
        return np.array([dist.matnorm.K for dist in self.dists])

    @Ks.setter
    def Ks(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.K = value[k, ...]

    @property
    def psis(self):
        return np.array([dist.wishart.psi for dist in self.dists])

    @psis.setter
    def psis(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.psi = value[k, ...]

    @property
    def nus(self):
        return np.array([dist.wishart.nu for dist in self.dists])

    @nus.setter
    def nus(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.nu = value[k, ...]

    def mean(self):
        zipped = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def mode(self):
        zipped = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def rvs(self):
        zipped = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])

    def log_likelihood_grad(self, x):
        grad_list = [dist.log_likelihood_grad(_x)
                     for dist, _x in zip(self.dists, list(zip(*x)))]
        grad_stack = tuple(map(partial(np.stack, axis=0), zip(*grad_list)))
        return grad_stack


class TiedMatrixNormalWisharts(StackedMatrixNormalWisharts):

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Ks=None, psis=None, nus=None):

        super(TiedMatrixNormalWisharts, self).__init__(size, column_dim, row_dim,
                                                       Ms, Ks, psis, nus)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        Ms = np.einsum('kdl,klh->kdh', natparam[0], np.linalg.inv(natparam[1]))
        Ks = natparam[1]
        psi = np.linalg.inv(np.mean(natparam[2] - np.einsum('kdl,klm,khm->kdh', Ms, Ks, Ms), axis=0))
        nu = np.mean(natparam[3] + self.row_dim + 1 - self.column_dim)

        psis = np.array(self.size * [psi])
        nus = np.array(self.size * [nu])
        return Ms, Ks, psis, nus


class MatrixNormalGamma:

    def __init__(self, column_dim, row_dim,
                 M=None, K=None, alphas=None, betas=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.matnorm = MatrixNormalWithDiagonalPrecision(column_dim, row_dim, M=M, K=K)
        self.gamma = Gamma(dim=row_dim, alphas=alphas, betas=betas)

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K, self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K, self.gamma.alphas, self.gamma.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [A.T * V_diag,
        #          -0.5 * A.T @ A,
        #          0.5 * log(V_diag),
        #          -0.5 * V_diag]
        #
        # nats = [M @ K,
        #         K,
        #         2. * alpha - 1.,
        #         2. * beta + M @ K @ M.T]

        a = params[0] @ params[1]
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + np.einsum('dl,lm,dm->d', params[0], params[1], params[0])
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        M = natparam[0] @ np.linalg.inv(natparam[1])
        K = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - np.einsum('dl,lm,dm->d', M, K, M))
        return M, K, alphas, betas

    def mean(self):
        return self.matnorm.mean(), self.gamma.mean()

    def mode(self):
        A = self.matnorm.mode()
        lmbda_diag = (self.gamma.alphas - 1. / 2.) / self.gamma.betas
        return A, lmbda_diag

    def rvs(self, size=1):
        lmbdas = self.gamma.rvs()
        self.matnorm.V_diag = lmbdas
        A = self.matnorm.rvs()
        return A, lmbdas

    @property
    def base(self):
        return self.matnorm.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, K, alphas, betas = self.params
        return - self.row_dim * (0.5 * self.column_dim * np.linalg.slogdet(K)[1])\
               + Gamma(dim=self.row_dim, alphas=alphas, betas=betas).log_partition()

    def log_likelihood(self, x):
        A, lmbda_diag = x
        return MatrixNormalWithDiagonalPrecision(column_dim=self.column_dim,
                                                 row_dim=self.row_dim,
                                                 M=self.matnorm.M, V_diag=lmbda_diag,
                                                 K=self.matnorm.K).log_likelihood(A)\
               + self.gamma.log_likelihood(lmbda_diag)


class StackedMatrixNormalGammas:

    def __init__(self, size,
                 column_dim, row_dim,
                 Ms=None, Ks=None,
                 alphas=None, betas=None):

        self.size = size

        self.column_dim = column_dim
        self.row_dim = row_dim

        Ms = [None] * self.size if Ms is None else Ms
        Ks = [None] * self.size if Ks is None else Ks
        alphas = [None] * self.size if alphas is None else alphas
        betas = [None] * self.size if betas is None else betas

        self.dists = [MatrixNormalGamma(column_dim, row_dim,
                                        Ms[k], Ks[k], alphas[k], betas[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.Ms, self.Ks, self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.Ms, self.Ks, self.alphas, self.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def Ms(self):
        return np.array([dist.matnorm.M for dist in self.dists])

    @Ms.setter
    def Ms(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.M = value[k, ...]

    @property
    def Ks(self):
        return np.array([dist.matnorm.K for dist in self.dists])

    @Ks.setter
    def Ks(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.K = value[k, ...]

    @property
    def alphas(self):
        return np.array([dist.gamma.alphas for dist in self.dists])

    @alphas.setter
    def alphas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.alphas = value[k, ...]

    @property
    def betas(self):
        return np.array([dist.gamma.betas for dist in self.dists])

    @betas.setter
    def betas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.betas = value[k, ...]

    def mean(self):
        zipped = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def mode(self):
        zipped = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def rvs(self):
        zipped = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])


class TiedMatrixNormalGammas(StackedMatrixNormalGammas):

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Ks=None, alphas=None, betas=None):

        super(TiedMatrixNormalGammas, self).__init__(size, column_dim, row_dim,
                                                     Ms, Ks, alphas, betas)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        aT = np.transpose(natparam[0], (0, 2, 1))
        bT = np.transpose(natparam[1], (0, 2, 1))

        Ms = np.transpose(np.linalg.solve(bT, aT), (0, 2, 1))
        Ks = natparam[1]
        alphas = np.mean(0.5 * (natparam[2] + 1.), axis=0)
        betas = np.mean(0.5 * (natparam[3] - np.einsum('kdl,klm,kdm->kd', Ms, Ks, Ms)), axis=0)

        alphas = np.array(self.size * [alphas])
        betas = np.array(self.size * [betas])
        return Ms, Ks, alphas, betas


class SingleOutputLinearGaussianWithAutomaticRelevance:

    def __init__(self, input_dim,
                 likelihood_precision_prior,
                 parameter_precision_prior, affine=True):

        self.input_dim = input_dim
        self.affine = affine

        self.likelihood_precision_prior = likelihood_precision_prior
        self.parameter_precision_prior = parameter_precision_prior

        alphas = self.parameter_precision_prior.rvs()
        self.parameter_prior = GaussianWithPrecision(dim=input_dim,
                                                     mu=np.zeros((self.input_dim, )),
                                                     lmbda=np.diag(alphas))

        self.likelihood_precision_posterior = deepcopy(likelihood_precision_prior)
        self.parameter_precision_posterior = deepcopy(parameter_precision_prior)
        self.parameter_posterior = deepcopy(self.parameter_prior)

        beta = self.likelihood_precision_prior.rvs()
        self.likelihood_known_precision = SingleOutputLinearGaussianWithKnownPrecision(column_dim=input_dim,
                                                                                       lmbda=beta,
                                                                                       affine=affine)

        coef = self.parameter_prior.rvs()
        self.likelihood_known_mean = SingleOutputLinearGaussianWithKnownMean(column_dim=input_dim,
                                                                             W=coef, affine=affine)

        self.likelihood = LinearGaussianWithPrecision(column_dim=input_dim, row_dim=1,
                                                      A=np.expand_dims(coef, axis=0),
                                                      lmbda=np.diag(beta), affine=affine)

    @property
    def params(self):
        return self.A, self.lmbda

    @params.setter
    def params(self, values):
        self.A, self.lmbda = values

    @property
    def A(self):
        return self.likelihood.A

    @A.setter
    def A(self, value):
        # value is a single row 1d-array
        self.likelihood.A = np.expand_dims(value, axis=0)

    @property
    def lmbda(self):
        return self.likelihood.lmbda

    @lmbda.setter
    def lmbda(self, value):
        # value is a 1d-array
        self.likelihood.lmbda = np.diag(value)

    @property
    def sigma(self):
        return self.likelihood.sigma

    def predict(self, x):
        return self.likelihood.predict(x)

    def mean(self, x):
        return self.likelihood.mean(x)

    def mode(self, x):
        return self.likelihood.mode(x)

    def rvs(self, x):
        return self.likelihood.rvs(x)

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            yi = np.expand_dims(y, axis=1)
            return self.likelihood.log_likelihood(x, yi)
        else:
            return list(map(self.log_likelihood, x, y))

    def _em(self, x, y, w=None, nb_iter=10):
        # self.likelihood_precision_posterior = deepcopy(self.likelihood_precision_prior)
        # self.parameter_precision_posterior = deepcopy(self.parameter_precision_prior)
        # self.parameter_posterior = deepcopy(self.parameter_prior)

        for i in range(nb_iter):
            # variational e-step

            # parameter posterior
            alphas = self.parameter_precision_posterior.mean()
            self.parameter_prior.lmbda = np.diag(alphas)

            beta = self.likelihood_precision_posterior.mean()
            self.likelihood_known_precision.lmbda = beta

            stats = self.likelihood_known_precision.statistics(x, y) if w is None\
                    else self.likelihood_known_precision.weighted_statistics(x, y, w)
            self.parameter_posterior.nat_param = self.parameter_prior.nat_param + stats

            # variatinoal m-step

            # likelihood precision posterior
            coef = self.parameter_posterior.mean()
            self.likelihood_known_mean.W = coef

            stats = self.likelihood_known_mean.statistics(x, y) if w is None\
                    else self.likelihood_known_mean.weighted_statistics(x, y, w)
            self.likelihood_precision_posterior.nat_param = self.likelihood_precision_prior.nat_param + stats

            # parameter precision posterior
            parameter_likelihood = GaussianWithKnownMeanAndDiagonalPrecision(dim=self.input_dim)

            stats = parameter_likelihood.statistics(coef)
            self.parameter_precision_posterior.nat_param = self.parameter_precision_prior.nat_param + stats

    def em(self, x, y, w=None, **kwargs):
        nb_iter = kwargs.get('nb_iter', 10)
        self._em(x, y, w, nb_iter)

        values = kwargs.get('values', 'mode')
        if values == 'mode':
            coef = self.parameter_posterior.mode()
            beta = self.likelihood_precision_posterior.mode()
        else:
            coef = self.parameter_posterior.rvs()
            beta = self.likelihood_precision_posterior.rvs()
        self.A, self.lmbda = coef, beta


class MultiOutputLinearGaussianWithAutomaticRelevance:

    def __init__(self, input_dim, output_dim,
                 likelihood_precision_prior,
                 parameter_precision_prior, affine=True):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.affine = affine

        self.dists = []
        for i in range(self.output_dim):
            dist = SingleOutputLinearGaussianWithAutomaticRelevance(input_dim,
                                                                    likelihood_precision_prior,
                                                                    parameter_precision_prior,
                                                                    affine)
            self.dists.append(dist)

    @property
    def params(self):
        return self.A, self.lmbda

    @params.setter
    def params(self, values):
        self.A = values[0]
        self.lmbda = values[1]

    @property
    def A(self):
        return np.vstack([dist.A for dist in self.dists])

    @A.setter
    def A(self, values):
        for i, dist in enumerate(self.dists):
            dist.A = values[i]

    @property
    def lmbda(self):
        lmbdas = [dist.lmbda for dist in self.dists]
        return sc.linalg.block_diag(*lmbdas)

    @lmbda.setter
    def lmbda(self, values):
        diags = np.diag(values)
        for i, dist in enumerate(self.dists):
            dist.lmbda = np.atleast_1d(diags[i])

    def predict(self, x):
        return np.hstack([dist.predict(x) for dist in self.dists])

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        lmbda_chol_inv = 1. / np.sqrt(self.lmbda)
        return self.mean(x) + npr.normal(size=self.output_dim).dot(lmbda_chol_inv.T)

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            log_lik = np.zeros((len(x), ))
            for i, dist in enumerate(self.dists):
                log_lik += dist.log_likelihood(x, y[:, i])
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    def em(self, x, y, w=None, **kwargs):
        for i, dist in enumerate(self.dists):
            dist.em(x, y[:, i], w, **kwargs)


class StackedMultiOutputLinearGaussianWithAutomaticRelevance:

    def __init__(self, stack_size, input_dim, output_dim,
                 likelihood_precision_prior,
                 parameter_precision_prior, affine=True):

        self.stack_size = stack_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.affine = affine

        self.stack = []
        for k in range(self.stack_size):
            dist = MultiOutputLinearGaussianWithAutomaticRelevance(input_dim, output_dim,
                                                                   likelihood_precision_prior,
                                                                   parameter_precision_prior,
                                                                   affine)
            self.stack.append(dist)

    @property
    def params(self):
        return self.As, self.lmbdas

    @params.setter
    def params(self, values):
        self.As = values[0]
        self.lmbdas = values[1]

    @property
    def As(self):
        return np.array([dist.A for dist in self.stack])

    @As.setter
    def As(self, values):
        for k, dist in enumerate(self.stack):
            dist.A = values[k]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.stack])

    @lmbdas.setter
    def lmbdas(self, values):
        for k, dist in enumerate(self.stack):
            dist.lmbda = values[k]

    def predict(self, z, x):
        return self.stack[z].predict(x)

    def mean(self, z, x):
        return self.predict(z, x)

    def mode(self, z, x):
        return self.predict(z, x)

    def rvs(self, z, x):
        return self.stack[z].rvs(x)

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            log_lik = np.zeros((len(x), self.stack_size))
            for k, dist in enumerate(self.stack):
                log_lik[:, k] = dist.log_likelihood(x, y)
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    def em(self, x, y, w=None, **kwargs):
        for k, dist in enumerate(self.stack):
            dist.em(x, y, w[:, k], **kwargs)

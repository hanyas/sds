import numpy as np

from sds.distributions.gaussian import GaussianWithPrecision
from sds.distributions.gaussian import StackedGaussiansWithPrecision
from sds.distributions.gaussian import GaussianWithDiagonalPrecision
from sds.distributions.gaussian import StackedGaussiansWithDiagonalPrecision

from sds.distributions.matrix import MatrixNormalWithPrecision
from sds.distributions.matrix import StackedMatrixNormalWithPrecision
from sds.distributions.matrix import MatrixNormalWithDiagonalPrecision
from sds.distributions.matrix import StackedMatrixNormalWithDiagonalPrecision

from sds.distributions.wishart import Wishart
from sds.distributions.gamma import Gamma

from sds.utils.general import Statistics as Stats

from functools import partial


class NormalWishart:

    def __init__(self, dim, mu=None, kappa=None,
                 psi=None, nu=None):

        self.dim = dim

        self.gaussian = GaussianWithPrecision(dim=dim, mu=mu)
        self.wishart = Wishart(dim=dim, psi=psi, nu=nu)
        self.kappa = kappa

    @property
    def params(self):
        return self.gaussian.mu, self.kappa,\
               self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappa,\
        self.wishart.psi, self.wishart.nu = values

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
        return self.gaussian.mode(), self.wishart.mode()

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


class StackedNormalWishart:

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
        list_params = list(zip(*params))
        list_natparams = [dist.std_to_nat(par) for dist, par in zip(self.dists, list_params)]
        stacked_natparams = Stats(map(partial(np.stack, axis=0), zip(*list_natparams)))
        return stacked_natparams

    def nat_to_std(self, natparam):
        list_natparams = list(zip(*natparam))
        list_params = [dist.nat_to_std(par) for dist, par in zip(self.dists, list_natparams)]
        stacked_params = tuple(map(partial(np.stack, axis=0), zip(*list_params)))
        return stacked_params

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
        raise NotImplementedError


class TiedNormalWishart:

    def __init__(self, size, dim,
                 mus=None, kappas=None,
                 psi=None, nu=None):

        self.size = size
        self.dim = dim

        self.gaussians = StackedGaussiansWithPrecision(size=size, dim=dim, mus=mus)
        self.wishart = Wishart(dim=dim, psi=psi, nu=nu)
        self.kappas = kappas

    @property
    def params(self):
        return self.gaussians.mus, self.kappas,\
               self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.gaussians.mus, self.kappas,\
        self.wishart.psi, self.wishart.nu = values

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
        # nats = [kappa * m,
        #         kappa,
        #         psi^-1 + kappa * (m @ m.T),
        #         nu - d]

        a = np.einsum('k,kd->kd', params[1], params[0])
        b = params[1]
        c = np.linalg.inv(params[2]) + np.einsum('k,kd,kl->dl', params[1], params[0], params[0])
        d = params[3] - params[2].shape[0]
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        mus = np.einsum('k,kd->kd', 1. / natparam[1], natparam[0])
        kappas = natparam[1]
        psi = np.linalg.inv(natparam[2] - np.einsum('k,kd,kl->dl', kappas, mus, mus))
        nu = natparam[3] + natparam[2].shape[0]
        return mus, kappas, psi, nu

    def mean(self):
        return self.gaussians.mean(), self.wishart.mean()

    def mode(self):
        return self.gaussians.mode(), self.wishart.mode()

    def rvs(self):
        lmbda = self.wishart.rvs()
        self.gaussians.lmbdas = np.einsum('k,dl->kdl', self.kappas, lmbda)
        mu = self.gaussians.rvs()
        return mu, lmbda

    @property
    def base(self):
        return self.gaussians.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, kappas, psi, nu = self.params
        return - 0.5 * self.dim * np.sum(np.log(kappas))\
               + Wishart(dim=self.dim, psi=psi, nu=nu).log_partition()

    def log_likelihood(self, x):
        raise NotImplementedError


class NormalGamma:

    def __init__(self, dim, mu=None, kappas=None,
                 alphas=None, betas=None):

        self.dim = dim

        self.gaussian = GaussianWithDiagonalPrecision(dim=dim, mu=mu)
        self.gamma = Gamma(dim=dim, alphas=alphas, betas=betas)
        self.kappas = kappas

    @property
    def params(self):
        return self.gaussian.mu, self.kappas,\
               self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappas,\
        self.gamma.alphas, self.gamma.betas = values

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
        # nats = [kappa * m,
        #         kappa,
        #         2 * alpha - 1,
        #         2 * beta + kappa * m * m]

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
        return self.gaussian.mode(), self.gamma.mode()

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


class StackedNormalGamma:

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
        list_params = list(zip(*params))
        list_natparams = [dist.std_to_nat(par) for dist, par in zip(self.dists, list_params)]
        stacked_natparams = Stats(map(partial(np.stack, axis=0), zip(*list_natparams)))
        return stacked_natparams

    def nat_to_std(self, natparam):
        list_natparams = list(zip(*natparam))
        list_params = [dist.nat_to_std(par) for dist, par in zip(self.dists, list_natparams)]
        stacked_params = tuple(map(partial(np.stack, axis=0), zip(*list_params)))
        return stacked_params

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
        raise NotImplementedError


class TiedNormalGamma:

    def __init__(self, size, dim,
                 mus=None, kappas=None,
                 alphas=None, betas=None):

        self.size = size
        self.dim = dim

        self.gaussians = StackedGaussiansWithDiagonalPrecision(size=size, dim=dim, mus=mus)
        self.gamma = Gamma(dim=dim, alphas=alphas, betas=betas)
        self.kappas = kappas

    @property
    def params(self):
        return self.gaussians.mus, self.kappas,\
               self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.gaussians.mus, self.kappas, \
        self.gamma.alphas, self.gamma.betas = values

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
        # nats = [kappa * m,
        #         kappa,
        #         2 * alpha - 1,
        #         2 * beta + kappa * m * m]

        a = params[1] * params[0]
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + np.einsum('kd,kd->d', params[1], params[0]**2)
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        mus = natparam[0] / natparam[1]
        kappas = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - np.einsum('kd,kd->d', kappas, mus**2))
        return mus, kappas, alphas, betas

    def mean(self):
        return self.gaussians.mean(), self.gamma.mean()

    def mode(self):
        return self.gaussians.mode(), self.gamma.mode()

    def rvs(self):
        lmbda_diag = self.gamma.rvs()
        self.gaussians.lmbdas_diags = self.kappas * lmbda_diag
        mus = self.gaussians.rvs()
        return mus, lmbda_diag

    @property
    def base(self):
        return self.gaussians.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, kappas, alphas, betas = self.params
        return - 0.5 * np.sum(np.log(kappas))\
               + Gamma(dim=self.dim, alphas=alphas, betas=betas).log_partition()

    def log_likelihood(self, x):
        raise NotImplementedError


class MatrixNormalWishart:

    def __init__(self, input_dim, output_dim,
                 M=None, K=None, psi=None, nu=None):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.matnorm = MatrixNormalWithPrecision(input_dim,
                                                 output_dim,
                                                 M=M, K=K)
        self.wishart = Wishart(dim=output_dim, psi=psi, nu=nu)

    @property
    def dcol(self):
        return self.matnorm.dcol

    @property
    def drow(self):
        return self.matnorm.drow

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K,\
               self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K,\
        self.wishart.psi, self.wishart.nu = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [A.T @ V,
        #          -0.5 * A.T @ V @ A,
        #          -0.5 * V,
        #          0.5 * logdet(V)]
        # nats = [M @ K,
        #         K,
        #         psi^-1 + M @ K @ M.T,
        #         nu - d - 1 + l]

        a = params[0].dot(params[1])
        b = params[1]
        c = np.linalg.inv(params[2]) + params[0].dot(params[1]).dot(params[0].T)
        d = params[3] - params[2].shape[0] - 1 + params[0].shape[-1]
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        M = np.dot(natparam[0], np.linalg.pinv(natparam[1]))
        K = natparam[1]
        psi = np.linalg.inv(natparam[2] - M.dot(K).dot(M.T))
        nu = natparam[3] + natparam[2].shape[0] + 1 - natparam[0].shape[-1]
        return M, K, psi, nu

    def mean(self):
        return self.matnorm.mean(), self.wishart.mean()

    def mode(self):
        return self.matnorm.mode(), self.wishart.mode()

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
        return - 0.5 * self.drow * np.linalg.slogdet(K)[1]\
               + Wishart(dim=self.output_dim, psi=psi, nu=nu).log_partition()

    def log_likelihood(self, x):
        A, lmbda = x
        return MatrixNormalWithPrecision(input_dim=self.input_dim,
                                         output_dim=self.output_dim,
                                         M=self.matnorm.M, V=lmbda,
                                         K=self.matnorm.K).log_likelihood(A)\
               + self.wishart.log_likelihood(lmbda)


class StackedMatrixNormalWishart:

    def __init__(self, size, input_dim, output_dim,
                 Ms=None, Ks=None, psis=None, nus=None):

        self.size = size

        self.input_dim = input_dim
        self.output_dim = output_dim

        Ms = [None] * self.size if Ms is None else Ms
        Ks = [None] * self.size if Ks is None else Ks
        psis = [None] * self.size if psis is None else psis
        nus = [None] * self.size if nus is None else nus

        self.dists = [MatrixNormalWishart(input_dim, output_dim,
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
        list_params = list(zip(*params))
        list_natparams = [dist.std_to_nat(par) for dist, par in zip(self.dists, list_params)]
        stacked_natparams = Stats(map(partial(np.stack, axis=0), zip(*list_natparams)))
        return stacked_natparams

    def nat_to_std(self, natparam):
        list_natparams = list(zip(*natparam))
        list_params = [dist.nat_to_std(par) for dist, par in zip(self.dists, list_natparams)]
        stacked_params = tuple(map(partial(np.stack, axis=0), zip(*list_params)))
        return stacked_params

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
        raise NotImplementedError


class TiedMatrixNormalWishart:

    def __init__(self, size, input_dim, output_dim,
                 Ms=None, Ks=None, psi=None, nu=None):

        self.size = size
        self.input_dim = input_dim
        self.outpt_dim = output_dim

        self.matnorms = StackedMatrixNormalWithPrecision(size=size,
                                                         input_dim=input_dim,
                                                         output_dim=output_dim,
                                                         Ms=Ms, Ks=Ks)

        self.wishart = Wishart(dim=output_dim, psi=psi, nu=nu)

    @property
    def params(self):
        return self.matnorms.Ms, self.matnorms.Ks,\
               self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.matnorms.Ms, self.matnorms.Ks,\
        self.wishart.psi, self.wishart.nu = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [A.T @ V,
        #          -0.5 * A.T @ V @ A,
        #          -0.5 * V,
        #          0.5 * logdet(V)]
        # nats = [M @ K,
        #         K,
        #         psi^-1 + M @ K @ M.T,
        #         nu - d - 1 + l]

        a = np.einsum('kdl,klm->kdm', params[0], params[1])
        b = params[1]
        c = np.linalg.inv(params[2]) + np.einsum('kdl,klm,khm->dh', params[0], params[1], params[0])
        d = params[3] - params[2].shape[0]
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        Ms = np.einsum('ndl,nlh->ndh', natparam[0], np.linalg.pinv(natparam[1]))
        Ks = natparam[1]
        psi = np.linalg.inv(natparam[2] - np.einsum('kdl,klm,khm->dh', Ms, Ks, Ms))
        nu = natparam[3] + natparam[2].shape[0]
        return Ms, Ks, psi, nu

    def mean(self):
        return self.matnorms.mean(), self.wishart.mean()

    def mode(self):
        return self.matnorms.mode(), self.wishart.mode()

    def rvs(self):
        lmbda = self.wishart.rvs()
        self.matnorms.Vs = np.array([lmbda for k in range(self.size)])
        A = self.matnorms.rvs()
        return A, lmbda

    @property
    def base(self):
        return self.matnorms.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError


class MatrixNormalGamma:

    def __init__(self, input_dim, output_dim,
                 M=None, K=None, alphas=None, betas=None):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.matnorm = MatrixNormalWithDiagonalPrecision(input_dim,
                                                         output_dim,
                                                         M=M, K=K)
        self.gamma = Gamma(dim=output_dim, alphas=alphas, betas=betas)

    @property
    def dcol(self):
        return self.matnorm.dcol

    @property
    def drow(self):
        return self.matnorm.drow

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K, \
               self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K, \
        self.gamma.alphas, self.gamma.betas = values

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
        # nats = [M @ K,
        #         K,
        #         2 * alpha - 1,
        #         2 * beta + M @ K @ M.T]

        a = params[0].dot(params[1])
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + np.einsum('dl,lm,dm->d', params[0], params[1], params[0])
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        M = np.dot(natparam[0], np.linalg.pinv(natparam[1]))
        K = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - np.einsum('dl,lm,dm->d', M, K, M))
        return M, K, alphas, betas

    def mean(self):
        return self.matnorm.mean(), self.gamma.mean()

    def mode(self):
        return self.matnorm.mode(), self.gamma.mode()

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
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError


class StackedMatrixNormalGamma:

    def __init__(self, size,
                 input_dim, output_dim,
                 Ms=None, Ks=None,
                 alphas=None, betas=None):

        self.size = size

        self.input_dim = input_dim
        self.output_dim = output_dim

        Ms = [None] * self.size if Ms is None else Ms
        Ks = [None] * self.size if Ks is None else Ks
        alphas = [None] * self.size if alphas is None else alphas
        betas = [None] * self.size if betas is None else betas

        self.dists = [MatrixNormalGamma(input_dim, output_dim,
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
        list_params = list(zip(*params))
        list_natparams = [dist.std_to_nat(par) for dist, par in zip(self.dists, list_params)]
        stacked_natparams = Stats(map(partial(np.stack, axis=0), zip(*list_natparams)))
        return stacked_natparams

    def nat_to_std(self, natparam):
        list_natparams = list(zip(*natparam))
        list_params = [dist.nat_to_std(par) for dist, par in zip(self.dists, list_natparams)]
        stacked_params = tuple(map(partial(np.stack, axis=0), zip(*list_params)))
        return stacked_params

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
        raise NotImplementedError


class TiedMatrixNormalGamma:

    def __init__(self, size, input_dim, output_dim,
                 Ms=None, Ks=None, alphas=None, betas=None):

        self.size = size
        self.input_dim = input_dim
        self.outpt_dim = output_dim

        self.matnorms = StackedMatrixNormalWithDiagonalPrecision(size=size,
                                                                 input_dim=input_dim,
                                                                 output_dim=output_dim,
                                                                 Ms=Ms, Ks=Ks)

        self.gamma = Gamma(dim=output_dim, alphas=alphas, betas=betas)

    @property
    def params(self):
        return self.matnorms.Ms, self.matnorms.Ks,\
               self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.matnorms.Ms, self.matnorms.Ks,\
        self.gamma.alphas, self.gamma.betas = values

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
        # nats = [M @ K,
        #         K,
        #         2 * alpha - 1,
        #         2 * beta + M @ K @ M.T]

        a = np.einsum('kdl,klm->kdm', params[0], params[1])
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + np.einsum('kdl,klm,kdm->d', params[0], params[1], params[0])
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        aT = np.transpose(natparam[0], (0, 2, 1))
        bT = np.transpose(natparam[1], (0, 2, 1))

        Ms = np.transpose(np.linalg.solve(bT, aT), (0, 2, 1))
        Ks = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - np.einsum('kdl,klm,kdm->d', Ms, Ks, Ms))
        return Ms, Ks, alphas, betas

    def mean(self):
        return self.matnorms.mean(), self.gamma.mean()

    def mode(self):
        return self.matnorms.mode(), self.gamma.mode()

    def rvs(self):
        lmbda_diag = self.gamma.rvs()
        self.matnorms.Vs_diag = np.array([lmbda_diag for k in range(self.size)])
        A = self.matnorms.rvs()
        return A, lmbda_diag

    @property
    def base(self):
        return self.matnorms.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError

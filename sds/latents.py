import numpy as np

from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision
from sds.distributions.lingauss import StackedLinearGaussiansWithDiagonalPrecision

from sds.distributions.lingauss import LinearGaussianWithPrecision
from sds.distributions.lingauss import LinearGaussianWithDiagonalPrecision

from copy import deepcopy


class _SingleBayesianAutoRegressiveLatentBase:

    def __init__(self, ltn_dim, act_dim,
                 nb_lags, prior, likelihood=None):

        assert nb_lags > 0

        self.ltn_dim = ltn_dim
        self.act_dim = act_dim
        self.nb_lags = nb_lags

        self.input_dim = self.ltn_dim * self.nb_lags\
                         + self.act_dim + 1
        self.output_dim = self.ltn_dim

        self.prior = prior
        self.posterior = deepcopy(prior)
        self.likelihood = likelihood

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def initialize(self, x, u, **kwargs):
        pass

    def mstep(self, stats, **kwargs):
        self.posterior.nat_param = self.prior.nat_param + stats
        self.likelihood.params = self.posterior.mode()

        # yxT, xxT, yyT, n = stats
        #
        # A = np.linalg.solve(xxT, yxT.T).T
        # sigma = (yyT - A.dot(yxT.T)) / n
        #
        # from sds.utils.linalg import symmetrize
        # # numerical stabilization
        # _sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
        # assert np.allclose(_sigma, _sigma.T)
        # assert np.all(np.linalg.eigvalsh(_sigma) > 0.)
        #
        # lmbda = np.linalg.inv(_sigma)
        #
        # self.likelihood.params = A, lmbda

    # Kalman prediction
    def propagate(self, mean, covar, action):

        m, S, u = mean, covar, action

        A = self.likelihood.A[:, :self.ltn_dim]
        B = self.likelihood.A[:, self.ltn_dim:-1]
        c = self.likelihood.A[:, -1]
        Q = self.likelihood.sigma

        mp = A @ m + B @ u + c
        Sp = A @ S @ A.T + Q
        Sp = 0.5 * (Sp.T + Sp)

        return mp, Sp

    # Kalman Smoothing
    def smooth(self, nxt_mean, nxt_covar,
               filt_mean, filt_covar, action):

        mn, Sn = nxt_mean, nxt_covar
        mf, Sf = filt_mean, filt_covar
        u = action

        A = self.likelihood.A[:, :self.ltn_dim]
        B = self.likelihood.A[:, self.ltn_dim:-1]
        c = self.likelihood.A[:, -1]
        Q = self.likelihood.sigma

        G = np.linalg.solve(Q + A @ Sf @ A.T, A @ Sf).T

        ms = mf + G @ (mn - (A @ mf + B @ u + c))
        Ss = Sf + G @ (Sn - A @ Sf @ A.T - Q) @ G.T

        return ms, Ss, G


class SingleBayesianAutoRegressiveGaussianLatent(_SingleBayesianAutoRegressiveLatentBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-6 * np.eye(input_dim)
    # psi = 1e8 * np.eye(output_dim) / (output_dim + 1)
    # nu = (output_dim + 1) + output_dim + 1
    #
    # from sds.distributions.composite import MatrixNormalWishart
    # prior = MatrixNormalWishart(input_dim, output_dim,
    #                             M=M, K=K, psi=psi, nu=nu)

    def __init__(self, ltn_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(SingleBayesianAutoRegressiveGaussianLatent, self).__init__(ltn_dim, act_dim,
                                                                         nb_lags, prior, likelihood)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda = self.prior.rvs()
            self.likelihood = LinearGaussianWithPrecision(column_dim=self.input_dim,
                                                          row_dim=self.output_dim,
                                                          A=A, lmbda=lmbda, affine=True)


class SingleBayesianAutoRegressiveDiagonalGaussianLatent(_SingleBayesianAutoRegressiveLatentBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-64 * np.eye(input_dim)
    # alpha = ((obs_dim + 1) + obs_dim + 1) / 2. * np.ones((output_dim,))
    # beta = 1. / (2. * 1e16 * np.ones((output_dim,)) / (output_dim + 1))
    #
    # from sds.distributions.composite import MatrixNormalGamma
    # prior = MatrixNormalGamma(input_dim, output_dim,
    #                           M=M, K=K, alphas=alphas, betas=betas)

    def __init__(self, ltn_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(SingleBayesianAutoRegressiveDiagonalGaussianLatent, self).__init__(ltn_dim, act_dim,
                                                                                 nb_lags, prior, likelihood)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda_diag = self.prior.rvs()
            self.likelihood = LinearGaussianWithDiagonalPrecision(column_dim=self.input_dim,
                                                                  row_dim=self.output_dim,
                                                                  A=A, lmbda_diag=lmbda_diag,
                                                                  affine=True)


class _BayesianAutoRegressiveLatentBase(_SingleBayesianAutoRegressiveLatentBase):

    def __init__(self, nb_states, ltn_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(_BayesianAutoRegressiveLatentBase, self).__init__(ltn_dim, act_dim,
                                                                nb_lags, prior, likelihood)

        self.nb_states = self.nb_states

    def permute(self, perm):
        pass

    # Kalman prediction
    def propagate(self, mean, covar, action):
        pass

    # Kalman Smoothing
    def smooth(self, nxt_mean, nxt_covar,
               filt_mean, filt_covar, action):
        pass


class BayesianAutoRegressiveGaussianLatent(_BayesianAutoRegressiveLatentBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-6 * np.eye(input_dim)
    # psi = 1e8 * np.eye(output_dim) / (output_dim + 1)
    # nu = (output_dim + 1) + output_dim + 1
    #
    # from sds.distributions.composite import StackedMatrixNormalWishart
    # prior = StackedMatrixNormalWishart(nb_states, input_dim, output_dim,
    #                                    Ms=np.array([M for _ in range(nb_states)]),
    #                                    Ks=np.array([K for _ in range(nb_states)]),
    #                                    psis=np.array([psi for _ in range(nb_states)]),
    #                                    nus=np.array([nu for _ in range(nb_states)]))

    def __init__(self, nb_states, ltn_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianAutoRegressiveGaussianLatent, self).__init__(nb_states, ltn_dim, act_dim,
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
        pass


class BayesianAutoRegressiveDiagonalGaussianLatent(_BayesianAutoRegressiveLatentBase):

    # M = np.zeros((output_dim, input_dim))
    # K = 1e-64 * np.eye(input_dim)
    # alpha = ((obs_dim + 1) + obs_dim + 1) / 2. * np.ones((output_dim,))
    # beta = 1. / (2. * 1e16 * np.ones((output_dim,)) / (output_dim + 1))
    #
    # from sds.distributions.composite import StackedMatrixNormalGamma
    # prior = StackedMatrixNormalGamma(nb_states, input_dim, output_dim,
    #                                  Ms=np.array([M for _ in range(nb_states)]),
    #                                  Ks=np.array([K for _ in range(nb_states)]),
    #                                  alphas=np.array([alpha for _ in range(nb_states)]),
    #                                  betas=np.array([beta for _ in range(nb_states)]))

    def __init__(self, nb_states, ltn_dim, act_dim,
                 nb_lags, prior, likelihood=None):
        super(BayesianAutoRegressiveDiagonalGaussianLatent, self).__init__(nb_states, ltn_dim, act_dim,
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
        pass

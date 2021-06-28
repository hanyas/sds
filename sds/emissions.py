import numpy as np
import numpy.random as npr

from sds.distributions.lingauss import StackedLinearGaussiansWithPrecision
from sds.distributions.lingauss import LinearGaussianWithPrecision

from sds.utils.decorate import ensure_args_are_viable

from copy import deepcopy


class _SingleBayesianLinearGaussianEmissionBase:

    def __init__(self, ltn_dim, ems_dim,
                 prior, likelihood=None):

        self.ltn_dim = ltn_dim
        self.ems_dim = ems_dim

        self.input_dim = self.ltn_dim + 1
        self.output_dim = self.ems_dim

        self.prior = prior
        self.posterior = deepcopy(prior)
        self.likelihood = likelihood

    @property
    def params(self):
        return self.likelihood.params

    @params.setter
    def params(self, values):
        self.likelihood.params = values

    def initialize(self, y, **kwargs):
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

    # Kalman conditioning
    def condition(self, mean, covar, obs):
        m, S, y = mean, covar, obs

        H = self.likelihood.A[:, :self.ltn_dim]
        g = self.likelihood.A[:, -1]
        R = self.likelihood.sigma

        K = np.linalg.solve(R + H @ S @ H.T, H @ S).T

        mc = m + K @ (y - (H @ m + g))
        Sc = S - K @ (H @ S @ H.T + R) @ K.T
        Sc = 0.5 * (Sc + Sc.T)

        return mc, Sc

    def expected_log_liklihood(self, mean, covar, obs):
        m, S, y = mean, covar, obs

        H = self.likelihood.A[:, :self.ltn_dim]
        g = self.likelihood.A[:, -1]
        R = self.likelihood.sigma

        w = H @ m + g
        W = R + H @ S @ H.T

        L = np.linalg.cholesky(W)
        sqerr = np.linalg.inv(L) @ (w - y)
        ll = - 0.5 * self.ems_dim * np.log(2. * np.pi) \
             - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(sqerr**2)

        return ll


class SingleBayesianLinearGaussianEmission(_SingleBayesianLinearGaussianEmissionBase):

    def __init__(self, ltn_dim, ems_dim,
                 prior, likelihood=None):
        super(SingleBayesianLinearGaussianEmission, self).__init__(ltn_dim, ems_dim,
                                                                   prior, likelihood)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda = self.prior.rvs()
            self.likelihood = LinearGaussianWithPrecision(column_dim=self.input_dim,
                                                          row_dim=self.output_dim,
                                                          A=A, lmbda=lmbda,
                                                          affine=True)


class _BayesianLinearGaussianEmissionBase(_SingleBayesianLinearGaussianEmissionBase):

    def __init__(self, nb_states, ltn_dim, ems_dim,
                 prior, likelihood=None):

        super(_BayesianLinearGaussianEmissionBase, self).__init__(ltn_dim, ems_dim,
                                                                  prior, likelihood)
        self.nb_states = nb_states

    def permute(self, perm):
        pass

    # Kalman conditioning
    def condition(self, mean, covar, obs):
        pass


class BayesianLinearGaussianEmission(_BayesianLinearGaussianEmissionBase):

    def __init__(self, nb_states, ltn_dim, ems_dim,
                 prior, likelihood=None):

        super(BayesianLinearGaussianEmission, self).__init__(nb_states, ltn_dim,
                                                                 ems_dim, prior,
                                                                 likelihood)

        # Linear-Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            As, lmbdas = self.prior.rvs()
            self.likelihood = StackedLinearGaussiansWithPrecision(size=self.nb_states,
                                                                  column_dim=self.input_dim,
                                                                  row_dim=self.output_dim,
                                                                  As=As, lmbdas=lmbdas,
                                                                  affine=True)

    def permute(self, perm):
        pass

from autograd import numpy as np
from autograd.numpy import random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet

import scipy as sc
from scipy import special

from sds.utils import bfgs, relu, adam
from sds.utils import ensure_args_are_viable_lists

from sklearn.preprocessing import PolynomialFeatures


class StationaryTransition:

    def __init__(self, nb_states, reg=1e-16):
        self.nb_states = nb_states
        self.reg = reg

        _mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        _mat /= np.sum(_mat, axis=1, keepdims=True)
        self.logmat = np.log(_mat)

    @property
    def params(self):
        return (self.logmat,)

    @params.setter
    def params(self, value):
        self.logmat = value[0]

    @property
    def matrix(self):
        return np.exp(self.logmat - logsumexp(self.logmat, axis=-1, keepdims=True))

    def initialize(self, x, u):
        pass

    def sample(self, z, x, u):
        return npr.choice(self.nb_states, p=self.matrix[z, :])

    def maximum(self, z, x, u):
        return np.argmax(self.matrix[z, :])

    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = len(_x)
            _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))

            _logtrans = _logtrans[:-1, ...] if len(_x) > 1 else _logtrans
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]

    def mstep(self, gamma, x, u, nb_iters=100):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + self.reg
        _mat = counts / np.sum(counts, axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


class StickyTransition(StationaryTransition):

    def __init__(self, nb_states, alpha=1, kappa=100):
        super(StickyTransition, self).__init__(nb_states)

        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        lp = 0
        for k in range(self.nb_states):
            alpha = self.alpha * np.ones(self.nb_states)\
                    + self.kappa * (np.arange(self.nb_states) == k)
            lp += dirichlet.logpdf(self.matrix[k], alpha)
        return lp

    def mstep(self, gamma, x, u, nb_iter=100):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + self.reg
        counts += self.kappa * np.eye(self.nb_states)
        _mat = counts / counts.sum(axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


class RecurrentTransition(StickyTransition):

    def __init__(self, nb_states, dm_obs, dm_act, degree=1):
        super(RecurrentTransition, self).__init__(nb_states)

        self.dm_obs = dm_obs
        self.dm_act = dm_act
        self.dm_in = self.dm_obs + self.dm_act

        self.degree = degree

        self.nb_feat = int(sc.special.comb(self.degree + self.dm_in, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.coef = 0. * npr.randn(self.nb_states, self.nb_feat)

    @property
    def params(self):
        return super(RecurrentTransition, self).params + (self.coef, )

    @params.setter
    def params(self, value):
        self.coef = value[-1]
        super(RecurrentTransition, self.__class__).params.fset(self, value[:-1])

    def sample(self, z, x, u):
        _mat = np.squeeze(np.exp(self.log_transition(x, u)[0] + self.reg))
        return npr.choice(self.nb_states, p=_mat[z, :])

    def maximum(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0] + self.reg))
        return np.argmax(mat[z, :])

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = len(_x)
            _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))

            _in = np.hstack((_x, _u[:, :self.dm_act]))
            _logtrans += (self.basis.fit_transform(_in) @ self.coef.T)[:, None, :]

            _logtrans = _logtrans[:-1, ...] if len(_x) > 1 else _logtrans
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))

        return logtrans

    def permute(self, perm):
        super(RecurrentTransition, self).permute(perm)
        self.coef = self.coef[perm, :]

    def mstep(self, zeta, x, u, nb_iters=100):

        def _expected_log_zeta(zeta):
            elbo = self.log_prior()
            logtrans = self.log_transition(x, u)
            for _slice, _logtrans in zip(zeta, logtrans):
                elbo += np.sum(_slice * _logtrans)
            return elbo

        T = sum([_x.shape[0] for _x in x])

        def _objective(params, itr):
            self.params = params
            obj = _expected_log_zeta(zeta)
            return -obj / T

        self.params = bfgs(_objective, self.params, nb_iters=nb_iters)


class NeuralRecurrentTransition(StickyTransition):

    def __init__(self, nb_states, dm_obs, dm_act,
                 hidden_layer_sizes=(50,), nonlinearity='relu'):
        super(NeuralRecurrentTransition, self).__init__(nb_states)

        self.dm_obs = dm_obs
        self.dm_act = dm_act
        self.dm_in = self.dm_obs + self.dm_act

        layer_sizes = (self.dm_in,) + hidden_layer_sizes + (self.nb_states,)

        _stdv = np.sqrt(1. / (self.dm_in + self.nb_states))
        self.weights = [_stdv * npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(relu=relu, tanh=np.tanh)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return super(NeuralRecurrentTransition, self).params + (self.weights, self.biases, )

    @params.setter
    def params(self, value):
        self.biases = value[-1]
        self.weights = value[-2]
        super(NeuralRecurrentTransition, self.__class__).params.fset(self, value[:-2])

    def sample(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return npr.choice(self.nb_states, p=mat[z, :])

    def maximum(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            _in = np.hstack((_x, _u[:, :self.dm_act]))

            for W, b in zip(self.weights, self.biases):
                y = np.dot(_in, W) + b
                _in = self.nonlinearity(y)

            _logtrans = self.logmat[None, :, :] + y[:, None, :]

            _logtrans = _logtrans[:-1, ...] if len(_x) > 1 else _logtrans
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))

        return logtrans

    def permute(self, perm):
        super(NeuralRecurrentTransition, self).permute(perm)
        self.weights[-1] = self.weights[-1][:, perm]
        self.biases[-1] = self.biases[-1][perm]

    def mstep(self, zeta, x, u, nb_iters=100):

        def _expected_log_zeta(zeta):
            elbo = self.log_prior()
            logtrans = self.log_transition(x, u)
            for _slice, _logtrans in zip(zeta, logtrans):
                elbo += np.sum(_slice * _logtrans)
            return elbo

        T = sum([_x.shape[0] for _x in x])

        def _objective(params, itr):
            self.params = params
            obj = _expected_log_zeta(zeta)
            return -obj / T

        opt_state = self.opt_state if hasattr(self, "opt_state") else None
        self.params, self.opt_state = adam(_objective, self.params, state=opt_state,
                                           nb_iters=nb_iters, full_output=True)

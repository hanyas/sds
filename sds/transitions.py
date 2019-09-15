from autograd import numpy as np
from autograd.numpy import random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet

import scipy as sc
from scipy import special

from sds.utils import lbfgs, bfgs, adam, relu
from sds.utils import ensure_args_are_viable_lists
from sds.utils import batches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import distributions

from sklearn.preprocessing import PolynomialFeatures

to_torch = lambda arr: torch.from_numpy(arr).float()
to_npy = lambda arr: arr.detach().double().numpy()


class StationaryTransition:

    def __init__(self, nb_states, prior, **kwargs):
        self.nb_states = nb_states
        self.prior = prior

        _mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        _mat /= np.sum(_mat, axis=1, keepdims=True)
        self.logmat = np.log(_mat)

    @property
    def params(self):
        return tuple([self.logmat])

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

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, gamma, x, u, reg=1.e-16):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + reg
        _mat = counts / np.sum(counts, axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


class StickyTransition(StationaryTransition):

    def __init__(self, nb_states, prior, **kwargs):
        super(StickyTransition, self).__init__(nb_states)
        if not prior:
            prior = {'alpha': 1, 'kappa': 100}
        self.prior = prior

    def log_prior(self):
        lp = 0
        for k in range(self.nb_states):
            alpha = self.prior['alpha'] * np.ones(self.nb_states)\
                    + self.prior['kappa'] * (np.arange(self.nb_states) == k)
            lp += dirichlet.logpdf(self.matrix[k], alpha)
        return lp

    def mstep(self, gamma, x, u, reg=1.e-32):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + reg
        counts += self.prior['kappa'] * np.eye(self.nb_states)
        _mat = counts / counts.sum(axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


# class RecurrentTransition(StickyTransition):
#
#     def __init__(self, nb_states, dm_obs, dm_act, prior, degree=1):
#         super(RecurrentTransition, self).__init__(nb_states, prior)
#
#         self.dm_obs = dm_obs
#         self.dm_act = dm_act
#
#         self.degree = degree
#
#         self.nb_feat = int(sc.special.comb(self.degree + (self.dm_obs + self.dm_act), self.degree)) - 1
#         self.basis = PolynomialFeatures(self.degree, include_bias=False)
#
#         self.coef = 0. * npr.randn(self.nb_states, self.nb_feat)
#
#     @property
#     def params(self):
#         return super(RecurrentTransition, self).params + (self.coef, )
#
#     @params.setter
#     def params(self, value):
#         self.coef = value[-1]
#         super(RecurrentTransition, self.__class__).params.fset(self, value[:-1])
#
#     def sample(self, z, x, u):
#         _mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#         return npr.choice(self.nb_states, p=_mat[z, :])
#
#     def maximum(self, z, x, u):
#         mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#         return np.argmax(mat[z, :])
#
#     def permute(self, perm):
#         super(RecurrentTransition, self).permute(perm)
#         self.coef = self.coef[perm, :]
#
#     @ensure_args_are_viable_lists
#     def log_transition(self, x, u):
#         logtrans = []
#         for _x, _u in zip(x, u):
#             T = np.maximum(len(_x) - 1, 1)
#             _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))
#
#             _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
#             _logtrans += (self.basis.fit_transform(_in) @ self.coef.T)[:, None, :]
#
#             logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
#
#         return logtrans
#
#     def mstep(self, zeta, x, u, nb_iter=100):
#
#         def _expected_log_zeta(zeta):
#             elbo = self.log_prior()
#             logtrans = self.log_transition(x, u)
#             for _slice, _logtrans in zip(zeta, logtrans):
#                 elbo += np.sum(_slice * _logtrans)
#             return elbo
#
#         def _objective(params, itr):
#             self.params = params
#             obj = _expected_log_zeta(zeta)
#             return - obj
#
#         self.params = lbfgs(_objective, self.params, nb_iter=nb_iter)


class RecurrentTransition:
    def __init__(self, nb_states, dm_obs, dm_act, prior, degree=3):
        self.nb_states = nb_states

        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.degree = degree
        self.rpr = RecurrentPolyRegressor(self.nb_states, self.dm_obs,
                                          self.dm_act, prior=prior, degree=self.degree)

    @property
    def logmat(self):
        return to_npy(self.rpr.logmat.data)

    @logmat.setter
    def logmat(self, value):
        self.rpr.logmat.data = to_torch(value)

    @property
    def coef(self):
        return to_npy(self.rpr.coef.data)

    @coef.setter
    def coef(self, value):
        self.rpr.coef.data = to_torch(value)

    @property
    def params(self):
        return tuple([self.logmat, self.coef])

    @params.setter
    def params(self, value):
        self.logmat = value[0]
        self.coef = value[1]

    def initialize(self, x, u, **kwargs):
        pass

    def sample(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return npr.choice(self.nb_states, p=mat[z, :])

    def maximum(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.coef = self.coef[perm, :]

    def log_prior(self):
        return self.rpr.log_prior()

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
            _logtrans = to_npy(self.rpr.forward(to_torch(_in)))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, zeta, x, u, **kwargs):
        xu = []
        for _x, _u in zip(x, u):
            xu.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act])))

        self.rpr.fit(to_torch(np.vstack(zeta)), to_torch(np.vstack(xu)), **kwargs)


class RecurrentPolyRegressor(nn.Module):
    def __init__(self, nb_states, dm_obs, dm_act, prior, degree=3):
        super(RecurrentPolyRegressor, self).__init__()

        self.nb_states = nb_states

        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.degree = degree

        self.nb_feat = int(sc.special.comb(self.degree + (self.dm_obs + self.dm_act), self.degree))
        self.basis = PolynomialFeatures(self.degree, include_bias=True)

        self.coef = nn.Parameter(1e-4 * torch.randn(self.nb_states, self.nb_feat))

        _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat))

        self.optim = None

    def log_prior(self):
        lp = 0.
        if self.prior:
            _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
            for k in range(self.nb_states):
                alpha = self.prior['alpha'] * torch.ones(self.nb_states)\
                        + self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k, dtype=torch.float32)
                _dirichlet = torch.distributions.dirichlet.Dirichlet(alpha)
                lp += _dirichlet.log_prob(_matrix[k])
        return lp

    def forward(self, xu):
        _feat = to_torch(self.basis.fit_transform(to_npy(xu)))
        out = torch.mm(_feat, torch.transpose(self.coef, 0, 1))
        _logtrans = self.logmat[None, :, :] + out[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) + self.log_prior()

    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1.e-3):
        batch_size = xu.shape[0] if batch_size is None else batch_size
        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_iter):
            for batch in batches(batch_size, xu.shape[0]):
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch])
                loss.backward()
                self.optim.step()

            # if n % 10 == 0:
            #     print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
            #     print("Loss: {:.4f}".format(loss))


# class NeuralRecurrentTransition(StickyTransition):
#
#     def __init__(self, nb_states, dm_obs, dm_act, prior,
#                  hidden_layer_sizes=(10,), nonlinearity='relu'):
#         super(NeuralRecurrentTransition, self).__init__(nb_states, prior)
#
#         self.dm_obs = dm_obs
#         self.dm_act = dm_act
#
#         layer_sizes = (self.dm_obs + self.dm_act,) + hidden_layer_sizes + (self.nb_states,)
#
#         _stdv = np.sqrt(1. / (self.dm_obs + self.dm_act + self.nb_states))
#         self.weights = [_stdv * npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
#         self.biases = [0. * npr.randn(n) for n in layer_sizes[1:]]
#
#         nonlinearities = dict(relu=relu, tanh=np.tanh)
#         self.nonlinearity = nonlinearities[nonlinearity]
#
#     @property
#     def params(self):
#         return super(NeuralRecurrentTransition, self).params + (self.weights, self.biases, )
#
#     @params.setter
#     def params(self, value):
#         self.biases = value[-1]
#         self.weights = value[-2]
#         super(NeuralRecurrentTransition, self.__class__).params.fset(self, value[:-2])
#
#     def sample(self, z, x, u):
#         mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#         return npr.choice(self.nb_states, p=mat[z, :])
#
#     def maximum(self, z, x, u):
#         mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#         return np.argmax(mat[z, :])
#
#     def permute(self, perm):
#         super(NeuralRecurrentTransition, self).permute(perm)
#         self.weights[-1] = self.weights[-1][:, perm]
#         self.biases[-1] = self.biases[-1][perm]
#
#     @ensure_args_are_viable_lists
#     def log_transition(self, x, u):
#         logtrans = []
#         for _x, _u in zip(x, u):
#             T = np.maximum(len(_x) - 1, 1)
#             _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
#
#             for W, b in zip(self.weights, self.biases):
#                 y = np.dot(_in, W) + b
#                 _in = self.nonlinearity(y)
#
#             _logtrans = self.logmat[None, :, :] + y[:, None, :]
#             logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
#
#         return logtrans
#
#     def mstep(self, zeta, x, u, nb_iter=100):
#
#         def _expected_log_zeta(zeta):
#             elbo = self.log_prior()
#             logtrans = self.log_transition(x, u)
#             for _slice, _logtrans in zip(zeta, logtrans):
#                 elbo += np.sum(_slice * _logtrans)
#             return elbo
#
#         def _objective(params, itr):
#             self.params = params
#             obj = _expected_log_zeta(zeta)
#             return - obj
#
#         opt_state = self.opt_state if hasattr(self, "opt_state") else None
#         self.params, self.opt_state = adam(_objective, self.params, state=opt_state,
#                                            nb_iter=nb_iter, full_output=True)


class NeuralRecurrentTransition:

    def __init__(self, nb_states, dm_obs, dm_act, prior,
                 hidden_layer_sizes=(50, ), nonlinearity='relu'):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior

        sizes = [self.dm_obs + self.dm_act] + list(hidden_layer_sizes) + [self.nb_states]
        self.rnr = RecurrentNeuralRegressor(sizes, prior=prior, nonlin=nonlinearity)

    @property
    def logmat(self):
        return to_npy(self.rnr.logmat.data)

    @logmat.setter
    def logmat(self, value):
        self.rnr.logmat.data = to_torch(value)

    @property
    def weights(self):
        return [to_npy(self.rnr.layer.weight.data), to_npy(self.rnr.output.weight.data)]

    @weights.setter
    def weights(self, value):
        self.rnr.layer.weight.data = to_torch(value[0])
        self.rnr.output.weight.data = to_torch(value[1])

    @property
    def biases(self):
        return [to_npy(self.rnr.layer.bias.data), to_npy(self.rnr.output.bias.data)]

    @biases.setter
    def biases(self, value):
        self.rnr.layer.bias.data = to_torch(value[0])
        self.rnr.output.bias.data = to_torch(value[1])

    @property
    def params(self):
        return tuple([self.logmat, self.weights, self.biases])

    @params.setter
    def params(self, value):
        self.logmat = value[0]
        self.weights = value[1]
        self.biases = value[2]

    def initialize(self, x, u, **kwargs):
        pass

    def sample(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return npr.choice(self.nb_states, p=mat[z, :])

    def maximum(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:, perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_prior(self):
        return self.rnr.log_prior()

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
            _logtrans = to_npy(self.rnr.forward(to_torch(_in)))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, zeta, x, u, **kwargs):
        xu = []
        for _x, _u in zip(x, u):
            xu.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act])))

        self.rnr.fit(to_torch(np.vstack(zeta)), to_torch(np.vstack(xu)), **kwargs)


class RecurrentNeuralRegressor(nn.Module):
    def __init__(self, sizes, prior, nonlin='relu'):
        super(RecurrentNeuralRegressor, self).__init__()

        self.sizes = sizes
        self.nb_states = self.sizes[-1]

        self.prior = prior

        nlist = dict(relu=F.relu, tanh=F.tanh,
                     softmax=F.log_softmax, linear=F.linear)

        self.nonlin = nlist[nonlin]
        self.layer = nn.Linear(self.sizes[0], self.sizes[1])
        self.output = nn.Linear(self.sizes[1], self.sizes[2])

        _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat))

        self.optim = None

    def log_prior(self):
        lp = 0.
        if self.prior:
            _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
            for k in range(self.nb_states):
                alpha = self.prior['alpha'] * torch.ones(self.nb_states)\
                        + self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k, dtype=torch.float32)
                _dirichlet = torch.distributions.dirichlet.Dirichlet(alpha)
                lp += _dirichlet.log_prob(_matrix[k])
        return lp

    def forward(self, xu):
        out = self.output(self.nonlin(self.layer(xu)))
        _logtrans = self.logmat[None, :, :] + out[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) + self.log_prior()

    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1.e-3):
        batch_size = xu.shape[0] if batch_size is None else batch_size
        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_iter):
            for batch in batches(batch_size, xu.shape[0]):
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch])
                loss.backward()
                self.optim.step()

            # if n % 10 == 0:
            #     print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
            #     print("Loss: {:.4f}".format(loss))

from autograd import numpy as np
from autograd.numpy import random as npr

from autograd.scipy.special import logsumexp, logit
from autograd.scipy.stats import dirichlet

from autograd.tracer import getval

import scipy as sc
from scipy import special
from scipy.stats import multivariate_normal as mvn

from sds.utils import lbfgs, bfgs, adam, relu
from sds.utils import ensure_args_are_viable_lists

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torch.optim import Adam, SGD
from torch.utils.data import BatchSampler, SubsetRandomSampler

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()


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

    def sample(self, z, x, u, stoch=True):
        if stoch:
            return npr.choice(self.nb_states, p=self.matrix[z, :])
        else:
            return self.maximum(z, x, u)

    def maximum(self, z, x, u):
        return np.argmax(self.matrix[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]

    def log_prior(self):
        lp = 0.
        return lp

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, gamma, x, u, weights=None, reg=1e-16):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + reg
        _mat = counts / np.sum(counts, axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


class StickyTransition(StationaryTransition):

    def __init__(self, nb_states, prior, **kwargs):
        super(StickyTransition, self).__init__(nb_states, prior={})
        if not prior:
            prior = {'alpha': 1, 'kappa': 100}
        self.prior = prior

    def log_prior(self):
        lp = 0
        for k in range(self.nb_states):
            alpha = self.prior['alpha'] * np.ones(self.nb_states) + self.prior['kappa'] * (np.arange(self.nb_states) == k)
            lp += dirichlet.logpdf(self.matrix[k], alpha)
        return lp

    def mstep(self, gamma, x, u, weights=None, reg=1e-16):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + reg
        counts += self.prior['kappa'] * np.eye(self.nb_states)\
                  + (self.prior['alpha'] - 1) * np.ones((self.nb_states, self.nb_states))
        _mat = counts / counts.sum(axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


class PolyRecurrentTransition:
    def __init__(self, nb_states, dm_obs, dm_act, prior, norm=None, degree=1):
        self.nb_states = nb_states

        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        if norm is None:
            self.norm = {'mean': np.zeros((1, self.dm_obs + self.dm_act)),
                         'std': np.ones((1, self.dm_obs + self.dm_act))}
        else:
            self.norm = norm

        self.degree = degree
        self.regressor = PolyRecurrentRegressor(self.nb_states, self.dm_obs, self.dm_act,
                                                prior=self.prior, norm=self.norm, degree=self.degree)
        self.regressor.to(device)

    @property
    def logmat(self):
        return to_npy(self.regressor.logmat.data)

    @logmat.setter
    def logmat(self, value):
        self.regressor.logmat.data = to_torch(value)

    @property
    def coef(self):
        return to_npy(self.regressor.coef.data)

    @coef.setter
    def coef(self, value):
        self.regressor.coef.data = to_torch(value)

    @property
    def params(self):
        return tuple([self.logmat, self.coef])

    @params.setter
    def params(self, value):
        self.logmat = value[0]
        self.coef = value[1]

    def initialize(self, x, u, **kwargs):
        pass

    def sample(self, z, x, u, stoch=True):
        if stoch:
            mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
            return npr.choice(self.nb_states, p=mat[z, :])
        else:
            return self.maximum(z, x, u)

    def maximum(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.coef = self.coef[perm, :]

    def log_prior(self):
        self.regressor.eval()
        if self.prior:
            return to_npy(self.regressor.log_prior())
        else:
            return self.regressor.log_prior()

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        self.regressor.eval()

        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
            _logtrans = to_npy(self.regressor.forward(to_torch(_in)))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, zeta, x, u, weights=None, **kwargs):
        xu = []
        for _x, _u in zip(x, u):
            xu.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act])))

        aux = []
        if weights is not None:
            for _w, _zeta in zip(weights, zeta):
               aux.append(_w[1:, None, None] * _zeta)
            zeta = aux

        self.regressor.fit(to_torch(np.vstack(zeta)), to_torch(np.vstack(xu)), **kwargs)


class PolyRecurrentRegressor(nn.Module):
    def __init__(self, nb_states, dm_obs, dm_act, prior, norm, degree=1):
        super(PolyRecurrentRegressor, self).__init__()

        self.nb_states = nb_states

        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.norm = norm

        self.degree = degree

        self.nb_feat = int(sc.special.comb(self.degree + (self.dm_obs + self.dm_act), self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        _stdv = torch.sqrt(torch.as_tensor(1. / (self.dm_obs + self.dm_act + self.nb_states)))
        self.coef = nn.Parameter(_stdv * torch.randn(self.nb_states, self.nb_feat), requires_grad=True)

        # _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.nb_states, self.nb_states)
        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat), requires_grad=True)

        self._mean = torch.as_tensor(self.norm['mean'], dtype=torch.float32, device=device)
        self._std = torch.as_tensor(self.norm['std'], dtype=torch.float32, device=device)

        self.optim = None

    def reinit(self):
        _stdv = torch.sqrt(torch.as_tensor(1. / (self.dm_obs + self.dm_act + self.nb_states)))
        self.coef.data = _stdv * torch.randn(self.nb_states, self.nb_feat)

        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat.data = torch.log(_mat)

    def log_prior(self):
        lp = 0.
        if self.prior:
            if 'alpha' in self.prior and 'kappa' in self.prior:
                _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
                for k in range(self.nb_states):
                    alpha = self.prior['alpha'] * torch.ones(self.nb_states)\
                            + self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k, dtype=torch.float32)
                    _dirichlet = dist.dirichlet.Dirichlet(alpha.to(device))
                    lp += _dirichlet.log_prob(_matrix[k].to(device))
            if 'lp_penalty' in self.prior:
                lp = self.prior['lp_penalty'] * lp
        return lp

    def forward(self, xu):
        norm_xu = (xu - self._mean) / self._std
        _feat = to_torch(self.basis.fit_transform(to_npy(norm_xu)))
        output = torch.mm(_feat, torch.transpose(self.coef, 0, 1))
        _logtrans = self.logmat[None, :, :] + output[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) + self.log_prior()

    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1e-3):
        if self.prior and 'l2_penalty' in self.prior:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=self.prior['l2_penalty'])
        else:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=0.)

        data_size = xu.shape[0]
        batch_size = data_size if batch_size is None else batch_size
        batches = list(BatchSampler(SubsetRandomSampler(range(xu.shape[0])), batch_size, False))

        for n in range(nb_iter):
            for batch in batches:
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch])
                loss.backward()
                self.optim.step()

            # if n % 100 == 0:
            #     print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
            #     print("Loss: {:.4f}".format(loss))


class NeuralRecurrentTransition:

    def __init__(self, nb_states, dm_obs, dm_act, prior, norm=None,
                 hidden_layer_sizes=(25, ), nonlinearity='relu'):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior

        if norm is None:
            self.norm = {'mean': np.zeros((1, self.dm_obs + self.dm_act)),
                         'std': np.ones((1, self.dm_obs + self.dm_act))}
        else:
            self.norm = norm

        self.nonlinearity = nonlinearity

        sizes = [self.dm_obs + self.dm_act] + list(hidden_layer_sizes) + [self.nb_states]
        self.regressor = NeuralRecurrentRegressor(sizes, prior=self.prior, norm=self.norm,
                                                  nonlin=self.nonlinearity)
        self.regressor.to(device)

    @property
    def logmat(self):
        return to_npy(self.regressor.logmat.data)

    @logmat.setter
    def logmat(self, value):
        self.regressor.logmat.data = to_torch(value)

    @property
    def weights(self):
        return [to_npy(self.regressor.layer.weight.data), to_npy(self.regressor.output.weight.data)]

    @weights.setter
    def weights(self, value):
        self.regressor.layer.weight.data = to_torch(value[0])
        self.regressor.output.weight.data = to_torch(value[1])

    @property
    def biases(self):
        return [to_npy(self.regressor.layer.bias.data), to_npy(self.regressor.output.bias.data)]

    @biases.setter
    def biases(self, value):
        self.regressor.layer.bias.data = to_torch(value[0])
        self.regressor.output.bias.data = to_torch(value[1])

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

    def sample(self, z, x, u, stoch=True):
        if stoch:
            mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
            return npr.choice(self.nb_states, p=mat[z, :])
        else:
            return self.maximum(z, x, u)

    def maximum(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:, perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_prior(self):
        self.regressor.eval()
        if self.prior:
            return to_npy(self.regressor.log_prior())
        else:
            return self.regressor.log_prior()

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        self.regressor.eval()

        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
            _logtrans = to_npy(self.regressor.forward(to_torch(_in)))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, zeta, x, u, weights=None, **kwargs):
        xu = []
        for _x, _u in zip(x, u):
            xu.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act])))

        aux = []
        if weights is not None:
            for _w, _zeta in zip(weights, zeta):
               aux.append(_w[:-1, None, None] * _zeta)
            zeta = aux

        self.regressor.fit(to_torch(np.vstack(zeta)), to_torch(np.vstack(xu)), **kwargs)


class NeuralRecurrentRegressor(nn.Module):
    def __init__(self, sizes, prior, norm, nonlin='relu'):
        super(NeuralRecurrentRegressor, self).__init__()

        self.sizes = sizes
        self.nb_states = self.sizes[-1]

        self.prior = prior
        self.norm = norm

        nlist = dict(relu=F.relu, tanh=F.tanh,
                     softmax=F.log_softmax, linear=F.linear)

        self.nonlin = nlist[nonlin]
        self.layer = nn.Linear(self.sizes[0], self.sizes[1])
        self.output = nn.Linear(self.sizes[1], self.sizes[2])

        # _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.nb_states, self.nb_states)
        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat), requires_grad=True)

        self._mean = torch.as_tensor(self.norm['mean'], dtype=torch.float32, device=device)
        self._std = torch.as_tensor(self.norm['std'], dtype=torch.float32, device=device)

        self.optim = None

    def reinit(self):
        self.layer.reset_parameters()
        self.output.reset_parameters()

        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat.data = torch.log(_mat)

    def log_prior(self):
        lp = 0.
        if self.prior:
            if 'alpha' in self.prior and 'kappa' in self.prior:
                _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
                for k in range(self.nb_states):
                    alpha = self.prior['alpha'] * torch.ones(self.nb_states)\
                            + self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k, dtype=torch.float32)
                    _dirichlet = dist.dirichlet.Dirichlet(alpha.to(device))
                    lp += _dirichlet.log_prob(_matrix[k].to(device))
            if 'lp_penalty' in self.prior:
                lp = self.prior['lp_penalty'] * lp
        return lp

    def forward(self, xu):
        norm_xu = (xu - self._mean) / self._std
        out = self.output(self.nonlin(self.layer(norm_xu)))
        _logtrans = self.logmat[None, :, :] + out[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) + self.log_prior()

    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1e-3):
        if self.prior and 'l2_penalty' in self.prior:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=self.prior['l2_penalty'])
        else:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=0.)

        data_size = xu.shape[0]
        batch_size = data_size if batch_size is None else batch_size
        batches = list(BatchSampler(SubsetRandomSampler(range(xu.shape[0])), batch_size, False))

        for n in range(nb_iter):
            for batch in batches:
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch])
                loss.backward()
                self.optim.step()

            # if n % 10 == 0:
            #     print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
            #     print("Loss: {:.4f}".format(loss))


# class PolyRecurrentTransition(StickyTransition):
#
#     def __init__(self, nb_states, dm_obs, dm_act, prior, norm=None, degree=1):
#         super(PolyRecurrentTransition, self).__init__(nb_states, prior)
#
#         self.dm_obs = dm_obs
#         self.dm_act = dm_act
#
#         self.degree = degree
#
#         if norm is None:
#             self.norm = {'mean': np.zeros((1, self.dm_obs + self.dm_act)), 'std': np.ones((1, self.dm_obs + self.dm_act))}
#         else:
#             self.norm = norm
#
#         self.nb_feat = int(sc.special.comb(self.degree + (self.dm_obs + self.dm_act), self.degree)) - 1
#         self.basis = PolynomialFeatures(self.degree, include_bias=False)
#
#         self.coef = 0. * npr.randn(self.nb_states, self.nb_feat)
#
#     @property
#     def params(self):
#         return super(PolyRecurrentTransition, self).params + (self.coef,)
#
#     @params.setter
#     def params(self, value):
#         self.coef = value[-1]
#         super(PolyRecurrentTransition, self.__class__).params.fset(self, value[:-1])
#
#     def initialize(self, x, u, **kwargs):
#         pass
#
#     def sample(self, z, x, u, stoch=True):
#         if stoch:
#             mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#             return npr.choice(self.nb_states, p=mat[z, :])
#         else:
#             return self.maximum(z, x, u)
#
#     def maximum(self, z, x, u):
#         mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#         return np.argmax(mat[z, :])
#
#     def permute(self, perm):
#         super(PolyRecurrentTransition, self).permute(perm)
#         self.coef = self.coef[perm, :]
#
#     def log_prior(self):
#         return super(PolyRecurrentTransition, self).log_prior()
#
#     @ensure_args_are_viable_lists
#     def log_transition(self, x, u):
#         logtrans = []
#         for _x, _u in zip(x, u):
#             T = np.maximum(len(_x) - 1, 1)
#             _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))
#
#             _aux = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
#             _in = (_aux - self.norm['mean']) / self.norm['std']
#             _logtrans += (self.basis.fit_transform(_in) @ self.coef.T)[:, None, :]
#
#             logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
#
#         return logtrans
#
#     def mstep(self, zeta, x, u, weights=None, nb_iter=100):
#         aux = []
#         if weights is not None:
#             for _w, _zeta in zip(weights, zeta):
#                 aux.append(_w[:-1, None, None] * _zeta)
#             zeta = aux
#
#         def _expected_log_zeta(zeta):
#             elbo = self.log_prior()
#             logtrans = self.log_transition(x, u)
#             for _slice, _logtrans in zip(zeta, logtrans):
#                 elbo += np.sum(_slice * _logtrans)
#             return elbo
#
#         # Normalize and negate for minimization
#         T = sum([_x.shape[0] for _x in x])
#
#         def _objective(params, itr):
#             self.params = params
#             obj = _expected_log_zeta(zeta)
#             return - obj / T
#
#         self.params = lbfgs(_objective, self.params, nb_iter=nb_iter)


# class NeuralRecurrentTransition(StickyTransition):
#
#     def __init__(self, nb_states, dm_obs, dm_act, prior, norm=None, hidden_layer_sizes=(25,), nonlinearity='relu'):
#         super(NeuralRecurrentTransition, self).__init__(nb_states, prior)
#
#         self.dm_obs = dm_obs
#         self.dm_act = dm_act
#
#         if norm is None:
#             self.norm = {'mean': np.zeros((1, self.dm_obs + self.dm_act)), 'std': np.ones((1, self.dm_obs + self.dm_act))}
#         else:
#             self.norm = norm
#
#         nonlinearities = dict(relu=relu, tanh=np.tanh)
#         self.nonlinearity = nonlinearities[nonlinearity]
#
#         sizes = (self.dm_obs + self.dm_act,) + hidden_layer_sizes + (self.nb_states,)
#
#         _stdv = np.sqrt(1. / (self.dm_obs + self.dm_act + self.nb_states))
#         self.weights = [_stdv * npr.randn(m, n) for m, n in zip(sizes[:-1], sizes[1:])]
#         self.biases = [0. * npr.randn(n) for n in sizes[1:]]
#
#     @property
#     def params(self):
#         return super(NeuralRecurrentTransition, self).params + (self.weights, self.biases,)
#
#     @params.setter
#     def params(self, value):
#         self.biases = value[-1]
#         self.weights = value[-2]
#         super(NeuralRecurrentTransition, self.__class__).params.fset(self, value[:-2])
#
#     def initialize(self, x, u, **kwargs):
#         pass
#
#     def sample(self, z, x, u, stoch=True):
#         if stoch:
#             mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
#             return npr.choice(self.nb_states, p=mat[z, :])
#         else:
#             return self.maximum(z, x, u)
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
#     def log_prior(self):
#         return super(NeuralRecurrentTransition, self).log_prior()
#
#     @ensure_args_are_viable_lists
#     def log_transition(self, x, u):
#         logtrans = []
#         for _x, _u in zip(x, u):
#             T = np.maximum(len(_x) - 1, 1)
#             _aux = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
#             _in = (_aux - self.norm['mean']) / self.norm['std']
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
#     def mstep(self, zeta, x, u, weights=None, nb_iter=100):
#         aux = []
#         if weights is not None:
#             for _w, _zeta in zip(weights, zeta):
#                 aux.append(_w[:-1, None, None] * _zeta)
#             zeta = aux
#
#         def _expected_log_zeta(zeta):
#             elbo = self.log_prior()
#             logtrans = self.log_transition(x, u)
#             for _slice, _logtrans in zip(zeta, logtrans):
#                 elbo += np.sum(_slice * _logtrans)
#             return elbo
#
#         # Normalize and negate for minimization
#         T = sum([_x.shape[0] for _x in x])
#
#         def _objective(params, itr):
#             self.params = params
#             obj = _expected_log_zeta(zeta)
#             return - obj / T
#
#         opt_state = self.opt_state if hasattr(self, "opt_state") else None
#         self.params, self.opt_state = adam(_objective, self.params, state=opt_state, nb_iter=nb_iter, full_output=True)

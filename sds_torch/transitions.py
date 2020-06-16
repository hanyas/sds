import numpy as np
from numpy import random as npr

from scipy.special import logsumexp as spy_logsumexp
from scipy.stats import dirichlet

import scipy as sc
from scipy import special

import torch
import torch.nn as nn
import torch.distributions as dist

from torch.optim import Adam
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch import logsumexp

from sklearn.preprocessing import PolynomialFeatures

from sds_torch.utils import ensure_args_are_viable_lists
from sds_torch.utils import ensure_args_torch_floats
from sds_torch.utils import ensure_res_numpy_floats
from sds_torch.utils import to_float, np_float


class StationaryTransition:

    def __init__(self, nb_states, prior, **kwargs):
        self.nb_states = nb_states
        self.prior = prior

        # _mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        _mat = torch.ones((self.nb_states, self.nb_states), dtype=torch.float64)
        _mat /= torch.sum(_mat, dim=1)

        self.logmat = torch.log(_mat)

    @property
    def params(self):
        return tuple([self.logmat])

    @params.setter
    def params(self, value):
        self.logmat = value[0]

    @property
    def matrix(self):
        return torch.exp(self.logmat - logsumexp(self.logmat, dim=-1))

    def initialize(self, x, u):
        pass

    # sample transiton
    def sample(self, z, x=None, u=None):
        return npr.choice(self.nb_states, p=self.matrix[z, :])

    # most likely transition
    def likeliest(self, z, x=None, u=None):
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
            _logtrans = self.logmat.repeat(T, 1, 1)
            logtrans.append(_logtrans - logsumexp(_logtrans, dim=-1, keepdim=True))
        return logtrans

    def mstep(self, gamma, x, u, weights=None, reg=1e-16):
        counts = sum([torch.sum(_gamma, dim=0) for _gamma in gamma]) + reg
        _mat = counts / torch.sum(counts, dim=-1, keepdim=True)
        self.logmat = torch.log(_mat)


class StickyTransition(StationaryTransition):

    def __init__(self, nb_states, prior, **kwargs):
        super(StickyTransition, self).__init__(nb_states, prior={})
        if not prior:
            prior = {'alpha': 1, 'kappa': 10}
        self.prior = prior

    def log_prior(self):
        lp = 0
        for k in range(self.nb_states):
            alpha = self.prior['alpha'] * np.ones(self.nb_states)\
                    + self.prior['kappa'] * (np.arange(self.nb_states) == k)
            lp += dirichlet.logpdf(self.matrix[k], alpha)
        return lp

    def mstep(self, gamma, x, u, weights=None, reg=1e-16):
        counts = sum([np.sum(_gamma, axis=0) for _gamma in gamma]) + reg
        counts += self.prior['kappa'] * np.eye(self.nb_states)\
                  + (self.prior['alpha'] - 1) * np.ones((self.nb_states, self.nb_states))
        _mat = counts / counts.sum(axis=-1, keepdims=True)
        self.logmat = np.log(_mat)


class PolyRecurrentTransition:
    def __init__(self, nb_states, dm_obs, dm_act, prior,
                 norm=None, degree=1, device='cpu'):

        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

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
                                                prior=self.prior, norm=self.norm,
                                                degree=self.degree, device=self.device)

    @property
    @ensure_res_numpy_floats
    def logmat(self):
        return self.regressor.logmat.data

    @logmat.setter
    @ensure_args_torch_floats
    def logmat(self, value):
        self.regressor.logmat.data = value

    @property
    @ensure_res_numpy_floats
    def coef(self):
        return self.regressor.coef.data

    @coef.setter
    @ensure_args_torch_floats
    def coef(self, value):
        self.regressor.coef.data = value

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

    def likeliest(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.coef = self.coef[perm, :]

    @ensure_res_numpy_floats
    def log_prior(self):
        self.regressor.eval()
        return self.regressor.log_prior()

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        self.regressor.eval()

        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
            _logtrans = np_float(self.regressor.forward(_in))
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

        self.regressor.fit(np.vstack(zeta), np.vstack(xu), **kwargs)


class PolyRecurrentRegressor(nn.Module):
    def __init__(self, nb_states, dm_obs, dm_act, prior,
                 norm, degree=1, device=torch.device('cpu')):
        super(PolyRecurrentRegressor, self).__init__()

        self.device = device

        self.nb_states = nb_states

        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.norm = norm

        self.degree = degree

        self.nb_feat = int(sc.special.comb(self.degree + (self.dm_obs + self.dm_act), self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        _stdv = torch.sqrt(torch.as_tensor(1. / (self.dm_obs + self.dm_act + self.nb_states)))
        self.coef = nn.Parameter(_stdv * torch.randn(self.nb_states, self.nb_feat), requires_grad=True).to(self.device)

        # _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.nb_states, self.nb_states)
        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=-1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat), requires_grad=True).to(self.device)

        self._mean = torch.as_tensor(self.norm['mean'], dtype=torch.float32).to(self.device)
        self._std = torch.as_tensor(self.norm['std'], dtype=torch.float32).to(self.device)

        if self.prior:
            if 'alpha' in self.prior and 'kappa' in self.prior:
                self._concentration = torch.zeros(self.nb_states, self.nb_states, dtype=torch.float32)
                for k in range(self.nb_states):
                    self._concentration[k, ...] = self.prior['alpha'] * torch.ones(self.nb_states)\
                            + self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k, dtype=torch.float32)
                self._dirichlet = _dirichlet = dist.dirichlet.Dirichlet(self._concentration.to(self.device))

        self.optim = None

    @torch.no_grad()
    def reset(self):
        _stdv = torch.sqrt(torch.as_tensor(1. / (self.dm_obs + self.dm_act + self.nb_states)))
        self.coef.data = (_stdv * torch.randn(self.nb_states, self.nb_feat)).to(self.device)

        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=-1, keepdim=True)
        self.logmat.data = torch.log(_mat).to(self.device)

    def log_prior(self):
        lp = torch.as_tensor(0., device=self.device)
        if self.prior:
            if hasattr(self, '_dirichlet'):
                _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
                lp += self._dirichlet.log_prob(_matrix.to(self.device)).sum()
        return lp

    def propagate(self, xu):
        norm_xu = (xu - self._mean) / self._std
        _feat = to_float(self.basis.fit_transform(np_float(norm_xu))).to(self.device)
        return torch.mm(_feat, torch.transpose(self.coef, 0, 1))

    @ensure_args_torch_floats
    def forward(self, xu):
        output = self.propagate(xu)
        _logtrans = self.logmat[None, :, :] + output[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu, batch_size, set_size):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) * set_size / batch_size + self.log_prior()

    @ensure_args_torch_floats
    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1e-3):
        if self.prior and 'l2_penalty' in self.prior:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=self.prior['l2_penalty'])
        else:
            self.optim = Adam(self.parameters(), lr=lr)

        set_size = xu.shape[0]
        batch_size = set_size if batch_size is None else batch_size
        batches = list(BatchSampler(SubsetRandomSampler(range(set_size)), batch_size, True))

        for n in range(nb_iter):
            for batch in batches:
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch], batch_size, set_size)
                loss.backward()
                self.optim.step()

            # if n % 100 == 0:
            #     print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
            #     print("Loss: {:.4f}".format(loss))


class NeuralRecurrentTransition:

    def __init__(self, nb_states, dm_obs, dm_act, prior, norm=None,
                 hidden_layer_sizes=(25, ), nonlinearity='relu', device='cpu'):

        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

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
                                                  nonlin=self.nonlinearity, device=self.device)

    @property
    @ensure_res_numpy_floats
    def logmat(self):
        return self.regressor.logmat.data

    @logmat.setter
    @ensure_args_torch_floats
    def logmat(self, value):
        self.regressor.logmat.data = value

    @property
    @ensure_res_numpy_floats
    def weights(self):
        return [self.regressor.layer.weight.data, self.regressor.output.weight.data]

    @weights.setter
    @ensure_args_torch_floats
    def weights(self, value):
        self.regressor.layer.weight.data = value[0]
        self.regressor.output.weight.data = value[1]

    @property
    @ensure_res_numpy_floats
    def biases(self):
        return [self.regressor.layer.bias.data, self.regressor.output.bias.data]

    @biases.setter
    @ensure_args_torch_floats
    def biases(self, value):
        self.regressor.layer.bias.data = value[0]
        self.regressor.output.bias.data = value[1]

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

    def likeliest(self, z, x, u):
        mat = np.squeeze(np.exp(self.log_transition(x, u)[0]))
        return np.argmax(mat[z, :])

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:, perm]
        self.biases[-1] = self.biases[-1][perm]

    @ensure_res_numpy_floats
    def log_prior(self):
        self.regressor.eval()
        return self.regressor.log_prior()

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        self.regressor.eval()

        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _in = np.hstack((_x[:T, :], _u[:T, :self.dm_act]))
            _logtrans = np_float(self.regressor.forward(_in))
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

        self.regressor.fit(np.vstack(zeta), np.vstack(xu), **kwargs)


class NeuralRecurrentRegressor(nn.Module):
    def __init__(self, sizes, prior, norm, nonlin='relu',
                 device=torch.device('cpu')):
        super(NeuralRecurrentRegressor, self).__init__()

        self.device = device

        self.sizes = sizes
        self.nb_states = self.sizes[-1]

        self.prior = prior
        self.norm = norm

        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus)
        self.nonlin = nlist[nonlin]

        _layers = []
        for n in range(len(self.sizes) - 2):
            _layers.append(nn.Linear(self.sizes[n], self.sizes[n+1]))
            _layers.append(self.nonlin())
        _output = _layers.append(nn.Linear(self.sizes[-2], self.sizes[-1], bias=False))

        self.layers = nn.Sequential(*_layers).to(self.device)

        # _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.nb_states, self.nb_states)
        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=-1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat), requires_grad=True).to(self.device)

        self._mean = torch.as_tensor(self.norm['mean'], dtype=torch.float32).to(self.device)
        self._std = torch.as_tensor(self.norm['std'], dtype=torch.float32).to(self.device)

        if self.prior:
            if 'alpha' in self.prior and 'kappa' in self.prior:
                self._concentration = torch.zeros(self.nb_states, self.nb_states, dtype=torch.float32)
                for k in range(self.nb_states):
                    self._concentration[k, ...] = self.prior['alpha'] * torch.ones(self.nb_states)\
                            + self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k, dtype=torch.float32)
                self._dirichlet = dist.dirichlet.Dirichlet(self._concentration.to(self.device))

        self.optim = None

    @torch.no_grad()
    def reset(self):
        self.layers.reset_parameters()

        _mat = torch.ones(self.nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=-1, keepdim=True)
        self.logmat.data = torch.log(_mat).to(self.device)

    def log_prior(self):
        lp = torch.as_tensor(0., device=self.device)
        if self.prior:
            if hasattr(self, '_dirichlet'):
                _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat, dim=-1, keepdim=True))
                lp += self._dirichlet.log_prob(_matrix.to(self.device)).sum()
        return lp

    def normalize(self, xu):
        return (xu - self._mean) / self._std

    def propagate(self, xu):
        out = self.normalize(xu)
        return self.layers.forward(out)

    @ensure_args_torch_floats
    def forward(self, xu):
        out = self.propagate(xu)
        _logtrans = self.logmat[None, :, :] + out[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu, batch_size, set_size):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) * set_size / batch_size + self.log_prior()

    @ensure_args_torch_floats
    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1e-3):
        if self.prior and 'l2_penalty' in self.prior:
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=self.prior['l2_penalty'])
        else:
            self.optim = Adam(self.parameters(), lr=lr)

        set_size = xu.shape[0]
        batch_size = set_size if batch_size is None else batch_size
        batches = list(BatchSampler(SubsetRandomSampler(range(set_size)), batch_size, True))

        for n in range(nb_iter):
            for batch in batches:
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch], batch_size, set_size)
                loss.backward()
                self.optim.step()

                # if n % 10 == 0:
                #     print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
                #     print("Loss: {:.4f}".format(loss))

from abc import ABC

import numpy as np
from numpy import random as npr

import scipy as sc
from scipy.special import logsumexp

import torch
import torch.nn as nn
import torch.distributions as dist

from torch.optim import Adam, SGD
from torch.utils.data import BatchSampler, SubsetRandomSampler

from sds.utils.decorate import to_float, np_float
from sds.utils.decorate import ensure_args_are_viable_lists
from sds.utils.decorate import ensure_args_torch_floats
from sds.utils.decorate import ensure_return_numpy_floats

from sklearn.preprocessing import PolynomialFeatures


class StationaryTransition:

    def __init__(self, nb_states, reg=1e-8):
        self.nb_states = nb_states

        self.reg = reg

        # mat = 0.95 * np.eye(self.nb_states)\
        #       + 0.05 * npr.rand(self.nb_states, self.nb_states)
        mat = np.ones((self.nb_states, self.nb_states))
        mat /= np.sum(mat, axis=1, keepdims=True)
        self.logmat = np.log(mat)

    @property
    def matrix(self):
        return np.exp(self.logmat - logsumexp(self.logmat, axis=-1, keepdims=True))

    @matrix.setter
    def matrix(self, value):
        value /= np.sum(value, axis=1, keepdims=True)
        self.logmat = np.log(value)

    @property
    def params(self):
        return self.logmat

    @params.setter
    def params(self, value):
        self.logmat = value

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]

    def initialize(self, x, u):
        pass

    def likeliest(self, z, x=None, u=None):
        return np.argmax(self.matrix[z, :])

    def sample(self, z, x=None, u=None):
        return npr.choice(self.nb_states, p=self.matrix[z, :])

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))
            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, p, x, u, weights=None):
        counts = sum([np.sum(_p, axis=0) for _p in p]) + self.reg
        mat = counts / np.sum(counts, axis=-1, keepdims=True)
        self.logmat = np.log(mat)


class StickyTransition(StationaryTransition):

    def __init__(self, nb_states, reg, **kwargs):
        super(StickyTransition, self).__init__(nb_states, reg=reg)

        self.alpha = kwargs.get('alpha', 1.)
        self.kappa = kwargs.get('kappa', .1)

    def mstep(self, p, x, u, weights=None):
        counts = sum([np.sum(_p, axis=0) for _p in p]) + self.reg
        counts += self.kappa * np.eye(self.nb_states)\
                  + (self.alpha - 1) * np.ones((self.nb_states, self.nb_states))
        mat = counts / counts.sum(axis=-1, keepdims=True)
        self.logmat = np.log(mat)


class AugmentedTransition:
    def __init__(self, nb_states, obs_dim, act_dim,
                 prior=None, norm=None, device='cpu', **kwargs):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.norm = norm if norm is not None else\
            {'mean': np.zeros((1, self.obs_dim + self.act_dim)),
             'std': np.ones((1, self.obs_dim + self.act_dim))}

        self.prior = prior if prior is not None else\
            {'alpha': 1., 'kappa': 0.}

        use_gpu = (device == 'gpu' and torch.cuda.is_available())
        self.device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

        self.regressor = None

    def initialize(self, x, u, **kwargs):
        pass

    def permute(self, perm):
        self.regressor.permute(perm)

    def matrix(self, x, u):
        return np.squeeze(np.exp(self.log_transition(x, u)[0]))

    def likeliest(self, z, x, u):
        mat = self.matrix(x, u)
        return np.argmax(mat[z, :])

    def sample(self, z, x, u):
        mat = self.matrix(z, x, u)
        return npr.choice(self.nb_states, p=mat[z, :])

    @ensure_args_are_viable_lists
    def log_transition(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = np.maximum(len(_x) - 1, 1)
            input = np.hstack((_x[:T, :], _u[:T, :self.act_dim]))
            output = np_float(self.regressor.predict(input))
            logtrans.append(output - logsumexp(output, axis=-1, keepdims=True))
        return logtrans

    def mstep(self, zeta, x, u, weights=None, **kwargs):
        xu = []
        for _x, _u in zip(x, u):
            xu.append(np.hstack((_x[:-1, :], _u[:-1, :self.act_dim])))

        self.regressor.fit(np.vstack(zeta), np.vstack(xu), **kwargs)


class ParametricAugmentedTransition(AugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim,
                 prior=None, norm=None, device='cpu', **kwargs):
        super(ParametricAugmentedTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                            prior, norm, device, **kwargs)

        self.regressor = ParametricAugmentationRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                                         self.prior, self.norm, self.device, **kwargs)


class ParametricAugmentationRegressor(nn.Module):
    def __init__(self, nb_states, obs_dim, act_dim,
                 prior, norm, device, **kwargs):
        super(ParametricAugmentationRegressor, self).__init__()

        self.device = device

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Sticky parameters
        self.alpha = torch.tensor(prior['alpha'], dtype=torch.float32, device=self.device)
        self.kappa = torch.tensor(prior['kappa'], dtype=torch.float32, device=self.device)

        # Normalization parameters
        self.mean = torch.as_tensor(norm['mean'], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(norm['std'], dtype=torch.float32, device=self.device)

        self.optim = None

        self.dirichlets = []

        alphas = self.alpha * torch.ones(self.nb_states, dtype=torch.float32, device=self.device)
        for k in range(self.nb_states):
            kappas = self.kappa * torch.as_tensor(torch.arange(self.nb_states) == k,
                                                  dtype=torch.float32, device=self.device)
            self.dirichlets.append(dist.Dirichlet(alphas + kappas, validate_args=True))

    def standardize(self, xu):
        return (xu - self.mean) / self.std

    def permute(self, perm):
        raise NotImplementedError

    @torch.no_grad()
    def reset(self):
        raise NotImplementedError

    def log_prior(self, params):
        matrix = (torch.exp(params) + 1e-16) \
                 / torch.sum(torch.exp(params) + 1e-16, dim=-1, keepdim=True)
        lp = 0.
        for k in range(self.nb_states):
            lp += self.dirichlets[k].log_prob(matrix[:, k, :]).sum()
        return lp

    def predict(self, xu):
        self.eval()
        return self.forward(xu)

    def forward(self, xu):
        raise NotImplementedError

    def elbo(self, zeta, xu, batch_size, set_size):
        logtrans = self.forward(xu)
        return torch.sum(torch.mean(zeta * logtrans, dim=0))\
               + 1e-3 * self.log_prior(logtrans)

    @ensure_args_torch_floats
    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=1e-3,
            method='adam', verbose=False, **kwargs):

        l2 = kwargs.get('l2', 0.)
        if method == 'adam':
            self.optim = Adam(self.parameters(), lr=lr, weight_decay=l2)
        else:
            momentum = kwargs.get('momentum', 0.)
            self.optim = SGD(self.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

        set_size = xu.shape[0]
        batch_size = set_size if batch_size is None else batch_size
        batches = list(BatchSampler(SubsetRandomSampler(range(set_size)), batch_size, True))

        for n in range(nb_iter):
            for batch in batches:
                self.optim.zero_grad()
                loss = - self.elbo(zeta[batch], xu[batch], batch_size, set_size)
                loss.backward()
                self.optim.step()

            if verbose:
                if n % 100 == 0:
                    print('Epoch: {}/{}.............'.format(n, nb_iter), end=' ')
                    print("Loss: {:.4f}".format(loss))


class SharedPolyOnlyTransition(ParametricAugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None,
                 degree=1, norm=None, device='cpu', **kwargs):
        super(SharedPolyOnlyTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                       prior, norm, device, **kwargs)

        self.degree = degree
        self.regressor = SharedPolyOnlyRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                                 prior=self.prior, degree=self.degree,
                                                 norm=self.norm, device=self.device, **kwargs)


class SharedPolyOnlyRegressor(ParametricAugmentationRegressor):
    def __init__(self, nb_states, obs_dim, act_dim,
                 prior, degree, norm, device, **kwargs):
        super(SharedPolyOnlyRegressor, self).__init__(nb_states, obs_dim, act_dim,
                                                      prior, norm, device, **kwargs)

        self.degree = degree
        self.nb_feat = int(sc.special.comb(self.degree + (self.obs_dim + self.act_dim), self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        stdv = torch.sqrt(torch.as_tensor(1. / (self.obs_dim + self.act_dim + self.nb_states)))
        self.coef = nn.Parameter(stdv * torch.randn(self.nb_states, self.nb_feat, device=self.device), requires_grad=True)

    def permute(self, perm):
        self.coef.data = self.coef.data[perm, :]

    @torch.no_grad()
    def reset(self):
        stdv = torch.sqrt(torch.as_tensor(1. / (self.obs_dim + self.act_dim + 1 + self.nb_states), device=self.device))
        self.coef.data = (stdv * torch.randn(self.nb_states, self.nb_feat, device=self.device))

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        feat = to_float(self.basis.fit_transform(np_float(input))).to(self.device)
        output = torch.mm(feat, torch.transpose(self.coef, 0, 1))
        logtrans = torch.swapaxes(torch.tile(output, (self.nb_states, 1, 1)), 0, 1)
        return logtrans - torch.logsumexp(logtrans, dim=-1, keepdim=True)


class SharedNeuralOnlyTransition(ParametricAugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None, hidden_sizes=(16, ),
                 activation='relu', norm=None, device='cpu', **kwargs):
        super(SharedNeuralOnlyTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                         prior, norm, device, **kwargs)

        self.sizes = [self.obs_dim + self.act_dim] + list(hidden_sizes) + [self.nb_states]
        self.activation = activation

        self.regressor = SharedNeuralOnlyRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                                   prior=self.prior, sizes=self.sizes,
                                                   activation=self.activation, norm=self.norm,
                                                   device=self.device, **kwargs)


class SharedNeuralOnlyRegressor(ParametricAugmentationRegressor):
    def __init__(self, nb_states, obs_dim, act_dim, prior,
                 sizes, activation, norm, device, **kwargs):
        super(SharedNeuralOnlyRegressor, self).__init__(nb_states, obs_dim, act_dim,
                                                        prior, norm, device, **kwargs)

        self.sizes = sizes
        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus)
        self.nonlin = nlist[activation]

        _layers = []
        for n in range(len(self.sizes) - 2):
            _layers.append(nn.Linear(self.sizes[n], self.sizes[n+1]))
            _layers.append(self.nonlin())
        _layers.append(nn.Linear(self.sizes[-2], self.sizes[-1], bias=False))  # output layer

        self.layers = nn.Sequential(*_layers).to(self.device)

    def permute(self, perm):
        self.layers[-1].weight.data = self.layers[-1].weight.data[perm, :]
        self.layers[-1].bias.data = self.layers[-1].bias.data[perm]

    @torch.no_grad()
    def reset(self):
        for l in self.layers:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        output = self.layers.forward(input)
        logtrans = torch.swapaxes(torch.tile(output, (self.nb_states, 1, 1)), 0, 1)
        return logtrans - torch.logsumexp(logtrans, dim=-1, keepdim=True)


class SharedPolyTransition(ParametricAugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None,
                 degree=1, norm=None, device='cpu', **kwargs):
        super(SharedPolyTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                   prior, norm, device, **kwargs)

        self.degree = degree
        self.regressor = SharedPolyRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                             prior=self.prior, degree=self.degree,
                                             norm=self.norm, device=self.device, **kwargs)


class SharedPolyRegressor(ParametricAugmentationRegressor):
    def __init__(self, nb_states, obs_dim, act_dim, prior,
                 degree, norm, device, **kwargs):
        super(SharedPolyRegressor, self).__init__(nb_states, obs_dim, act_dim,
                                                  prior, norm, device, **kwargs)

        self.degree = degree

        self.nb_feat = int(sc.special.comb(self.degree + (self.obs_dim + self.act_dim), self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        stdv = torch.sqrt(torch.as_tensor(1. / (self.obs_dim + self.act_dim + self.nb_states)))
        self.coef = nn.Parameter(stdv * torch.randn(self.nb_states, self.nb_feat, device=self.device), requires_grad=True)

        mat = torch.ones(self.nb_states, self.nb_states, device=self.device)
        mat /= torch.sum(mat, dim=-1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(mat), requires_grad=True)

    def permute(self, perm):
        self.coef.data = self.coef.data[perm, :]
        self.logmat.data = self.logmat.data[np.ix_(perm, perm)]

    @torch.no_grad()
    def reset(self):
        stdv = torch.sqrt(torch.as_tensor(1. / (self.obs_dim + self.act_dim + self.nb_states), device=self.device))
        self.coef.data = (stdv * torch.randn(self.nb_states, self.nb_feat, device=self.device))

        mat = torch.ones(self.nb_states, self.nb_states, device=self.device)
        mat /= torch.sum(mat, dim=-1, keepdim=True)
        self.logmat.data = torch.log(mat)

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        feat = to_float(self.basis.fit_transform(np_float(input))).to(self.device)
        output = torch.mm(feat, torch.transpose(self.coef, 0, 1))
        logtrans = output[:, None, :] + self.logmat[None, :, :]
        return logtrans - torch.logsumexp(logtrans, dim=-1, keepdim=True)


class SharedNeuralTransition(ParametricAugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None, hidden_sizes=(16,),
                 activation='relu', norm=None, device='cpu', **kwargs):
        super(SharedNeuralTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                     prior, norm, device, **kwargs)

        self.sizes = [self.obs_dim + self.act_dim] + list(hidden_sizes) + [self.nb_states]
        self.activation = activation

        self.regressor = SharedNeuralRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                               prior=self.prior, sizes=self.sizes,
                                               activation=self.activation, norm=self.norm,
                                               device=self.device, **kwargs)


class SharedNeuralRegressor(ParametricAugmentationRegressor):
    def __init__(self, nb_states, obs_dim, act_dim, prior,
                 sizes, activation, norm, device, **kwargs):
        super(SharedNeuralRegressor, self).__init__(nb_states, obs_dim, act_dim,
                                                    prior, norm, device, **kwargs)

        self.sizes = sizes
        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus)
        self.nonlin = nlist[activation]

        _layers = []
        for n in range(len(self.sizes) - 2):
            _layers.append(nn.Linear(self.sizes[n], self.sizes[n+1]))
            _layers.append(self.nonlin())
        _layers.append(nn.Linear(self.sizes[-2], self.sizes[-1], bias=True))  # output layer

        self.layers = nn.Sequential(*_layers).to(self.device)

        mat = torch.ones(self.nb_states, self.nb_states, device=self.device)
        mat /= torch.sum(mat, dim=-1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(mat), requires_grad=True)

    def permute(self, perm):
        self.layers[-1].weight.data = self.layers[-1].weight.data[perm, :]
        self.layers[-1].bias.data = self.layers[-1].bias.data[perm]

    @torch.no_grad()
    def reset(self):
        for l in self.layers:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

        mat = torch.ones(self.nb_states, self.nb_states, device=self.device)
        mat /= torch.sum(mat, dim=-1, keepdim=True)
        self.logmat.data = torch.log(mat)

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        output = self.layers.forward(input)
        logtrans = output[:, None, :] + self.logmat[None, :, :]
        return logtrans - torch.logsumexp(logtrans, dim=-1, keepdim=True)


# Not sure yet how to implement this one
class NonParametricAugmentedTransition(AugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None,
                 norm=None, device='cpu', **kwargs):
        super(NonParametricAugmentedTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                               prior, norm, device, **kwargs)

        self.regressor = None

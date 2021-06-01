import math
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
from sds.utils.decorate import ensure_args_torch_floats
from sds.utils.decorate import ensure_return_numpy_floats
from sds.utils.decorate import ensure_args_are_viable

from sklearn.preprocessing import PolynomialFeatures


class StationaryTransition:

    def __init__(self, nb_states, obs_dim, act_dim, **kwargs):
        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

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
        return np.argmax(self.matrix[z])

    def sample(self, z, x=None, u=None):
        return npr.choice(self.nb_states, p=self.matrix[z])

    @ensure_args_are_viable
    def log_transition(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            nb_steps = np.maximum(len(x) - 1, 1)
            logtrans = np.broadcast_to(self.logmat, (nb_steps,) + self.logmat.shape)
            return logtrans - logsumexp(logtrans, axis=-1, keepdims=True)
        else:
            def inner(x, u):
                return self.log_transition.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        eps = kwargs.get('eps', 1e-8)

        counts = sum([np.sum(_p, axis=0) for _p in p]) + eps
        mat = counts / np.sum(counts, axis=-1, keepdims=True)
        self.logmat = np.log(mat)


class StickyTransition(StationaryTransition):

    def __init__(self, nb_states, obs_dim, act_dim, **kwargs):
        super(StickyTransition, self).__init__(nb_states, obs_dim,
                                               act_dim, **kwargs)

        self.alpha = kwargs.get('alpha', 1.)
        self.kappa = kwargs.get('kappa', .1)

    def mstep(self, p, x, u, **kwargs):
        eps = kwargs.get('eps', 1e-8)

        counts = sum([np.sum(_p, axis=0) for _p in p]) + eps
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

    def matrix(self, x, u):
        return np.squeeze(np.exp(self.log_transition(x, u)))

    @property
    @ensure_return_numpy_floats
    def params(self):
        return self.regressor.params

    @params.setter
    @ensure_args_torch_floats
    def params(self, value):
        self.regressor.params = value

    def permute(self, perm):
        self.regressor.permute(perm)

    def initialize(self, x, u, **kwargs):
        pass

    def likeliest(self, z, x, u):
        mat = self.matrix(x, u)
        return np.argmax(mat[z])

    def sample(self, z, x, u):
        mat = self.matrix(x, u)
        return npr.choice(self.nb_states, p=mat[z])

    @ensure_args_are_viable
    def log_transition(self, x, u):
        if isinstance(x, np.ndarray) and isinstance(u, np.ndarray):
            nb_steps = np.maximum(len(x) - 1, 1)
            input = np.hstack((x[:nb_steps], u[:nb_steps]))
            output = np_float(self.regressor.predict(input))
            return output - logsumexp(output, axis=-1, keepdims=True)
        else:
            def inner(x, u):
                return self.log_transition.__wrapped__(self, x, u)
            return list(map(inner, x, u))

    def mstep(self, p, x, u, **kwargs):
        xu = [np.hstack((_x[:-1], _u[:-1])) for _x, _u in zip(x, u)]
        self.regressor.fit(np.vstack(p), np.vstack(xu), **kwargs)


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

        # Dirichlet parameters
        self.prior = {'alpha': torch.as_tensor(prior['alpha'], dtype=torch.float32, device=self.device),
                      'kappa': torch.as_tensor(prior['kappa'], dtype=torch.float32, device=self.device)}

        # Normalization parameters
        self.norm = {'mean': torch.as_tensor(norm['mean'], dtype=torch.float32, device=self.device),
                     'std': torch.as_tensor(norm['std'], dtype=torch.float32, device=self.device)}

        self.dirichlets = []
        alphas = self.prior['alpha'] * torch.ones(self.nb_states, dtype=torch.float32, device=self.device)
        for k in range(self.nb_states):
            kappas = self.prior['kappa'] * torch.as_tensor(torch.arange(self.nb_states) == k,
                                                           dtype=torch.float32, device=self.device)
            self.dirichlets.append(dist.Dirichlet(alphas + kappas, validate_args=True))

        self.optim = None

    def standardize(self, xu):
        mu, std = list(self.norm.values())
        return (xu - mu) / std

    def permute(self, perm):
        raise NotImplementedError

    @torch.no_grad()
    def reset_parameters(self):
        raise NotImplementedError

    def log_prior(self, params):
        matrix = (torch.exp(params) + 1e-16) \
                 / torch.sum(torch.exp(params) + 1e-16, dim=-1, keepdim=True)

        lp = torch.zeros((len(params), self.nb_states), device=self.device)
        for k in range(self.nb_states):
            lp[:, k] = self.dirichlets[k].log_prob(matrix[..., k, :])
        return lp

    def predict(self, xu):
        self.eval()
        return self.forward(xu)

    def forward(self, xu):
        raise NotImplementedError

    def elbo(self, w, xu, batch_size, set_size):
        logtrans = self.forward(xu)
        return torch.sum(torch.mean(torch.sum(w * logtrans, dim=2)
                                    + self.log_prior(logtrans), dim=0))

    @ensure_args_torch_floats
    def fit(self, w, xu, nb_iter=100, batch_size=None,
            lr=1e-3, method='adam', verbose=False, **kwargs):

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
                loss = - self.elbo(w[batch], xu[batch], batch_size, set_size)
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

        self.weight = nn.Parameter(torch.Tensor(self.nb_states, self.nb_feat), requires_grad=True).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(self.nb_states), requires_grad=True).to(self.device)

        self.reset_parameters()

    @property
    def params(self):
        return self.weight.data, self.bias.data

    @params.setter
    def params(self, value):
        self.weight.data = value[0]
        self.bias.data = value[1]

    def permute(self, perm):
        self.weight.data = self.weight.data[perm]
        self.bias.data = self.bias.data[perm]

    @torch.no_grad()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        feat = to_float(self.basis.fit_transform(np_float(input))).to(self.device)
        output = torch.einsum('...d,kd->...k', feat, self.weight) + self.bias
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
        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus, lrelu=nn.LeakyReLU)
        self.nonlin = nlist[activation]

        layers = []
        for n in range(len(self.sizes) - 2):
            layers.append(nn.Linear(self.sizes[n], self.sizes[n+1]))
            layers.append(self.nonlin())
        layers.append(nn.Linear(self.sizes[-2], self.sizes[-1]))  # output layer

        self.layers = nn.Sequential(*layers).to(self.device)

        self.reset_parameters()

    @property
    def params(self):
        return None

    @params.setter
    def params(self, value):
        pass

    def permute(self, perm):
        self.layers[-1].weight.data = self.layers[-1].weight.data[perm]
        self.layers[-1].bias.data = self.layers[-1].bias.data[perm]

    @torch.no_grad()
    def reset_parameters(self):
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

        self.weight = nn.Parameter(torch.Tensor(self.nb_states, self.nb_feat), requires_grad=True).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(self.nb_states), requires_grad=True).to(self.device)
        self.logmat = nn.Parameter(torch.Tensor(self.nb_states, self.nb_states), requires_grad=True).to(self.device)

        self.reset_parameters()

    @property
    def params(self):
        return self.weight.data, self.bias.data, self.logmat.data

    @params.setter
    def params(self, value):
        self.weight.data = value[0]
        self.bias.data = value[1]
        self.logmat.data = value[2]

    def permute(self, perm):
        self.weight.data = self.weight.data[perm]
        self.bias.data = self.bias.data[perm]
        self.logmat.data = self.logmat.data[np.ix_(perm, perm)]

    @torch.no_grad()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        mat = torch.ones(self.nb_states, self.nb_states, device=self.device)
        mat /= torch.sum(mat, dim=-1, keepdim=True)
        self.logmat.data = torch.log(mat)

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        feat = to_float(self.basis.fit_transform(np_float(input))).to(self.device)
        output = torch.einsum('...d,kd->...k', feat, self.weight) + self.bias
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
        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus, lrelu=nn.LeakyReLU)
        self.nonlin = nlist[activation]

        layers = []
        for n in range(len(self.sizes) - 2):
            layers.append(nn.Linear(self.sizes[n], self.sizes[n+1]))
            layers.append(self.nonlin())
        layers.append(nn.Linear(self.sizes[-2], self.sizes[-1]))  # output layer

        self.layers = nn.Sequential(*layers).to(self.device)
        self.logmat = nn.Parameter(torch.Tensor(self.nb_states, self.nb_states),
                                   requires_grad=True).to(self.device)

        self.reset_parameters()

    @property
    def params(self):
        return None

    @params.setter
    def params(self, value):
        pass

    def permute(self, perm):
        self.layers[-1].weight.data = self.layers[-1].weight.data[perm]
        self.layers[-1].bias.data = self.layers[-1].bias.data[perm]
        self.logmat.data = self.logmat.data[np.ix_(perm, perm)]

    @torch.no_grad()
    def reset_parameters(self):
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


class SharedNeuralEnsembleTransition(ParametricAugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None, ensemble_size=5,
                 hidden_sizes=(16,), activation='relu', norm=None, device='cpu', **kwargs):
        super(SharedNeuralEnsembleTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                             prior, norm, device, **kwargs)

        self.ensemble_size = ensemble_size
        self.layer_sizes = [self.obs_dim + self.act_dim] + list(hidden_sizes) + [self.nb_states]
        self.activation = activation

        self.regressor = SharedNeuralEnsembleRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                                       prior=self.prior, ensemble_size=self.ensemble_size,
                                                       layer_sizes=self.layer_sizes, activation=self.activation,
                                                       norm=self.norm, device=self.device, **kwargs)


class EnsembleLinear(nn.Module):

    def __init__(self, ensemble_size, input_size, output_size, input_layer=False):
        super(EnsembleLinear, self).__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = input_layer  # important for broadcasting

        self.weight = nn.Parameter(torch.Tensor(self.ensemble_size, self.output_size, self.input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(self.ensemble_size, self.output_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.ensemble_size):
            stdv = 1. / math.sqrt(self.weight[k].size(1))
            self.weight[k].data.uniform_(-stdv, stdv)
            self.bias[k].data.uniform_(-stdv, stdv)

    def forward(self, input):
        # n: sample, k: networks, l: input, d: output
        contract = 'kdl,...l->...kd' if self.input_layer else 'kdl,...kl->...kd'
        return torch.einsum(contract, self.weight, input) + self.bias


class SharedNeuralEnsembleRegressor(ParametricAugmentationRegressor):

    def __init__(self, nb_states, obs_dim, act_dim, prior, ensemble_size,
                 layer_sizes, activation, norm, device, **kwargs):
        super(SharedNeuralEnsembleRegressor, self).__init__(nb_states, obs_dim, act_dim,
                                                            prior, norm, device, **kwargs)

        self.ensemble_size = ensemble_size
        self.layer_sizes = layer_sizes
        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus, lrelu=nn.LeakyReLU)
        self.nonlin = nlist[activation]

        layers = [EnsembleLinear(self.ensemble_size, self.layer_sizes[0], self.layer_sizes[1], input_layer=True)]
        for n in range(1, len(self.layer_sizes) - 1):
            layers.append(self.nonlin())
            layers.append(EnsembleLinear(self.ensemble_size, self.layer_sizes[n], self.layer_sizes[n+1]))

        self.layers = nn.Sequential(*layers).to(self.device)
        self.logmat = nn.Parameter(torch.Tensor(self.ensemble_size, self.nb_states, self.nb_states),
                                   requires_grad=True).to(self.device)

        self.reset_parameters()

    @property
    def params(self):
        return None

    @params.setter
    def params(self, value):
        pass

    def permute(self, perm):
        for k in range(self.ensemble_size):
            self.layers[-1].weight.data[k] = self.layers[-1].weight.data[k, perm]
            self.layers[-1].bias.data[k] = self.layers[-1].bias.data[k, perm]
            self.logmat.data[k] = self.logmat.data[k][np.ix_(perm, perm)]

    @torch.no_grad()
    def reset_parameters(self):
        for l in self.layers:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

        for k in range(self.ensemble_size):
            mat = torch.ones(self.nb_states, self.nb_states, device=self.device)
            mat /= torch.sum(mat, dim=-1, keepdim=True)
            self.logmat.data[k] = torch.log(mat)

    def log_prior(self, params):
        matrix = (torch.exp(params) + 1e-16) \
                 / torch.sum(torch.exp(params) + 1e-16, dim=-1, keepdim=True)

        lp = torch.zeros((len(params), self.ensemble_size, self.nb_states), device=self.device)
        for k in range(self.nb_states):
            lp[..., k] = self.dirichlets[k].log_prob(matrix[..., k, :])
        return lp

    def predict(self, xu):
        self.eval()
        return torch.mean(self.forward(xu), dim=1)

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        output = self.layers.forward(input)
        logtrans = output[:, :, None, :] + self.logmat[None, :, :, :]
        return logtrans - torch.logsumexp(logtrans, dim=-1, keepdim=True)

    def elbo(self, w, xu, batch_size, set_size):
        wb = w.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1)  # repeat across ensemble dim
        logtrans = self.forward(xu)  # [samples, ensemble, states, states]
        return torch.sum(torch.mean(torch.sum(wb * logtrans, dim=3) + self.log_prior(logtrans), dim=(0, 1)))


class StackedNeuralTransition(ParametricAugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None, hidden_sizes=(16,),
                 activation='relu', norm=None, device='cpu', **kwargs):
        super(StackedNeuralTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                      prior, norm, device, **kwargs)

        self.sizes = [self.obs_dim + self.act_dim] + list(hidden_sizes) + [self.nb_states]
        self.activation = activation

        self.regressor = StackedNeuralRegressor(self.nb_states, self.obs_dim, self.act_dim,
                                                prior=self.prior, sizes=self.sizes,
                                                activation=self.activation, norm=self.norm,
                                                device=self.device, **kwargs)


class StackedLinear(nn.Module):

    def __init__(self, stack_size, input_size, output_size, input_layer=False):
        super(StackedLinear, self).__init__()

        self.stack_size = stack_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = input_layer  # important for broadcasting

        self.weight = nn.Parameter(torch.Tensor(self.stack_size, self.output_size, self.input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(self.stack_size, self.output_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.stack_size):
            stdv = 1. / math.sqrt(self.weight[k].size(1))
            self.weight[k].data.uniform_(-stdv, stdv)
            self.bias[k].data.uniform_(-stdv, stdv)

    def forward(self, input):
        # n: sample, k: stack, l: input, d: output
        contract = 'kdl,...l->...kd' if self.input_layer else 'kdl,...kl->...kd'
        return torch.einsum(contract, self.weight, input) + self.bias


class StackedNeuralRegressor(ParametricAugmentationRegressor):

    def __init__(self, nb_states, obs_dim, act_dim, prior,
                 sizes, activation, norm, device, **kwargs):
        super(StackedNeuralRegressor, self).__init__(nb_states, obs_dim, act_dim,
                                                     prior, norm, device, **kwargs)

        self.sizes = sizes
        nlist = dict(relu=nn.ReLU, tanh=nn.Tanh, splus=nn.Softplus, lrelu=nn.LeakyReLU)
        self.nonlin = nlist[activation]

        layers = [StackedLinear(self.nb_states, self.sizes[0], self.sizes[1], input_layer=True)]
        for n in range(1, len(self.sizes) - 1):
            layers.append(self.nonlin())
            layers.append(StackedLinear(self.nb_states, self.sizes[n], self.sizes[n+1]))

        self.layers = nn.Sequential(*layers).to(self.device)

        self.reset_parameters()

    @property
    def params(self):
        return None

    @params.setter
    def params(self, value):
        pass

    def permute(self, perm):
        for l in self.layers[:-1]:
            if isinstance(l, StackedLinear):
                l.weight.data = l.weight.data[perm]
                l.bias.data = l.bias.data[perm]
        self.layers[-1].weight.data = self.layers[-1].weight.data[np.ix_(perm, perm)]
        self.layers[-1].bias.data = self.layers[-1].bias.data[np.ix_(perm, perm)]

    @torch.no_grad()
    def reset_parameters(self):
        for l in self.layers:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

    @ensure_args_torch_floats
    def forward(self, xu):
        input = self.standardize(xu)
        logtrans = self.layers.forward(input)
        return logtrans - torch.logsumexp(logtrans, dim=-1, keepdim=True)


# Not sure yet how to implement this one
class NonParametricAugmentedTransition(AugmentedTransition):
    def __init__(self, nb_states, obs_dim, act_dim, prior=None,
                 norm=None, device='cpu', **kwargs):
        super(NonParametricAugmentedTransition, self).__init__(nb_states, obs_dim, act_dim,
                                                               prior, norm, device, **kwargs)

        self.regressor = None

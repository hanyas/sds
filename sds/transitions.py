#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: transitions
# @Date: 2019-07-30-15-30
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de
from autograd import numpy as np
from autograd.numpy import random as npr
from autograd.scipy.misc import logsumexp

import scipy as sc
from scipy import special

from sds.utils import bfgs, relu, adam
from sklearn.preprocessing import PolynomialFeatures


class StationaryTransition:

    def __init__(self, nb_states, reg=1e-16):
        self.nb_states = nb_states
        self.reg = reg

        self.mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        self.mat /= np.sum(self.mat, axis=1, keepdims=True)

    def sample(self, z):
        return npr.choice(self.nb_states, p=self.mat[z, :])

    def likelihood(self, x, u):
        trans = []
        for _x, _u in zip(x, u):
            T = len(_x)
            _trans = np.tile(self.mat[None, :, :], (T, 1, 1))

            if len(_x) > 1:
                _trans = _trans[:-1, ...]
            else:
                _trans = _trans.squeeze()

            trans.append(_trans)

        return trans

    def log_likelihood(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            T = len(_x)
            _logtrans = np.log(np.tile(self.mat[None, :, :], (T, 1, 1)))

            if len(_x) > 1:
                _logtrans = _logtrans[:-1, ...]
            else:
                _logtrans = _logtrans.squeeze()

            logtrans.append(_logtrans)

        return logtrans

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.mat = self.mat[np.ix_(perm, perm)]

    def mstep(self, joint, x, u, num_iters=100):
        counts = sum([np.sum(_joint, axis=0) for _joint in joint]) + self.reg
        self.mat = counts / np.sum(counts, axis=-1, keepdims=True)


class RecurrentTransition:

    def __init__(self, nb_states, dim_obs, dim_act, degree=1, reg=1e-16):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_in = self.dim_obs + self.dim_act
        self.degree = degree
        self.reg = reg

        mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        mat /= np.sum(mat, axis=1, keepdims=True)
        self.logmat = np.log(mat)

        self.nb_feat = int(sc.special.comb(self.degree + self.dim_in, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.par = npr.randn(self.nb_states, self.nb_feat)

    @property
    def params(self):
        return self.logmat, self.par

    @params.setter
    def params(self, value):
        self.logmat, self.par = value

    def sample(self, z, x, u):
        mat = np.exp(self.log_likelihood([x], [u])[0] + 1e-8)
        return npr.choice(self.nb_states, p=mat[z, :])

    def log_likelihood(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            _x, _u = np.atleast_2d(_x), np.atleast_2d(_u)
            _in = np.concatenate((_x, _u[:, :self.dim_act]), axis=1)

            T = len(_x)
            _logtrans = np.tile(self.logmat[None, :, :], (T, 1, 1))
            _logtrans += (self.basis.fit_transform(_in) @ self.par.T)[:, None, :]

            if len(_x) > 1:
                _logtrans = _logtrans[:-1, ...]
            else:
                _logtrans = _logtrans.squeeze()

            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))

        return logtrans

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.par = self.par[perm, :]

    def mstep(self, two_slice, x, u, num_iters=100):

        def _expected_log_two_slice(two_slice):
            elbo = self.log_prior()
            logtrans = self.log_likelihood(x, u)
            for _slice, _logtrans in zip(two_slice, logtrans):
                elbo += np.sum(_slice * _logtrans)
            return elbo

        T = sum([_x.shape[0] for _x in x])

        def _objective(params, itr):
            self.params = params
            obj = _expected_log_two_slice(two_slice)
            return -obj / T

        self.params = bfgs(_objective, self.params, num_iters=num_iters)


class RecurrentOnlyTransition:

    def __init__(self, nb_states, dim_obs, dim_act, degree=1, reg=1e-16):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_in = self.dim_obs + self.dim_act
        self.degree = degree
        self.reg = reg

        self.nb_feat = int(sc.special.comb(self.degree + self.dim_in, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.par = npr.randn(self.nb_states, self.nb_feat)
        self.offset = npr.randn(self.nb_states)

    @property
    def params(self):
        return self.par, self.offset

    @params.setter
    def params(self, value):
        self.par, self.offset = value

    def sample(self, z, x, u):
        mat = np.exp(self.log_likelihood([x], [u])[0])
        return npr.choice(self.nb_states, p=mat[z, :])

    def log_likelihood(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            _x , _u = np.atleast_2d(_x), np.atleast_2d(_u)
            _in = np.concatenate((_x, _u[:, :self.dim_act]), axis=1)

            _logtrans = (self.basis.fit_transform(_in) @ self.par.T)[:, None, :]
            _logtrans += self.offset
            _logtrans = np.tile(_logtrans, (1, self.nb_states, 1))

            if len(_x) > 1:
                _logtrans = _logtrans[:-1, ...]
            else:
                _logtrans = _logtrans.squeeze()

            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))

        return logtrans

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.par = self.par[perm, :]
        self.offset = self.offset[perm]

    def mstep(self, two_slice, x, u, num_iters=100):

        def _expected_log_two_slice(two_slice):
            elbo = self.log_prior()
            logtrans = self.log_likelihood(x, u)
            for _slice, _logtrans in zip(two_slice, logtrans):
                elbo += np.sum(_slice * _logtrans)
            return elbo

        T = sum([_x.shape[0] for _x in x])

        def _objective(params, itr):
            self.params = params
            obj = _expected_log_two_slice(two_slice)
            return -obj / T

        self.params = bfgs(_objective, self.params, num_iters=num_iters)


class NeuralRecurrentTransition:

    def __init__(self, nb_states, dim_obs, dim_act,
                 hidden_layer_sizes=(50,), nonlinearity="relu",
                 reg=1e-16):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_in = self.dim_obs + self.dim_act
        self.reg = reg

        layer_sizes = (self.dim_in,) + hidden_layer_sizes + (self.nb_states,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        mat /= np.sum(mat, axis=1, keepdims=True)
        self.logmat = np.log(mat)

        nonlinearities = dict(relu=relu, tanh=np.tanh)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.logmat, self.weights, self.biases

    @params.setter
    def params(self, value):
        self.logmat, self.weights, self.biases = value

    def sample(self, z, x, u):
        mat = np.exp(self.log_likelihood([x], [u])[0])
        return npr.choice(self.nb_states, p=mat[z, :])

    def log_likelihood(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            _x , _u = np.atleast_2d(_x), np.atleast_2d(_u)
            _in = np.concatenate((_x, _u[:, :self.dim_act]), axis=1)

            for W, b in zip(self.weights, self.biases):
                y = np.dot(_in, W) + b
                _in = self.nonlinearity(y)

            _logtrans = self.logmat[None, :, :] + y[:, None, :]

            if len(_x) > 1:
                _logtrans = _logtrans[:-1, ...]
            else:
                _logtrans = _logtrans.squeeze()

            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))

        return logtrans

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:, perm]
        self.biases[-1] = self.biases[-1][perm]

    def mstep(self, two_slice, x, u, num_iters=100):

        def _expected_log_two_slice(two_slice):
            elbo = self.log_prior()
            logtrans = self.log_likelihood(x, u)
            for _slice, _logtrans in zip(two_slice, logtrans):
                elbo += np.sum(_slice * _logtrans)
            return elbo

        T = sum([_x.shape[0] for _x in x])

        def _objective(params, itr):
            self.params = params
            obj = _expected_log_two_slice(two_slice)
            return -obj / T

        self.params = adam(_objective, self.params, num_iters=num_iters)


class NeuralRecurrentOnlyTransition:

    def __init__(self, nb_states, dim_obs , dim_act,
                 hidden_layer_sizes=(50,), nonlinearity="relu", reg=1e-16):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_in = self.dim_obs + self.dim_act
        self.reg = reg

        layer_sizes = (self.dim_in,) + hidden_layer_sizes + (self.nb_states,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(relu=relu, tanh=np.tanh)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def sample(self, z, x, u):
        mat = np.exp(self.log_likelihood([x], [u])[0])
        return npr.choice(self.nb_states, p=mat[z, :])

    def log_likelihood(self, x, u):
        logtrans = []
        for _x, _u in zip(x, u):
            _x, _u = np.atleast_2d(_x), np.atleast_2d(_u)
            _in = np.concatenate((_x, _u[:, :self.dim_act]), axis=1)

            for W, b in zip(self.weights, self.biases):
                y = np.dot(_in, W) + b
                _in = self.nonlinearity(y)

            _logtrans = np.tile(y[:, None, :], (1, self.nb_states, 1))

            if len(_x) > 1:
                _logtrans = _logtrans[:-1, ...]
            else:
                _logtrans = _logtrans.squeeze()

            logtrans.append(_logtrans - logsumexp(_logtrans, axis=-1, keepdims=True))

        return logtrans

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.weights[-1] = self.weights[-1][:, perm]
        self.biases[-1] = self.biases[-1][perm]

    def mstep(self, two_slice, x, u, num_iters=100):

        def _expected_log_two_slice(two_slice):
            elbo = self.log_prior()
            logtrans = self.log_likelihood(x, u)
            for _slice, _logtrans in zip(two_slice, logtrans):
                elbo += np.sum(_slice * _logtrans)
            return elbo

        T = sum([_x.shape[0] for _x in x])

        def _objective(params, itr):
            self.params = params
            obj = _expected_log_two_slice(two_slice)
            return -obj / T

        self.params = adam(_objective, self.params, num_iters=num_iters)
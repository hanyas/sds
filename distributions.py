import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.misc import logsumexp

import scipy as sc
from scipy import special

from scipy.stats import multinomial as cat
from scipy.stats import multivariate_normal as mvn

from inf.sds.util import random_rotation
from inf.sds.util import bfgs, adam
from inf.sds.util import relu
from inf.sds.util import multivariate_normal_logpdf

from sklearn.preprocessing import PolynomialFeatures


class CategoricalInitState:

    def __init__(self, nb_states, reg=1e-8):
        self.nb_states = nb_states
        self.reg = reg

        self.pi = np.ones((self.nb_states, )) / self.nb_states

    def sample(self):
        return np.argmax(cat(1, self.pi).rvs())

    def likelihood(self):
        return self.pi

    def log_likelihood(self):
        return np.log(self.likelihood())

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.pi = self.pi[perm]

    def mstep(self, w):
        self.pi = sum([_w for _w in w]) + self.reg
        self.pi /= np.sum(self.pi)


class StationaryTransition:

    def __init__(self, nb_states, reg=1e-16):
        self.nb_states = nb_states
        self.reg = reg

        self.mat = 0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        self.mat /= np.sum(self.mat, axis=1, keepdims=True)

    def sample(self, z):
        return npr.choice(self.nb_states, p=self.mat[z, :])

    def likelihood(self):
        return self.mat

    def log_likelihood(self):
        return np.log(self.likelihood())

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.mat = self.mat[np.ix_(perm, perm)]

    def mstep(self, joint):
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


class GaussianObservation:

    def __init__(self, nb_states, dim_obs, reg=1e-16):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.reg = reg

        self.mu = np.zeros((self.nb_states, self.dim_obs))
        for k in range(self.nb_states):
            self.mu[k, :] = mvn(mean=np.zeros((self.dim_obs, )), cov=np.eye(self.dim_obs)).rvs()

        L = npr.randn(self.nb_states, self.dim_obs, self.dim_obs)
        self.cov = np.array([L[k, ...] @ L[k, ...].T for k in range(self.nb_states)])

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, mat):
        self._cov = mat + np.array([np.eye(self.dim_obs) * self.reg])

    def sample(self, z):
        return mvn(mean=self.mu[z, :], cov=self.cov[z, ...]).rvs()

    def likelihood(self, x):
        return [np.array([mvn(mean=self.mu[k, :], cov=self.cov[k, ...]).pdf(_x)
                          for k in range(self.nb_states)]).T for _x in x]

    def log_likelihood(self, x):
        return [np.array([mvn(mean=self.mu[k, :], cov=self.cov[k, ...]).logpdf(_x)
                         for k in range(self.nb_states)]).T for _x in x]

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.mu = self.mu[perm, :]
        self.cov = self.cov[perm, ...]

    def mstep(self, x, w):
        _J = np.zeros((self.nb_states, self.dim_obs))
        _h = np.zeros((self.nb_states, self.dim_obs))
        for _x, _w in zip(x, w):
            _J += np.sum(_w[:, :, None], axis=0)
            _h += np.sum(_w[:, :, None] * _x[:, None, :], axis=0)

        self.mu = _h / _J

        sqerr = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        weight = np.zeros((self.nb_states, ))
        for _x, _w in zip(x, w):
            resid = _x[:, None, :] - self.mu
            sqerr += np.sum(_w[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(_w, axis=0)

        _sqrt_cov = np.linalg.cholesky(sqerr / weight[:, None, None])
        self.cov = np.matmul(_sqrt_cov, np.swapaxes(_sqrt_cov, -1, -2))


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, dim_obs, dim_act=0, reg=1e-8):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.reg = reg

        self.A = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        self.B = np.zeros((self.nb_states, self.dim_obs, self.dim_act))
        self.c = np.zeros((self.nb_states, self.dim_obs))

        for k in range(self.nb_states):
            self.A[k, ...] = .95 * random_rotation(self.dim_obs)
            self.B[k, ...] = npr.randn(self.dim_obs, self.dim_act)
            self.c[k, :] = npr.randn(self.dim_obs)

        L = 0.1 * npr.randn(self.nb_states, self.dim_obs, self.dim_obs)
        self.cov = np.array([L[k, ...] @ L[k, ...].T for k in range(self.nb_states)])

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, mat):
        self._cov = mat + np.array([np.eye(self.dim_obs) * self.reg])

    def mean(self, z, x, u):
        return np.einsum('kh,...h->...k', self.A[z, ...], x) +\
               np.einsum('kh,...h->...k', self.B[z, ...], u) + self.c[z, :]

    def sample(self, z, x, u):
        mu = self.mean(z, x, u)
        return mvn(mu, cov=self.cov[z, ...]).rvs()

    def likelihood(self, x, u):
        lik = []
        for _x, _u in zip(x, u):
            T = _x.shape[0]
            _lik = np.zeros((T - 1, self.nb_states))
            for k in range(self.nb_states):
                mu = self.mean(k, _x[:-1, :], _u[:-1, :self.dim_act])
                for t in range(T - 1):
                    _lik[t, k] = mvn(mean=mu[t, :], cov=self.cov[k, ...]).pdf(_x[t + 1, :])

            lik.append(_lik)

        return lik

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            T = _x.shape[0]
            _loglik = np.zeros((T - 1, self.nb_states))
            for k in range(self.nb_states):
                mu = self.mean(k, _x[:-1, :], _u[:-1, :self.dim_act])
                _loglik[:, k] = multivariate_normal_logpdf(_x[1:, :], mu, self.cov[k, ...])

            loglik.append(_loglik)

        return loglik

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self.cov = self.cov[perm, ...]

    def mstep(self, x, u, w):

        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, w):
            xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dim_act], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _J_diag = np.concatenate((self.reg * np.ones(self.dim_obs),
                                  self.reg * np.ones(self.dim_act),
                                  self.reg * np.ones(1)))
        _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
        _h = np.zeros((self.nb_states, self.dim_obs + self.dim_act + 1, self.dim_obs))

        for _x, _y, _w in zip(xs, ys, ws):
            for k in range(self.nb_states):
                wx = _x * _w[:, k:k + 1]
                _J[k] += np.dot(wx.T, _x)
                _h[k] += np.dot(wx.T, _y)

        mu = np.linalg.solve(_J, _h)
        self.A = np.swapaxes(mu[:, :self.dim_obs, :], 1, 2)
        self.B = np.swapaxes(mu[:, self.nb_states:self.nb_states + self.dim_act, :], 1, 2)
        self.c = mu[:, -1, :]

        sqerr = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        weight = 1e-8 * np.ones(self.nb_states)
        for _x, _y, _w in zip(xs, ys, ws):
            yhat = np.matmul(_x[None, :, :], mu)
            resid = _y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
            weight += np.sum(_w, axis=0)

        self.cov = sqerr / weight[:, None, None]


class AutoRegressiveGaussianFullObservation:

    def __init__(self, nb_states, dim_obs, dim_act=0, reg=1e-8):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.reg = reg

        self.A = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        self.B = np.zeros((self.nb_states, self.dim_obs, self.dim_act))
        self.c = np.zeros((self.nb_states, self.dim_obs))

        self.K = np.zeros((self.nb_states, self.dim_act, self.dim_obs))
        self.kff = np.zeros((self.nb_states, self.dim_act))

        for k in range(self.nb_states):
            self.A[k, ...] = .95 * random_rotation(self.dim_obs)
            self.B[k, ...] = npr.randn(self.dim_obs, self.dim_act)
            self.c[k, :] = npr.randn(self.dim_obs)

        for k in range(self.nb_states):
            self.K = npr.randn(self.nb_states, self.dim_act, self.dim_obs)
            self.kff = npr.randn(self.nb_states, self.dim_act)

        L = 0.1 * npr.randn(self.nb_states, self.dim_obs, self.dim_obs)
        self.cov_x = np.array([L[k, ...] @ L[k, ...].T for k in range(self.nb_states)])

        L = 0.1 * npr.randn(self.nb_states, self.dim_act, self.dim_act)
        self.cov_u = np.array([L[k, ...] @ L[k, ...].T for k in range(self.nb_states)])

    @property
    def cov_x(self):
        return self._cov_x

    @cov_x.setter
    def cov_x(self, mat):
        self._cov_x = mat + np.array([np.eye(self.dim_obs) * self.reg])

    @property
    def cov_u(self):
        return self._cov_u

    @cov_u.setter
    def cov_u(self, mat):
        self._cov_u = mat + np.array([np.eye(self.dim_act) * self.reg])

    def mean_u(self, z, x):
        return np.einsum('kh,...h->...k', self.K[z, ...], x) + self.kff[z, ...]

    def mean_x(self, z, x, u):
        return np.einsum('kh,...h->...k', self.A[z, ...], x) +\
               np.einsum('kh,...h->...k', self.B[z, ...], u) + self.c[z, :]

    def sample_u(self, z, x):
        mu_u = self.mean_u(z, x)
        return mvn(mean=mu_u, cov=self.cov_u[z, ...]).rvs()

    def sample_x(self, z, x, u):
        mu_x = self.mean_x(z, x, u)
        return mvn(mean=mu_x, cov=self.cov_x[z, ...]).rvs()

    def sample(self, z, x, stoch=True):
        if stoch:
            u = self.sample_u(z, x)
            mu_x = self.mean_x(z, x, u)
            return mvn(mean=mu_x, cov=self.cov_x[z, ...]).rvs(), u
        else:
            mu_u = self.mean_u(z, x)
            mu_x = self.mean_x(z, x, mu_u)
            return mu_x, mu_u

    def likelihood(self, x, u):
        lik = []
        for _x, _u in zip(x, u):
            T = _x.shape[0]
            _lik = np.zeros((T - 1, self.nb_states))
            for k in range(self.nb_states):
                mu_x = self.mean_x(k, _x[:-1, :], _u[:-1, :self.dim_act])
                mu_u = self.mean_u(k, _x[:-1, :])
                for t in range(T - 1):
                    _lik[t, k] = mvn(mean=mu_x[t, :], cov=self.cov_x[k, ...]).pdf(_x[t + 1, :]) *\
                                 mvn(mean=mu_u[t, :], cov=self.cov_u[k, ...]).pdf(_u[t, :])

            lik.append(_lik)

        return lik

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            T = _x.shape[0]
            _loglik = np.zeros((T - 1, self.nb_states))
            for k in range(self.nb_states):
                mu_x = self.mean_x(k, _x[:-1, :], _u[:-1, :self.dim_act])
                mu_u = self.mean_u(k, _x[:-1, :])
                _loglik[:, k] = multivariate_normal_logpdf(_x[1:, :], mu_x, self.cov_x[k, ...])
                _loglik[:, k] += multivariate_normal_logpdf(_u[:-1, :], mu_u, self.cov_u[k, ...])

            loglik.append(_loglik)

        return loglik

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self.cov_x = self.cov_x[perm, ...]

        self.K = self.K[perm, ...]
        self.cov_u = self.cov_u[perm, ...]

    def mstep(self, x, u, w):

        # fit dynamics
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, w):
            xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dim_act], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _J_diag = np.concatenate((self.reg * np.ones(self.dim_obs),
                                  self.reg * np.ones(self.dim_act),
                                  self.reg * np.ones(1)))
        _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
        _h = np.zeros((self.nb_states, self.dim_obs + self.dim_act + 1, self.dim_obs))

        # solving p = (xT w x)^-1 xT w y
        for _x, _y, _w in zip(xs, ys, ws):
            for k in range(self.nb_states):
                wx = _x * _w[:, k:k + 1]
                _J[k] += np.dot(wx.T, _x)
                _h[k] += np.dot(wx.T, _y)

        mu = np.linalg.solve(_J, _h)
        self.A = np.swapaxes(mu[:, :self.dim_obs, :], 1, 2)
        self.B = np.swapaxes(mu[:, self.nb_states:self.nb_states + self.dim_act, :], 1, 2)
        self.c = mu[:, -1, :]

        sqerr = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        weight = 1e-8 * np.ones(self.nb_states)
        for _x, _y, _w in zip(xs, ys, ws):
            yhat = np.matmul(_x[None, :, :], mu)
            resid = _y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
            weight += np.sum(_w, axis=0)

        self.cov_x = sqerr / weight[:, None, None]

        # fit actions
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, w):
            xs.append(np.hstack((_x, np.ones((_x.shape[0], 1)))))
            ys.append(_u)
            ws.append(_w)

        _J_diag = np.concatenate((self.reg * np.ones(self.dim_obs),
                                  self.reg * np.ones(1)))
        _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
        _h = np.zeros((self.nb_states, self.dim_obs + 1, self.dim_act))

        # solving p = (xT w x)^-1 xT w y
        for _x, _y, _w in zip(xs, ys, ws):
            for k in range(self.nb_states):
                wx = _x * _w[:, k:k + 1]
                _J[k] += np.dot(wx.T, _x)
                _h[k] += np.dot(wx.T, _y)

        mu = np.linalg.solve(_J, _h)
        self.K = np.swapaxes(mu[:, :self.dim_act, :], 1, 2)
        self.kff = mu[:, -1, :]

        sqerr = np.zeros((self.nb_states, self.dim_act, self.dim_act))
        weight = 1e-8 * np.ones(self.nb_states)
        for _x, _y, _w in zip(xs, ys, ws):
            yhat = np.matmul(_x[None, :, :], mu)
            resid = _y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
            weight += np.sum(_w, axis=0)

        self.cov_u = sqerr / weight[:, None, None]

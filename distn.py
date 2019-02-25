import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.misc import logsumexp

import scipy as sc
import scipy.stats as scs
from scipy import special

from scipy.stats import multinomial as cat
from scipy.stats import multivariate_normal as mvn

from sds.util import random_rotation
from sds.util import fit_linear_regression
from sds.util import bfgs

from sklearn.preprocessing import PolynomialFeatures


class CategoricalInitState:

    def __init__(self, nb_states, reg=1e-8):
        self.nb_states = nb_states
        self.reg = reg

        self.pi = np.ones((self.nb_states, )) / self.nb_states

    def sample(self):
        return np.argmax(cat(1, self.pi).rvs())

    def lik(self):
        return self.pi

    def loglik(self):
        return np.log(self.lik())

    def logprior(self):
        return 0.0

    def permute(self, perm):
        self.pi = self.pi[perm]

    def update(self, w):
        self.pi = w + self.reg
        self.pi /= np.sum(self.pi)


class StationaryTransition:

    def __init__(self, nb_states, reg=1e-16):
        self.nb_states = nb_states
        self.reg = reg

        self.mat =  0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
        self.mat /= np.sum(self.mat, axis=1, keepdims=True)

    def sample(self, z):
        return np.argmax(cat(1, self.mat[z, :]).rvs())

    def lik(self):
        return self.mat

    def loglik(self):
        return np.log(self.lik())

    def logprior(self):
        return 0.0

    def permute(self, perm):
        self.mat = self.mat[np.ix_(perm, perm)]

    def update(self, joint):
        counts = np.sum(joint, axis=0) + self.reg
        self.mat = counts / np.sum(counts, axis=1, keepdims=True)


class RecurrentTransition:

    def __init__(self, nb_states, dim_in, degree=1, reg=1e-16):
        self.nb_states = nb_states
        self.dim_in = dim_in
        self.degree = degree
        self.reg = reg

        mat =  0.95 * np.eye(self.nb_states) + 0.05 * npr.rand(self.nb_states, self.nb_states)
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

    def sample(self, z, x):
        mat = np.exp(self.loglik(x))[0, ...]
        return np.argmax(cat(1, mat[z, :]).rvs())

    def loglik(self, x):
        T, D = x.shape

        logtrans = np.tile(self.logmat[np.newaxis, :, :], (T - 1, 1, 1))
        logtrans += (self.basis.fit_transform(x[:-1, :]) @ self.par.T)[:, np.newaxis, :]

        return logtrans - logsumexp(logtrans, axis=-1, keepdims=True)

    def logprior(self):
        return 0.0

    def permute(self, perm):
        self.logmat = self.logmat[np.ix_(perm, perm)]
        self.par = self.par[perm, :]

    def update(self, joints, x, num_iters=100):

        def _expected_log_joint(joints):
            elbo = self.logprior()
            logtrans = self.loglik(x)
            elbo += np.sum(joints * logtrans)
            return elbo

        T = x.shape[0]
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(joints)
            return -obj / T

        self.params = bfgs(_objective, self.params, num_iters=num_iters)


class RecurrentOnlyTransition:

    def __init__(self, nb_states, dim_in, degree=1, reg=1e-16):
        self.nb_states = nb_states
        self.dim_in = dim_in
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

    def sample(self, z, x):
        mat = np.exp(self.loglik(x))[0, ...]
        return np.argmax(cat(1, mat[z, :]).rvs())

    def loglik(self, x):
        logtrans = (self.basis.fit_transform(x[:-1, :]) @ self.par.T)[:, np.newaxis, :]
        logtrans += self.offset
        logtrans = np.tile(logtrans, (1, self.nb_states, 1))

        return logtrans - logsumexp(logtrans, axis=-1, keepdims=True)

    def logprior(self):
        return 0.0

    def permute(self, perm):
        self.par = self.par[perm, :]
        self.offset = self.offset[perm]

    def update(self, joints, x, num_iters=100):

        def _expected_log_joint(joints):
            elbo = self.logprior()
            logtrans = self.loglik(x)
            elbo += np.sum(joints * logtrans)
            return elbo

        T = x.shape[0]
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(joints)
            return -obj / T

        self.params = bfgs(_objective, self.params, num_iters=num_iters)


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

    def lik(self, x):
        return np.array([mvn(mean=self.mu[k, :], cov=self.cov[k, ...]).pdf(x)
                         for k in range(self.nb_states)]).T

    def loglik(self, x):
        return np.array([mvn(mean=self.mu[k, :], cov=self.cov[k, ...]).logpdf(x)
                         for k in range(self.nb_states)]).T

    def logprior(self):
        return 0.0

    def permute(self, perm):
        self.mu = self.mu[perm, :]
        self.cov = self.cov[perm, ...]

    def update(self, x, w, biased=False):
        aux = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        for k in range(self.nb_states):
            self.mu[k, :] = np.sum(np.einsum('n,nk->nk', w[:, k], x), axis=0) / np.sum(w[:, k])

            if biased:
                Z = np.sum(w[:, k])
            else:
                Z = (np.square(np.sum(w[:, k])) - np.sum(np.square(w[:, k]))) / np.sum(w[:, k])

            diff = x - self.mu[k, :]
            aux[k, ...] = np.einsum('nk,n,nh->kh', diff, w[:, k], diff) / Z

        self.cov = aux


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, dim_obs, reg=1e-8):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.reg = reg

        self.A = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        self.c = np.zeros((self.nb_states, self.dim_obs))
        for k in range(self.nb_states):
            self.A[k, ...] = .95 * random_rotation(self.dim_obs)
            self.c[k, :] = npr.randn(self.dim_obs)

        L = 0.1 * npr.randn(self.nb_states, self.dim_obs, self.dim_obs)
        self.cov = np.array([L[k, ...] @ L[k, ...].T for k in range(self.nb_states)])

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, mat):
        self._cov = mat + np.array([np.eye(self.dim_obs) * self.reg])

    def mean(self, z, x):
        return np.einsum('kh,...h->...k', self.A[z, ...], x) + self.c[z, :]

    def sample(self, z, x):
        mu = self.mean(z, x)
        return mvn(mu, cov=self.cov[z, ...]).rvs()

    def lik(self, x):
        T = x.shape[0]
        lik = np.zeros((T - 1, self.nb_states))
        for k in range(self.nb_states):
            mu = self.mean(k, x[:-1, :])
            for t in range(T - 1):
                lik[t, k] = mvn(mean=mu[t, :], cov=self.cov[k, ...]).pdf(x[t + 1, :])
        return lik

    def loglik(self, x):
        T = x.shape[0]
        loglik = np.zeros((T - 1, self.nb_states))
        for k in range(self.nb_states):
            mu = self.mean(k, x[:-1, :])
            for t in range(T - 1):
                loglik[t, k] = mvn(mean=mu[t, :], cov=self.cov[k, ...]).logpdf(x[t + 1, :])
        return loglik

    def logprior(self):
        return 0.0

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.c = self.c[perm, :]
        self.cov = self.cov[perm, ...]

    def update(self, x, w, biased=False):
        T = x.shape[0]

        ones = np.ones((T, 1))
        feat = np.column_stack((ones, x))

        _in, _out = feat[:-1, :], x[1:, :]

        aux = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        for k in range(self.nb_states):
            coef_, _ = fit_linear_regression(_in, _out,
                                             weights=w[1:, k],
                                             fit_intercept=False)
            self.c[k, :] = coef_[:, 0]
            self.A[k, ...] = coef_[:, 1:]

            if biased:
                Z = np.sum(w[1:, k])
            else:
                Z = (np.square(np.sum(w[1:, k])) - np.sum(np.square(w[1:, k]))) / np.sum(w[1:, k])

            diff = x[1:, :] - self.mean(k, x[:-1, :])
            aux[k, ...] = np.einsum('nk,n,nh->kh', diff, w[1:, k], diff) / Z

        self.cov = aux

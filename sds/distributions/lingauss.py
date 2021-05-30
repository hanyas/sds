import numpy as np
import numpy.random as npr

import scipy as sc

from operator import add
from functools import reduce

from sds.utils.general import Statistics as Stats
from sds.utils.linalg import symmetrize


class LinearGaussianWithPrecision:

    def __init__(self, input_dim, output_dim,
                 A=None, lmbda=None, affine=True):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A = A
        self.affine = affine

        self._lmbda = lmbda
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.A, self.lmbda

    @params.setter
    def params(self, values):
        self.A, self.lmbda = values

    @property
    def nb_params(self):
        return self.dcol * self.drow \
               + self.drow * (self.drow + 1) / 2

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.input_dim - 1
        else:
            return self.input_dim

    @property
    def drow(self):
        return self.output_dim

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._lmbda = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = sc.linalg.cholesky(self.lmbda, lower=False)
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = sc.linalg.inv(self.lmbda_chol)
        return self._lmbda_chol_inv

    @property
    def sigma(self):
        return self.lmbda_chol_inv @ self.lmbda_chol_inv.T

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('dl,...l->...d', A, x, optimize=True) + b.T
        else:
            y = np.einsum('dl,...l->...d', self.A, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return self.mean(x) + npr.normal(size=self.drow).dot(self.lmbda_chol_inv.T)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow / 2.)

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            contract = 'nd,nl->dl'

            yxT = np.einsum(contract, y, x, optimize=True)
            xxT = np.einsum(contract, x, x, optimize=True)
            yyT = np.einsum(contract, y, y, optimize=True)
            n = y.shape[0]

            return Stats([yxT, xxT, yyT, n])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            contract = 'nd,n,nl->dl'

            yxT = np.einsum(contract, y, weights, x, optimize=True)
            xxT = np.einsum(contract, x, weights, x, optimize=True)
            yyT = np.einsum(contract, y, weights, y, optimize=True)
            n = np.sum(weights)

            return Stats([yxT, xxT, yyT, n])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        mu = self.predict(x)
        return 0.5 * np.einsum('nd,dl,nl->n', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

            mu = self.mean(x)
            loglik = np.einsum('nd,dl,nl->n', mu, self.lmbda, y, optimize=True) \
                     - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)

            loglik[bads] = 0.
            loglik += - self.log_partition(x) + self.log_base()
            return loglik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxT, xxT, yyT, n = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        self.A = np.linalg.solve(xxT, yxT.T).T
        _sigma = (yyT - self.A.dot(yxT.T)) / n

        # numerical stabilization
        _sigma = symmetrize(_sigma) + 1e-16 * np.eye(self.drow)
        assert np.allclose(_sigma, _sigma.T)
        assert np.all(np.linalg.eigvalsh(_sigma) > 0.)

        self.lmbda = np.linalg.inv(_sigma)


class StackedLinearGaussiansWithPrecision:
    def __init__(self, size, input_dim, output_dim,
                 As=None, lmbdas=None, affine=True):

        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.affine = affine

        As = [None] * self.size if As is None else As
        lmbdas = [None] * self.size if lmbdas is None else lmbdas
        self.dists = [LinearGaussianWithPrecision(input_dim, output_dim,
                                                  As[k], lmbdas[k], affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbdas

    @params.setter
    def params(self, values):
        self.As, self.lmbdas = values

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.input_dim - 1
        else:
            return self.input_dim

    @property
    def drow(self):
        # output dimension
        return self.output_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k, ...]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @lmbdas.setter
    def lmbdas(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda = value[k, ...]

    @property
    def lmbdas_chol(self):
        return np.array([dist.lmbda_chol for dist in self.dists])

    @property
    def lmbdas_chol_inv(self):
        return np.array([dist.lmbda_chol_inv for dist in self.dists])

    @property
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def predict(self, x):
        if self.affine:
            As, bs = self.As[:, :, :-1], self.As[:, :, -1]
            y = np.einsum('kdl,...l->...kd', As, x, optimize=True) + bs[None, ...]
        else:
            y = np.einsum('kdl,...l->...kd', self.As, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return np.array([dist.rvs(x) for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            contract = 'nd,nl->dl'

            yxT = np.einsum(contract, y, x, optimize=True)
            xxT = np.einsum(contract, x, x, optimize=True)
            yyT = np.einsum(contract, y, y, optimize=True)
            n = y.shape[0]

            yxTk = np.array([yxT for _ in range(self.size)])
            xxTk = np.array([xxT for _ in range(self.size)])
            yyTk = np.array([yyT for _ in range(self.size)])
            nk = np.array([n for _ in range(self.size)])

            return Stats([yxTk, xxTk, yyTk, nk])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            contract = 'nd,nk,nl->kdl'

            yxTk = np.einsum(contract, y, weights, x, optimize=True)
            xxTk = np.einsum(contract, x, weights, x, optimize=True)
            yyTk = np.einsum(contract, y, weights, y, optimize=True)
            nk = np.sum(weights, axis=0)

            return Stats([yxTk, xxTk, yyTk, nk])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        return np.array([dist.log_partition(x) for dist in self.dists]).T

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

            mu = self.mean(x)
            loglik = np.einsum('nkd,kdl,nl->nk', mu, self.lmbdas, y, optimize=True)\
                     - 0.5 * np.einsum('nd,kdl,nl->nk', y, self.lmbdas, y, optimize=True)

            loglik[bads] = 0.
            loglik += - self.log_partition(x) + self.log_base()
            return loglik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxTk, xxTk, yyTk, nk = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.output_dim, self.input_dim))
        lmbdas = np.zeros((self.size, self.output_dim, self.output_dim))
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma = (yyTk[k] - As[k].dot(yxTk[k].T)) / nk[k]

            # numerical stabilization
            _sigma = symmetrize(sigma) + 1e-16 * np.eye(self.drow)
            assert np.allclose(sigma, sigma.T)
            assert np.all(np.linalg.eigvalsh(sigma) > 0.)

            lmbdas[k] = np.linalg.inv(sigma)

        self.As = As
        self.lmbdas = lmbdas


class TiedLinearGaussiansWithPrecision:
    def __init__(self, size, input_dim, output_dim,
                 As=None, lmbda=None, affine=True):

        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.affine = affine

        self._lmbda = lmbda
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

        As = [None] * self.size if As is None else As
        self.dists = [LinearGaussianWithPrecision(input_dim, output_dim,
                                                  As[k], lmbda, affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbda

    @params.setter
    def params(self, values):
        self.As, self.lmbda = values

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.input_dim - 1
        else:
            return self.input_dim

    @property
    def drow(self):
        # output dimension
        return self.output_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k, ...]

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._lmbda = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

        for k, dist in enumerate(self.dists):
            dist.lmbda = value

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = sc.linalg.cholesky(self.lmbda, lower=False)
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = sc.linalg.inv(self.lmbda_chol)
        return self._lmbda_chol_inv

    @property
    def sigma(self):
        return self.lmbda_chol_inv @ self.lmbda_chol_inv.T

    def predict(self, x):
        if self.affine:
            As, bs = self.As[:, :, :-1], self.As[:, :, -1]
            y = np.einsum('kdl,...l->...kd', As, x, optimize=True) + bs[None, ...]
        else:
            y = np.einsum('kdl,...l->...kd', self.As, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return np.array([dist.rvs(x) for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            yxT = np.einsum('nd,nl->dl', y, x, optimize=True)
            xxT = np.einsum('nd,nl->dl', x, x, optimize=True)

            yxTk = np.array([yxT for _ in range(self.size)])
            xxTk = np.array([xxT for _ in range(self.size)])
            yyT = np.einsum('nd,nl->dl', y, y, optimize=True)
            n = y.shape[0]

            return Stats([yxTk, xxTk, yyT, n])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            yxTk = np.einsum('nd,nk,nl->kdl', y, weights, x, optimize=True)
            xxTk = np.einsum('nd,nk,nl->kdl', x, weights, x, optimize=True)
            yyT = np.einsum('nd,nl->dl', y, y, optimize=True)
            n = np.sum(weights)

            return Stats([yxTk, xxTk, yyT, n])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        return np.array([dist.log_partition(x) for dist in self.dists]).T

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

            mu = self.mean(x)
            loglik = np.einsum('nkd,dl,nl->nk', mu, self.lmbda, y, optimize=True)\
                     - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)[:, None]

            loglik[bads] = 0.
            loglik += - self.log_partition(x) + self.log_base()
            return loglik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxTk, xxTk, yyT, n = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        A = np.zeros((self.size, self.output_dim, self.input_dim))
        sigma = np.zeros((self.output_dim, self.output_dim))

        sigma = yyT
        for k in range(self.size):
            A[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma -= A[k].dot(yxTk[k].T)
        sigma /= n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.drow)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.As = A
        self.lmbda = np.linalg.inv(sigma)


class LinearGaussianWithDiagonalPrecision:

    def __init__(self, input_dim, output_dim,
                 A=None, lmbda_diag=None, affine=True):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A = A
        self.affine = affine

        self._lmbda_diag = lmbda_diag
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.A, self.lmbda_diag

    @params.setter
    def params(self, values):
        self.A, self.lmbda_diag = values

    @property
    def nb_params(self):
        return self.dcol * self.drow + self.drow

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.input_dim - 1
        else:
            return self.input_dim

    @property
    def drow(self):
        # output dimension
        return self.output_dim

    @property
    def lmbda_diag(self):
        return self._lmbda_diag

    @lmbda_diag.setter
    def lmbda_diag(self, value):
        self._lmbda_diag = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def lmbda(self):
        assert self._lmbda_diag is not None
        return np.diag(self._lmbda_diag)

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = np.diag(np.sqrt(self.lmbda_diag))
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = np.diag(1. / np.sqrt(self.lmbda_diag))
        return self._lmbda_chol_inv

    @property
    def sigma_diag(self):
        return 1. / self.lmbda_diag

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('dl,...l->...d', A, x, optimize=True) + b.T
        else:
            y = np.einsum('dl,...l->...d', self.A, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return self.mean(x) + npr.normal(size=self.drow).dot(self.lmbda_chol_inv.T)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow / 2.)

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = np.einsum('nd,nl->dl', x, x, optimize=True)
            yxT = np.einsum('nd,nl->dl', y, x, optimize=True)
            yy = np.einsum('nd,nd->d', y, y, optimize=True)
            nd = np.broadcast_to(y.shape[0], (self.output_dim, ))

            return Stats([yxT, xxT, nd, yy])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = np.einsum('nd,n,nl->dl', x, weights, x, optimize=True)
            yxT = np.einsum('nd,n,nl->dl', y, weights, x, optimize=True)
            yy = np.einsum('nd,n,nd->d', y, weights, y, optimize=True)
            nd = np.broadcast_to(np.sum(weights), (self.output_dim, ))

            return Stats([yxT, xxT, nd, yy])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        mu = self.predict(x)
        return 0.5 * np.einsum('nd,dl,nl->n', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

            mu = self.mean(x)
            loglik = np.einsum('nd,dl,nl->n', mu, self.lmbda, y, optimize=True) \
                     - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)

            loglik[bads] = 0.
            loglik += - self.log_partition(x) + self.log_base()
            return loglik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxT, xxT, nd, yy = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        self.A = np.linalg.solve(xxT, yxT.T).T
        sigmas = (yy - np.einsum('dl,dl->d', self.A, yxT)) / nd
        self.lmbda_diag = 1. / sigmas


class StackedLinearGaussiansWithDiagonalPrecision:
    def __init__(self, size, input_dim, output_dim,
                 As=None, lmbdas_diag=None, affine=True):

        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.affine = affine

        As = [None] * self.size if As is None else As
        lmbdas_diag = [None] * self.size if lmbdas_diag is None else lmbdas_diag
        self.dists = [LinearGaussianWithDiagonalPrecision(input_dim, output_dim,
                                                          As[k], lmbdas_diag[k],
                                                          affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbdas_diag

    @params.setter
    def params(self, values):
        self.As, self.lmbdas_diag = values

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.input_dim - 1
        else:
            return self.input_dim

    @property
    def drow(self):
        # output dimension
        return self.output_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k, ...]

    @property
    def lmbdas_diag(self):
        return np.array([dist.lmbda_diag for dist in self.dists])

    @lmbdas_diag.setter
    def lmbdas_diag(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda_diag = value[k, ...]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @property
    def lmbdas_chol(self):
        return np.array([dist.lmbda_chol for dist in self.dists])

    @property
    def lmbdas_chol_inv(self):
        return np.array([dist.lmbda_chol_inv for dist in self.dists])

    @property
    def sigmas_diag(self):
        return np.array([dist.sigma_diag for dist in self.dists])

    def predict(self, x):
        if self.affine:
            As, bs = self.As[:, :, :-1], self.As[:, :, -1]
            y = np.einsum('kdl,...l->...kd', As, x, optimize=True) + bs[None, ...]
        else:
            y = np.einsum('kdl,...l->...kd', self.As, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return np.array([dist.rvs(x) for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = np.einsum('nd,nl->dl', x, x, optimize=True)
            yxT = np.einsum('nd,nl->dl', y, x, optimize=True)
            yy = np.einsum('nd,nd->d', y, y, optimize=True)
            nd = np.broadcast_to(y.shape[0], (self.output_dim, ))

            xxTk = np.array([xxT for _ in range(self.size)])
            yxTk = np.array([yxT for _ in range(self.size)])
            yyk = np.array([yy for _ in range(self.size)])
            ndk = np.array([nd for _ in range(self.size)])

            return Stats([yxTk, xxTk, ndk, yyk])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxTk = np.einsum('nd,nk,nl->kdl', x, weights, x, optimize=True)
            yxTk = np.einsum('nd,nk,nl->kdl', y, weights, x, optimize=True)
            yyk = np.einsum('nd,nk,nd->kd', y, weights, y, optimize=True)
            ndk = np.broadcast_to(np.sum(weights, axis=0)[:, None],
                                  (self.size, self.output_dim))

            return Stats([yxTk, xxTk, ndk, yyk])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        return np.array([dist.log_partition(x) for dist in self.dists]).T

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

            mu = self.mean(x)
            loglik = np.einsum('nkd,kdl,nl->nk', mu, self.lmbdas, y, optimize=True)\
                     - 0.5 * np.einsum('nd,kdl,nl->nk', y, self.lmbdas, y, optimize=True)

            loglik[bads] = 0.
            loglik += - self.log_partition(x) + self.log_base()
            return loglik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxTk, xxTk, ndk, yyk = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.output_dim, self.input_dim))
        lmbdas = np.zeros((self.size, self.output_dim))
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigmas = (yyk[k] - np.einsum('dl,dl->d', As[k], yxTk[k])) / ndk[k]
            lmbdas[k] = 1. / sigmas

        self.As = As
        self.lmbdas_diag = lmbdas


class TiedLinearGaussiansWithDiagonalPrecision:
    def __init__(self, size, input_dim, output_dim,
                 As=None, lmbda_diag=None, affine=True):

        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.affine = affine

        self._lmbda_diag = lmbda_diag
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

        As = [None] * self.size if As is None else As
        self.dists = [LinearGaussianWithDiagonalPrecision(input_dim, output_dim,
                                                          As[k], lmbda_diag,
                                                          affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbda_diag

    @params.setter
    def params(self, values):
        self.As, self.lmbda_diag = values

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.input_dim - 1
        else:
            return self.input_dim

    @property
    def drow(self):
        # output dimension
        return self.output_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k, ...]

    @property
    def lmbda_diag(self):
        return self._lmbda_diag

    @lmbda_diag.setter
    def lmbda_diag(self, value):
        self._lmbda_diag = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

        for k, dist in enumerate(self.dists):
            dist.lmbda_diag = value

    @property
    def lmbda(self):
        assert self._lmbda_diag is not None
        return np.diag(self._lmbda_diag)

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = np.diag(np.sqrt(self.lmbda_diag))
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = np.diag(1. / np.sqrt(self.lmbda_diag))
        return self._lmbda_chol_inv

    @property
    def sigma_diag(self):
        return 1. / self.lmbda_diag

    def predict(self, x):
        if self.affine:
            As, bs = self.As[:, :, :-1], self.As[:, :, -1]
            y = np.einsum('kdl,...l->...kd', As, x, optimize=True) + bs[None, ...]
        else:
            y = np.einsum('kdl,...l->...kd', self.As, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return np.array([dist.rvs(x) for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = np.einsum('nd,nl->dl', x, x, optimize=True)
            yxT = np.einsum('nd,nl->dl', y, x, optimize=True)

            xxTk = np.array([xxT for _ in range(self.size)])
            yxTk = np.array([yxT for _ in range(self.size)])
            yy = np.einsum('nd,nd->d', y, y, optimize=True)
            nd = np.broadcast_to(y.shape[0], (self.output_dim, ))

            return Stats([yxTk, xxTk, nd, yy])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxTk = np.einsum('nd,nk,nl->kdl', x, weights, x, optimize=True)
            yxTk = np.einsum('nd,nk,nl->kdl', y, weights, x, optimize=True)
            yy = np.einsum('nd,nd->d', y, y, optimize=True)
            nd = np.broadcast_to(np.sum(weights), (self.output_dim, ))

            return Stats([yxTk, xxTk, nd, yy])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        return np.array([dist.log_partition(x) for dist in self.dists]).T

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

            mu = self.mean(x)
            loglik = np.einsum('nkd,dl,nl->nk', mu, self.lmbda, y, optimize=True)\
                     - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)[:, None]

            loglik[bads] = 0.
            loglik += - self.log_partition(x) + self.log_base()
            return loglik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxTk, xxTk, nd, yy = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.output_dim, self.input_dim))
        sigma_diag = np.zeros((self.output_dim, ))

        sigma_diag = yy
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma_diag -= np.einsum('dl,dl->d', As[k], yxTk[k])
        sigma_diag /= nd

        self.As = As
        self.lmbda_diag = 1. / sigma_diag


class _IndependentLinearGaussianBase:

    def __init__(self, input_dim, A=None, lmbda=None, affine=True):

        self.input_dim = input_dim

        self.A = A
        self.lmbda = lmbda
        self.affine = affine

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def dcol(self):
        if self.affine:
            return self.input_dim + 1
        else:
            return self.input_dim

    @property
    def sigma(self):
        return 1. / self.lmbda

    def predict(self, x):
        if self.affine:
            A, b = self.A[:-1], self.A[-1]
            return x @ A + b
        else:
            return x @ self.A

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return self.mean(x) + npr.randn() * np.sqrt(self.sigma)

    def statistics(self, x, y):
        raise NotImplementedError

    def weighted_statistics(self, x, y, weights):
        raise NotImplementedError


class IndependentLinearGaussianWithKnownPrecision(_IndependentLinearGaussianBase):

    def __init__(self, input_dim, A=None, lmbda=None, affine=True):
        super(IndependentLinearGaussianWithKnownPrecision, self).__init__(input_dim, A,
                                                                          lmbda, affine)

    @property
    def params(self):
        return self.A

    @params.setter
    def params(self, values):
        self.A = values

    @property
    def nb_params(self):
        return self.dcol

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=0))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = - 0.5 * self.lmbda * np.einsum('nd,nl->dl', x, x, optimize=True)
            yxT = self.lmbda * np.einsum('n,nl->l', y, x, optimize=True)

            return Stats([yxT, xxT])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=0))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = - 0.5 * np.einsum('nd,n,nl->dl', x, weights, x, optimize=True)
            yxT = np.einsum('n,n,nl->l', y, weights, x, optimize=True)

            return Stats([yxT, xxT])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)


class IndependentLinearGaussianWithKnownMean(_IndependentLinearGaussianBase):

    def __init__(self, input_dim, A=None, lmbda=None, affine=True):
        super(IndependentLinearGaussianWithKnownMean, self).__init__(input_dim, A,
                                                                     lmbda, affine)

    @property
    def params(self):
        return self.lmbda

    @params.setter
    def params(self, values):
        self.lmbda = values

    @property
    def nb_params(self):
        return 1

    def statistics(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=0))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            n = 0.5 * y.shape[0]
            yy = - 0.5 * np.sum(y * y)
            yxTa = np.einsum('n,nl,l->', y, x, self.A, optimize=True)
            xTaaTx = - 0.5 * np.einsum('l,nl,nd,d->', self.A, x, x, self.A, optimize=True)

            return Stats([n, yy + yxTa + xTaaTx])
        else:
            stats = list(map(self.statistics, x, y))
            return reduce(add, stats)

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=0))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            n = 0.5 * np.sum(weights)
            yy = - 0.5 * np.sum(weights * y * y)
            yxTa = np.einsum('n,n,nl,l->', y, weights, x, self.A, optimize=True)
            xTaaTx = - 0.5 * np.einsum('l,nl,n,nd,d->', self.A, x, weights, x, self.A, optimize=True)

            return Stats([n, yy + yxTa + xTaaTx])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

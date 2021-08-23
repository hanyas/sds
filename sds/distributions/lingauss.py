import numpy as np
import numpy.random as npr

import scipy as sc

from operator import add
from functools import reduce

from sds.utils.general import Statistics as Stats
from sds.utils.linalg import symmetrize


class LinearGaussianWithPrecision:

    def __init__(self, column_dim, row_dim,
                 A=None, lmbda=None, affine=True):

        self.column_dim = column_dim
        self.row_dim = row_dim

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
        return self.column_dim * self.row_dim \
               + self.row_dim * (self.row_dim + 1) / 2

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

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
        return self.mean(x) + npr.normal(size=self.output_dim).dot(self.lmbda_chol_inv.T)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.output_dim / 2.)

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

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('nd,dl,nl->n', mu, self.lmbda, y, optimize=True)\
                      - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition(x) + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxT, xxT, yyT, n = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        self.A = np.linalg.solve(xxT, yxT.T).T
        sigma = (yyT - self.A.dot(yxT.T)) / n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.lmbda = np.linalg.inv(sigma)


class StackedLinearGaussiansWithPrecision:

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas=None, affine=True):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        self.affine = affine

        As = [None] * self.size if As is None else As
        lmbdas = [None] * self.size if lmbdas is None else lmbdas
        self.dists = [LinearGaussianWithPrecision(column_dim, row_dim,
                                                  As[k], lmbdas[k], affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbdas

    @params.setter
    def params(self, values):
        self.As, self.lmbdas = values

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @lmbdas.setter
    def lmbdas(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda = value[k]

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

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('nkd,kdl,nl->nk', mu, self.lmbdas, y, optimize=True)\
                      - 0.5 * np.einsum('nd,kdl,nl->nk', y, self.lmbdas, y, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition(x) + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, yyTk, nk = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        lmbdas = np.zeros((self.size, self.output_dim, self.output_dim))
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma = (yyTk[k] - As[k].dot(yxTk[k].T)) / nk[k]

            # numerical stabilization
            sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
            assert np.allclose(sigma, sigma.T)
            assert np.all(np.linalg.eigvalsh(sigma) > 0.)

            lmbdas[k] = np.linalg.inv(sigma)

        self.As = As
        self.lmbdas = lmbdas


class TiedLinearGaussiansWithPrecision(StackedLinearGaussiansWithPrecision):

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas=None, affine=True):

        super(TiedLinearGaussiansWithPrecision, self).__init__(size, column_dim, row_dim,
                                                               As, lmbdas, affine)

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, yyT, n = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        sigma = np.zeros((self.output_dim, self.output_dim))

        sigma = yyT
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma -= As[k].dot(yxTk[k].T)
        sigma /= n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.As = As
        lmbda = np.linalg.inv(sigma)
        self.lmbdas = np.array(self.size * [lmbda])


class LinearGaussianWithDiagonalPrecision:

    def __init__(self, column_dim, row_dim,
                 A=None, lmbda_diag=None, affine=True):

        self.column_dim = column_dim
        self.row_dim = row_dim

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
        return self.column_dim * self.row_dim + self.row_dim

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

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

    @property
    def sigma(self):
        return np.diag(self.sigma_diag)

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
        return self.mean(x) + npr.normal(size=self.output_dim).dot(self.lmbda_chol_inv.T)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.output_dim / 2.)

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

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('nd,dl,nl->n', mu, self.lmbda, y, optimize=True) \
                      - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition(x) + self.log_base()
            return log_lik
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

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas_diags=None, affine=True):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        self.affine = affine

        As = [None] * self.size if As is None else As
        lmbdas_diags = [None] * self.size if lmbdas_diags is None else lmbdas_diags
        self.dists = [LinearGaussianWithDiagonalPrecision(column_dim, row_dim,
                                                          As[k], lmbdas_diags[k],
                                                          affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbdas_diags

    @params.setter
    def params(self, values):
        self.As, self.lmbdas_diags = values

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k]

    @property
    def lmbdas_diags(self):
        return np.array([dist.lmbda_diag for dist in self.dists])

    @lmbdas_diags.setter
    def lmbdas_diags(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda_diag = value[k]

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
            ndk = np.broadcast_to(np.sum(weights, axis=0, keepdims=True),
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

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('nkd,kdl,nl->nk', mu, self.lmbdas, y, optimize=True)\
                      - 0.5 * np.einsum('nd,kdl,nl->nk', y, self.lmbdas, y, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition(x) + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, ndk, yyk = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        lmbdas = np.zeros((self.size, self.output_dim))
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigmas = (yyk[k] - np.einsum('dl,dl->d', As[k], yxTk[k])) / ndk[k]
            lmbdas[k] = 1. / sigmas

        self.As = As
        self.lmbdas_diags = lmbdas


class TiedLinearGaussiansWithDiagonalPrecision(StackedLinearGaussiansWithDiagonalPrecision):

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas_diags=None, affine=True):

        super(TiedLinearGaussiansWithDiagonalPrecision, self).__init__(size, column_dim, row_dim,
                                                                       As, lmbdas_diags, affine)

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, nd, yy = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        sigma_diag = np.zeros((self.output_dim, ))

        sigma_diag = yy
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma_diag -= np.einsum('dl,dl->d', As[k], yxTk[k])
        sigma_diag /= nd

        self.As = As
        lmbda_diag = 1. / sigma_diag
        self.lmbdas_diags = np.array(self.size * [lmbda_diag])


class _SingleOutputLinearGaussianBase:

    def __init__(self, column_dim, W=None, lmbda=None, affine=True):

        self.column_dim = column_dim

        self.W = W
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
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def sigma(self):
        return 1. / self.lmbda

    def predict(self, x):
        if self.affine:
            W, c = self.W[:-1], self.W[-1]
            return x @ W + c
        else:
            return x @ self.W

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


class SingleOutputLinearGaussianWithKnownPrecision(_SingleOutputLinearGaussianBase):

    def __init__(self, column_dim, W=None, lmbda=None, affine=True):
        super(SingleOutputLinearGaussianWithKnownPrecision, self).__init__(column_dim, W,
                                                                           lmbda, affine)

    @property
    def params(self):
        return self.W

    @params.setter
    def params(self, values):
        self.W = values

    @property
    def nb_params(self):
        return self.column_dim

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


class SingleOutputLinearGaussianWithKnownMean(_SingleOutputLinearGaussianBase):

    def __init__(self, column_dim, W=None, lmbda=None, affine=True):
        super(SingleOutputLinearGaussianWithKnownMean, self).__init__(column_dim, W,
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
            yxTa = np.einsum('n,nl,l->', y, x, self.W, optimize=True)
            xTaaTx = - 0.5 * np.einsum('l,nl,nd,d->', self.W, x, x, self.W, optimize=True)

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
            yxTa = np.einsum('n,n,nl,l->', y, weights, x, self.W, optimize=True)
            xTaaTx = - 0.5 * np.einsum('l,nl,n,nd,d->', self.W, x, weights, x, self.W, optimize=True)

            return Stats([n, yy + yxTa + xTaaTx])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

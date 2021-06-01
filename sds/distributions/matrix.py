import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import linalg


class MatrixNormalWithPrecision:

    def __init__(self, column_dim, row_dim,
                 M=None, V=None, K=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.M = M
        self._V = V
        self._K = K

        self._V_chol = None
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.M, self.V, self.K

    @params.setter
    def params(self, values):
        self.M, self.V, self.K = values

    @property
    def nb_params(self):
        num = self.dcol * self.drow
        return num + num * (num + 1) / 2

    @property
    def dcol(self):
        return self.column_dim

    @property
    def drow(self):
        return self.row_dim

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        self._V = value
        self._V_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def V_chol(self):
        if self._V_chol is None:
            self._V_chol = sc.linalg.cholesky(self.V, lower=False)
        return self._V_chol

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def K_chol(self):
        if self._K_chol is None:
            self._K_chol = sc.linalg.cholesky(self.K, lower=False)
        return self._K_chol

    @property
    def lmbda(self):
        return np.kron(self.K, self.V)

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

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def rvs(self):
        aux = npr.normal(size=self.drow * self.dcol).dot(self.lmbda_chol_inv.T)
        return self.M + np.reshape(aux, (self.drow, self.dcol), order='F')

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow * self.dcol / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')
        return 0.5 * np.einsum('d,dl,l->', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (-1, self.drow * self.dcol), order='F')
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')

        # Gaussian likelihood on vector dist.
        bads = np.isnan(np.atleast_2d(xr)).any(axis=1)
        xr = np.nan_to_num(xr, copy=False).reshape((-1, self.drow * self.dcol))

        log_lik = np.einsum('d,dl,nl->n', mu, self.lmbda, xr)\
                  - 0.5 * np.einsum('nd,dl,nl->n', xr, self.lmbda, xr)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik


class StackedMatrixNormalWithPrecision:

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Vs=None, Ks=None):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        Ms = [None] * self.size if Ms is None else Ms
        Vs = [None] * self.size if Vs is None else Vs
        Ks = [None] * self.size if Ks is None else Ks

        self.dists = [MatrixNormalWithPrecision(column_dim, row_dim,
                                                Ms[k], Vs[k], Ks[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.Ms, self.Vs, self.Ks

    @params.setter
    def params(self, values):
        self.Ms, self.Vs, self.Ks = values

    @property
    def nb_params(self):
        num = self.dcol * self.drow
        return self.size * (num + num * (num + 1) / 2)

    @property
    def dcol(self):
        return self.column_dim

    @property
    def drow(self):
        return self.row_dim

    @property
    def Ms(self):
        return np.array([dist.M for dist in self.dists])

    @Ms.setter
    def Ms(self, value):
        for k, dist in enumerate(self.dists):
            dist.M = value[k, ...]

    @property
    def Vs(self):
        return np.array([dist.V for dist in self.dists])

    @Vs.setter
    def Vs(self, value):
        for k, dist in enumerate(self.dists):
            dist.V = value[k, ...]

    @property
    def Vs_chol(self):
        return np.array([dist.V_chol for dist in self.dists])

    @property
    def Ks(self):
        return np.array([dist.K for dist in self.dists])

    @Ks.setter
    def Ks(self, value):
        for k, dist in enumerate(self.dists):
            dist.K = value[k, ...]

    @property
    def Ks_chol(self):
        return np.array([dist.K_chol for dist in self.dists])

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
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def mean(self):
        return np.array([dist.mean() for dist in self.dists])

    def mode(self):
        return np.array([dist.mode() for dist in self.dists])

    def rvs(self):
        return np.array([dist.rvs() for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        raise NotImplementedError


class MatrixNormalWithDiagonalPrecision:

    def __init__(self, column_dim, row_dim,
                 M=None, V_diag=None, K=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.M = M

        self._V_diag = V_diag
        self._K = K

        self._V_chol = None
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.M, self.V_diag, self.K

    @params.setter
    def params(self, values):
        self.M, self.V_diag, self.K = values

    @property
    def nb_params(self):
        return self.dcol * self.drow\
               + self.dcol * self.drow

    @property
    def dcol(self):
        return self.column_dim

    @property
    def drow(self):
        return self.row_dim

    @property
    def V_diag(self):
        return self._V_diag

    @V_diag.setter
    def V_diag(self, value):
        self._V_diag = value
        self._V_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def V(self):
        return np.diag(self.V_diag)

    @property
    def V_chol(self):
        if self._V_chol is None:
            self._V_chol = sc.linalg.cholesky(self.V, lower=False)
        return self._V_chol

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def K_chol(self):
        if self._K_chol is None:
            self._K_chol = sc.linalg.cholesky(self.K, lower=False)
        return self._K_chol

    @property
    def lmbda(self):
        return np.kron(self.K, self.V)

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

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def rvs(self):
        aux = npr.normal(size=self.drow * self.dcol).dot(self.lmbda_chol_inv.T)
        return self.M + np.reshape(aux, (self.drow, self.dcol), order='F')

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow * self.dcol / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')
        return 0.5 * np.einsum('d,dl,l->', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (-1, self.drow * self.dcol), order='F')
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')

        # Gaussian likelihood on vector dist.
        bads = np.isnan(np.atleast_2d(xr)).any(axis=1)
        xr = np.nan_to_num(xr, copy=False).reshape((-1, self.drow * self.dcol))

        log_lik = np.einsum('d,dl,nl->n', mu, self.lmbda, xr)\
                  - 0.5 * np.einsum('nd,dl,nl->n', xr, self.lmbda, xr)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik


class StackedMatrixNormalWithDiagonalPrecision:

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Vs_diag=None, Ks=None):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        Ms = [None] * self.size if Ms is None else Ms
        Vs_diag = [None] * self.size if Vs_diag is None else Vs_diag
        Ks = [None] * self.size if Ks is None else Ks

        self.dists = [MatrixNormalWithDiagonalPrecision(column_dim, row_dim,
                                                        Ms[k], Vs_diag[k], Ks[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.Ms, self.Vs_diag, self.Ks

    @params.setter
    def params(self, values):
        self.Ms, self.Vs_diag, self.Ks = values

    @property
    def nb_params(self):
        return self.size * (self.dcol * self.drow
                            + self.dcol * self.drow)

    @property
    def dcol(self):
        return self.column_dim

    @property
    def drow(self):
        return self.row_dim

    @property
    def Ms(self):
        return np.array([dist.M for dist in self.dists])

    @Ms.setter
    def Ms(self, value):
        for k, dist in enumerate(self.dists):
            dist.M = value[k, ...]

    @property
    def Vs_diag(self):
        return np.array([dist.V_diag for dist in self.dists])

    @Vs_diag.setter
    def Vs_diag(self, value):
        for k, dist in enumerate(self.dists):
            dist.V_diag = value[k, ...]

    @property
    def Vs(self):
        return np.array([dist.V for dist in self.dists])

    @property
    def Vs_chol(self):
        return np.array([dist.V_chol for dist in self.dists])

    @property
    def Ks(self):
        return np.array([dist.K for dist in self.dists])

    @Ks.setter
    def Ks(self, value):
        for k, dist in enumerate(self.dists):
            dist.K = value[k, ...]

    @property
    def Ks_chol(self):
        return np.array([dist.K_chol for dist in self.dists])

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
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def mean(self):
        return np.array([dist.mean() for dist in self.dists])

    def mode(self):
        return np.array([dist.mode() for dist in self.dists])

    def rvs(self):
        return np.array([dist.rvs() for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        raise NotImplementedError

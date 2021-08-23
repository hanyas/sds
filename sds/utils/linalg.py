import numpy as np

from scipy.linalg import lapack as lapack


def copy_lower_to_upper(A):
    A += np.tril(A, k=-1).T


def invpd(A, return_chol=False):
    L = np.linalg.cholesky(A)
    Ainv = lapack.dpotri(L, lower=True)[0]
    copy_lower_to_upper(Ainv)
    if return_chol:
        return Ainv, L
    else:
        return Ainv


def blockarray(*args, **kwargs):
    return np.array(np.bmat(*args, **kwargs), copy=False)


def symmetrize(A):
    return (A + A.T) / 2.

import numpy as np
import numpy.random as npr

from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment

from operator import add, sub
from functools import lru_cache
from functools import wraps
from functools import reduce

from itertools import tee


def train_validate_split(obs, act, nb_traj_splits=7, seed=0,
                         split_trajs=False, begin=50, horizon=150, nb_time_splits=3):

    from sklearn.model_selection import KFold

    train_obs, train_act, valid_obs, valid_act = [], [], [], []
    list_idx = np.linspace(0, len(obs) - 1, len(obs), dtype=int)

    kf = KFold(nb_traj_splits, shuffle=True, random_state=seed)
    for train_list_idx, valid_list_idx in kf.split(list_idx):
        _train_obs = [obs[i] for i in train_list_idx]
        _train_act = [act[i] for i in train_list_idx]

        if split_trajs:
            _train_obs_splits, _train_act_splits = [], []
            for _obs, _act in zip(_train_obs, _train_act):
                length = _obs.shape[0]
                points = np.linspace(0, length - horizon, nb_time_splits + 1, dtype=int)[1:]
                for t in points:
                    _train_obs_splits.append(_obs[t: t + horizon])
                    _train_act_splits.append(_act[t: t + horizon])

            _train_obs += _train_obs_splits
            _train_act += _train_act_splits

        train_obs.append(_train_obs)
        train_act.append(_train_act)
        valid_obs.append([obs[i] for i in valid_list_idx])
        valid_act.append([act[i] for i in valid_list_idx])

    return train_obs, train_act, valid_obs, valid_act


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array, *args):
        array = np.array(hashable_array)
        return function(array, *args)

    @wraps(function)
    def wrapper(array, *args):
        array_tuple = tuple(zip(*array.T.tolist()))
        return cached_wrapper(array_tuple, *args)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def groupwise(x, n=2):
    t = tee(x, n)

    for i in range(1, n):
        for j in range(0, i):
            next(t[i], None)

    xl = list(zip(*t))
    if len(xl) == 0:
        xr = np.zeros((0, n * x.shape[-1]))
        return xr
    else:
        xr = np.stack(xl)
        return np.reshape(xr, (len(xr), -1))


def flatten_to_dim(X, d):
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def find_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = find_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * npr.rand()

    if n == 1:
        return npr.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(npr.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def linear_regression(Xs, ys, weights=None,
                      mu0=0., sigma0=1e32,
                      nu0=0, psi0=1.,
                      fit_intercept=True):

    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = ys if isinstance(ys, (list, tuple)) else [ys]
    assert len(Xs) == len(ys)

    D = Xs[0].shape[1]
    P = ys[0].shape[1]
    assert all([X.shape[1] == D for X in Xs])
    assert all([y.shape[1] == P for y in ys])
    assert all([X.shape[0] == y.shape[0] for X, y in zip(Xs, ys)])

    nu0 = np.maximum(nu0, P + 1)

    mu0 = mu0 * np.ones((P, D))
    sigma0 = sigma0 * np.eye(D)

    # Make sure the weights are the weights
    if weights is not None:
        weights = weights if isinstance(weights, (list, tuple)) else [weights]
    else:
        weights = [np.ones(X.shape[0]) for X in Xs]

    # Add weak prior on intercept
    if fit_intercept:
        mu0 = np.column_stack((mu0, np.zeros(P)))
        sigma0 = block_diag(sigma0, np.eye(1))

    # Compute the posterior
    J = np.linalg.inv(sigma0)
    h = np.dot(J, mu0.T)

    for X, y, weight in zip(Xs, ys, weights):
        X = np.column_stack((X, np.ones(X.shape[0]))) if fit_intercept else X
        J += np.dot(X.T * weight, X)
        h += np.dot(X.T * weight, y)

    # Solve for the MAP estimate
    # W = np.linalg.solve(J, h).T  # method 1
    # W = np.dot(h.T, np.linalg.pinv(J))  # method 2
    W = np.linalg.lstsq(J, h, rcond=None)[0].T  # method 3

    if fit_intercept:
        W, b = W[:, :-1], W[:, -1]
    else:
        b = 0

    # Compute the residual and the posterior variance
    nu = nu0
    Psi = psi0 * np.eye(P)
    for X, y, weight in zip(Xs, ys, weights):
        yhat = np.dot(X, W.T) + b
        resid = y - yhat
        nu += np.sum(weight)
        Psi += np.sum(weight[:, None, None] * resid[:, :, None] * resid[:, None, :], axis=0)
        # Psi += np.einsum('t,ti,tj->ij', weight, resid, resid)

    # Get MAP estimate of posterior covariance
    Sigma = Psi / (nu + P + 1)
    if fit_intercept:
        return W, b, Sigma
    else:
        return W, Sigma


def islist(*args):
    return all(isinstance(_arg, list) for _arg in args)


class Statistics(tuple):

    def __new__(cls, x):
        return tuple.__new__(Statistics, x)

    def __add__(self, y):
        gsum = lambda x, y: reduce(lambda a, b: list(map(add, a, b)) if islist(x, y) else a + b, [x, y])
        return Statistics(tuple(map(gsum, self, y)))

    def __sub__(self, y):
        gsub = lambda x, y: reduce(lambda a, b: list(map(sub, a, b)) if islist(x, y) else a - b, [x, y])
        return Statistics(tuple(map(gsub, self, y)))

    def __mul__(self, a):
        return Statistics(a * e for e in self)

    def __rmul__(self, a):
        return Statistics(a * e for e in self)

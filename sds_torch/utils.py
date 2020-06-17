import numpy as np
import numpy.random as npr

from scipy.linalg import block_diag

from scipy.optimize import linear_sum_assignment

from functools import lru_cache
from functools import wraps

import torch


def brownian(x0, n, dt, delta, out=None):

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = npr.randn(x0.shape[0], n) * delta * np.sqrt(dt)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return np.reshape(out, x0.shape[0])


def sample_env(env, nb_rollouts, nb_steps,
               ctl=None, noise_std=0.1,
               apply_limit=True):
    obs, act = [], []

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    ulim = env.action_space.high

    for n in range(nb_rollouts):
        _obs = np.zeros((nb_steps, dm_obs))
        _act = np.zeros((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                # unifrom distribution
                u = np.random.uniform(-ulim, ulim)
            else:
                u = ctl(x)
                u = u + noise_std * npr.randn(1, )

            if apply_limit:
                u = np.clip(u, -ulim, ulim)

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act


# list of dicts to dict of lists
def lod2dol(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]
    return d


def ensure_args_are_viable_lists(f):
    def wrapper(self, obs, act=None, **kwargs):
        assert obs is not None
        obs = [torch.from_numpy(np.atleast_2d(obs.numpy()))] if not isinstance(obs, (list, tuple)) else obs

        if act is None:
            act = []
            for _obs in obs:
                act.append(torch.zeros((_obs.shape[0], self.dm_act), dtype=torch.float64))

        act = [torch.from_numpy(np.atleast_2d(act.numpy()))] if not isinstance(act, (list, tuple)) else act

        return f(self, obs, act, **kwargs)
    return wrapper


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


# stack ar observations and controls
@np_cache
def stack(x, shift):
    _hr = len(x) - shift
    _x = np.vstack([np.hstack([x[t + l] for l in range(shift + 1)])
                    for t in range(_hr)])
    return np.squeeze(_x)


def flatten_to_dim(X, d):
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def state_overlap(z1, z2, K1=None, K2=None):
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


def permutation(z1, z2, K1=None, K2=None):
    overlap = state_overlap(z1, z2, K1=K1, K2=K2)
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
                      nu0=0, psi0=1e-32,
                      fit_intercept=True):

    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = ys if isinstance(ys, (list, tuple)) else [ys]
    assert len(Xs) == len(ys)

    D = Xs[0].shape[1]
    P = ys[0].shape[1]
    assert all([X.shape[1] == D for X in Xs])
    assert all([y.shape[1] == P for y in ys])
    assert all([X.shape[0] == y.shape[0] for X, y in zip(Xs, ys)])

    mu0 = mu0 * torch.ones((P, D), dtype=torch.float64)
    sigma0 = sigma0 * torch.eye(D, dtype=torch.float64)

    # Make sure the weights are the weights
    if weights is not None:
        weights = weights if isinstance(weights, (list, tuple)) else [weights]
    else:
        weights = [torch.ones(X.shape[0], dtype=torch.float64) for X in Xs]

    # Add weak prior on intercept
    if fit_intercept:
        mu0 = torch.cat((mu0, torch.zeros(P, dtype=torch.float64)[:, None]), dim=1)
        sigma0 = torch.from_numpy(block_diag(sigma0, torch.eye(1, dtype=torch.float64)))

    # Compute the posterior
    J = torch.inverse(sigma0)
    h = torch.mm(J, mu0.T)
    # np.dot(J, mu0.T)

    for X, y, weight in zip(Xs, ys, weights):
        X = torch.cat((X, torch.ones(X.shape[0], dtype=torch.float64)[:, None]), dim=-1) if fit_intercept else X
        J += torch.mm(X.T * weight, X)
        h += torch.mm(X.T * weight, y)

    # Solve for the MAP estimate
    # W = np.linalg.solve(J, h).T
    # W = np.dot(h.T, np.linalg.pinv(J))
    # WT, _, _, _ = np.linalg.lstsq(J, h, rcond=None)
    WT, _ = torch.lstsq(h, J)
    W = WT.T

    if fit_intercept:
        W, b = W[:, :-1], W[:, -1]
    else:
        b = 0

    # Compute the residual and the posterior variance
    nu = nu0
    Psi = psi0 * torch.eye(P, dtype=torch.float64)
    for X, y, weight in zip(Xs, ys, weights):
        yhat = torch.mm(X, W.T) + b
        resid = y - yhat
        nu += torch.sum(weight)
        tmp = torch.einsum('t,ti,tj->ij', weight, resid, resid)
        # tmp = np.sum(weight[:, None, None] * resid[:, :, None] * resid[:, None, :], axis=0)
        # assert np.allclose(tmp1, tmp2)
        Psi += tmp

    # Get MAP estimate of posterior covariance
    Sigma = Psi / (nu + P + 1)
    if fit_intercept:
        return W, b, Sigma
    else:
        return W, Sigma


def to_float(arr, device=torch.device('cpu')):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float().to(device)
    elif isinstance(arr, torch.FloatTensor):
        return arr.to(device)
    else:
        raise arr


def np_float(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().double().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError


def ensure_args_torch_floats(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        _args = []
        for arg in args:
            if isinstance(arg, list):
                print(self.device)
                _args.append([to_float(_arr, self.device) for _arr in arg])
            elif isinstance(arg, np.ndarray):
                _args.append(to_float(arg, self.device))
            else:
                _args.append(arg)

        return f(self, *_args, **kwargs)
    return wrapper


def ensure_res_numpy_floats(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        outputs = f(self, *args, **kwargs)

        _outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor):
                _outputs.append(np_float(out))
            elif isinstance(out, list):
                _outputs.append([np_float(x) for x in out])

        return _outputs
    return wrapper

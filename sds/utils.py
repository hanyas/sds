import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad, value_and_grad

from autograd.scipy.special import logsumexp
from autograd.scipy.linalg import block_diag

from autograd.misc import flatten
from autograd.wrap_util import wraps

from scipy.optimize import linear_sum_assignment, minimize

from functools import partial


def sample_env(env, nb_rollouts, nb_steps, ctl=None, max_act=2.):
    obs, act = [], []

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    for n in range(nb_rollouts):
        _obs = np.empty((nb_steps, dm_obs))
        _act = np.empty((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                u = 2. * max_act * npr.randn(1, )
            else:
                u = ctl.actions(x, stoch=True)

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
        obs = [np.atleast_2d(obs)] if not isinstance(obs, (list, tuple)) else obs

        if act is None:
            act = []
            for _obs in obs:
                act.append(np.zeros((_obs.shape[0], self.dm_act)))

        act = [np.atleast_2d(act)] if not isinstance(act, (list, tuple)) else act

        return f(self, obs, act, **kwargs)

    return wrapper


def flatten_to_dim(X, d):
    """
    Flatten an array of dimension k + d into an array of dimension 1 + d.
    Example:
        X = npr.rand(10, 5, 2, 2)
        flatten_to_dim(X, 4).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 3).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 2).shape # (50, 2, 2)
        flatten_to_dim(X, 1).shape # (100, 2)
    Parameters
    ----------
    X : array_like
        The array to be flattened.  Must be at least d dimensional
    d : int (> 0)
        The number of dimensions to retain.  All leading dimensions are flattened.
    Returns
    -------
    flat_X : array_like
        The input X flattened into an array dimension d (if X.ndim == d)
        or d+1 (if X.ndim > d)
    """
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def undefined_division(a, b):
    # handle bad division
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0.0  # -inf inf NaN
    return c


def normalize(x, dim):
    norm = np.sum(x, axis=dim, keepdims=True)
    c = undefined_division(x, norm)
    return c, norm.flatten()


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


def relu(x):
    return np.maximum(0, x)


def logistic_regression(X, y, bias=None, K=None,
                        W0=None, mu0=0, sigmasq0=1,
                        verbose=False, maxiter=1000):
    """
    Fit a multiclass logistic regression

        y_i ~ Cat(softmax(W x_i))

    y is a one hot vector in {0, 1}^K
    x_i is a vector in R^D
    W is a matrix R^{K x D}

    The log likelihood is,

        L(W) = sum_i sum_k y_ik * w_k^T x_i - logsumexp(W x_i)

    The prior is w_k ~ Norm(mu0, diag(sigmasq0)).
    """
    N, D = X.shape
    assert y.shape[0] == N

    # Make sure y is one hot
    if y.ndim == 1 or y.shape[1] == 1:
        assert y.dtype == int and y.min() >= 0
        K = y.max() + 1 if K is None else K
        y_oh = np.zeros((N, K), dtype=int)
        y_oh[np.arange(N), y] = 1

    else:
        K = y.shape[1]
        assert y.min() == 0 and y.max() == 1 and np.allclose(y.sum(1), 1)
        y_oh = y

    # Check that bias is correct shape
    if bias is not None:
        assert bias.shape == (K,) or bias.shape == (N, K)
    else:
        bias = np.zeros((K,))

    def loss(W_flat):
        W = np.reshape(W_flat, (K, D))
        scores = np.dot(X, W.T) + bias
        lp = np.sum(y_oh * scores) - np.sum(logsumexp(scores, axis=1))
        prior = np.sum(-0.5 * (W - mu0)**2 / sigmasq0)
        return -(lp + prior) / N

    W0 = W0 if W0 is not None else np.zeros((K, D))
    assert W0.shape == (K, D)

    itr = [0]

    def callback(W_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0], loss(W_flat)))

    result = minimize(loss, np.ravel(W0), jac=grad(loss),
                      method="BFGS",
                      callback=callback if verbose else None,
                      options=dict(maxiter=maxiter, disp=verbose))

    W = np.reshape(result.x, (K, D))
    return W


def linear_regression(Xs, ys, weights=None,
                      mu0=0, sigmasq0=1,
                      nu0=1, Psi0=1,
                      fit_intercept=True):

    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = ys if isinstance(ys, (list, tuple)) else [ys]
    assert len(Xs) == len(ys)

    D = Xs[0].shape[1]
    P = ys[0].shape[1]
    assert all([X.shape[1] == D for X in Xs])
    assert all([y.shape[1] == P for y in ys])
    assert all([X.shape[0] == y.shape[0] for X, y in zip(Xs, ys)])

    mu0 = mu0 * np.zeros((P, D))
    sigmasq0 = sigmasq0 * np.eye(D)

    # Make sure the weights are the weights
    if weights is not None:
        weights = weights if isinstance(weights, (list, tuple)) else [weights]
    else:
        weights = [np.ones(X.shape[0]) for X in Xs]

    # Add weak prior on intercept
    if fit_intercept:
        mu0 = np.column_stack((mu0, np.zeros(P)))
        sigmasq0 = block_diag(sigmasq0, np.eye(1))

    # Compute the posterior
    J = np.linalg.inv(sigmasq0)
    h = np.dot(J, mu0.T)

    for X, y, weight in zip(Xs, ys, weights):
        X = np.column_stack((X, np.ones(X.shape[0]))) if fit_intercept else X
        J += np.dot(X.T * weight, X)
        h += np.dot(X.T * weight, y)

    # Solve for the MAP estimate
    W = np.linalg.solve(J, h).T
    if fit_intercept:
        W, b = W[:, :-1], W[:, -1]
    else:
        b = 0

    # Compute the residual and the posterior variance
    nu = nu0
    Psi = Psi0 * np.eye(P)
    for X, y, weight in zip(Xs, ys, weights):
        yhat = np.dot(X, W.T) + b
        resid = y - yhat
        nu += np.sum(weight)
        tmp1 = np.einsum('t,ti,tj->ij', weight, resid, resid)
        tmp2 = np.sum(weight[:, None, None] * resid[:, :, None] * resid[:, None, :], axis=0)
        assert np.allclose(tmp1, tmp2)
        Psi += tmp1

    # Get MAP estimate of posterior covariance
    Sigma = Psi / (nu + P + 1)
    if fit_intercept:
        return W, b, Sigma
    else:
        return W, Sigma


def unflatten_optimizer_step(step):
    """
    Wrap an optimizer step function that operates on flat 1D arrays
    with a version that handles trees of nested containers,
    i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
    """
    @wraps(step)
    def _step(value_and_grad, x, itr, state=None, *args, **kwargs):
        _x, unflatten = flatten(x)

        def _value_and_grad(x, i):
            v, g = value_and_grad(unflatten(x), i)
            return v, flatten(g)[0]

        _next_x, _next_val, _next_g, _next_state = \
            step(_value_and_grad, _x, itr, state=state, *args, **kwargs)
        return unflatten(_next_x), _next_val, _next_g, _next_state
    return _step


@unflatten_optimizer_step
def adam_step(value_and_grad, x, itr, state=None, step_size=0.001,
              b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    """
    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    val, g = value_and_grad(x, itr)
    m = (1 - b1) * g + b1 * m    # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
    mhat = m / (1 - b1**(itr + 1))    # Bias correction.
    vhat = v / (1 - b2**(itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, val, g, (m, v)


def _generic_sgd(method, loss, x0, callback=None,
                 nb_iters=200, state=None, full_output=False):
    """
    Generic stochastic gradient descent step.
    """
    step = dict(adam=adam_step)[method]

    # Initialize outputs
    x, losses, grads = x0, [], []
    for itr in range(nb_iters):
        x, val, g, state = step(value_and_grad(loss), x, itr, state)
        losses.append(val)
        grads.append(g)

    if full_output:
        return x, state
    else:
        return x


def _generic_minimize(method, loss, x0, verbose=False, nb_iters=1000,
                      state=None, full_output=False):
    """
    Minimize a given loss function with scipy.optimize.minimize.
    """
    # Flatten the loss
    _x0, unflatten = flatten(x0)
    _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)

    if verbose:
        print("Fitting with {}.".format(method))

    # Specify callback for fitting
    itr = [0]

    def callback(x_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0], loss(unflatten(x_flat), -1)))

    # Call the optimizer.
    # HACK: Pass in -1 as the iteration.
    result = minimize(_objective, _x0, args=(-1,), jac=grad(_objective),
                      method=method,
                      callback=callback if verbose else None,
                      options=dict(maxiter=nb_iters, disp=verbose))

    if full_output:
        return unflatten(result.x), result
    else:
        return unflatten(result.x)


# Define optimizers
adam = partial(_generic_sgd, "adam")
bfgs = partial(_generic_minimize, "BFGS")
lbfgs = partial(_generic_minimize, "L-BFGS-B")

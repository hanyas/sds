import autograd.numpy as np
from autograd import grad, value_and_grad

from autograd.scipy.misc import logsumexp
from autograd.scipy.linalg import block_diag

from autograd.misc import flatten
from autograd.wrap_util import wraps

from scipy.optimize import linear_sum_assignment, minimize

from functools import partial


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


def batch_mahalanobis(L, x):
    """
    Compute the squared Mahalanobis distance.
    :math:`x^T M^{-1} x` for a factored :math:`M = LL^T`.
    Copied from PyTorch torch.distributions.multivariate_normal.
    Parameters
    ----------
    L : array_like (..., D, D)
        Cholesky factorization(s) of covariance matrix
    x : array_like (..., D)
        Points at which to evaluate the quadratic term
    Returns
    -------
    y : array_like (...,)
        squared Mahalanobis distance :math:`x^T (LL^T)^{-1} x`
    """
    # Flatten the Cholesky into a (-1, D, D) array
    flat_L = flatten_to_dim(L, 2)
    # Invert each of the K arrays and reshape like L
    L_inv = np.reshape(np.array([np.linalg.inv(Li.T) for Li in flat_L]), L.shape)
    # Reshape x into (..., D, 1); dot with L_inv^T; square and sum.
    return np.sum(np.sum(x[..., None] * L_inv, axis=-2)**2, axis=-1)


def _multivariate_normal_logpdf(data, mus, Sigmas, Ls=None):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.
    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density
    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)
    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)
    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas
    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # Quadratic term
    lp = -0.5 * batch_mahalanobis(Ls, data - mus)                    # (...,)
    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp


def multivariate_normal_logpdf(data, mus, Sigmas, mask=None):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least compatible) leading dimensions.
    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density
    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)
    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)
    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed
    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D

    # If there's no mask, we can just use the standard log pdf code
    if mask is None:
        return _multivariate_normal_logpdf(data, mus, Sigmas)

    # Otherwise we need to separate the data into sets with the same mask,
    # since each one will entail a different covariance matrix.
    #
    # First, determine the output shape. Allow mus and Sigmas to
    # have different shapes; e.g. many Gaussians with the same
    # covariance but different means.
    shp1 = np.broadcast(data, mus).shape[:-1]
    shp2 = np.broadcast(data[..., None], Sigmas).shape[:-2]
    assert len(shp1) == len(shp2)
    shp = tuple(max(s1, s2) for s1, s2 in zip(shp1, shp2))

    # Broadcast the data into the full shape
    full_data = np.broadcast_to(data, shp + (D,))

    # Get the full mask
    assert mask.dtype == bool
    assert mask.shape == data.shape
    full_mask = np.broadcast_to(mask, shp + (D,))

    # Flatten the mask and get the unique values
    flat_data = flatten_to_dim(full_data, 1)
    flat_mask = flatten_to_dim(full_mask, 1)
    unique_masks, mask_index = np.unique(flat_mask, return_inverse=True, axis=0)

    # Initialize the output
    lls = np.nan * np.ones(flat_data.shape[0])

    # Compute the log probability for each mask
    for i, this_mask in enumerate(unique_masks):
        this_inds = np.where(mask_index == i)[0]
        this_D = np.sum(this_mask)
        if this_D == 0:
            lls[this_inds] = 0
            continue

        this_data = flat_data[np.ix_(this_inds, this_mask)]
        this_mus = mus[..., this_mask]
        this_Sigmas = Sigmas[np.ix_(*[np.ones(sz, dtype=bool) for sz in Sigmas.shape[:-2]], this_mask, this_mask)]

        # Precompute the Cholesky decomposition
        this_Ls = np.linalg.cholesky(this_Sigmas)

        # Broadcast mus and Sigmas to full shape and extract the necessary indices
        this_mus = flatten_to_dim(np.broadcast_to(this_mus, shp + (this_D,)), 1)[this_inds]
        this_Ls = flatten_to_dim(np.broadcast_to(this_Ls, shp + (this_D, this_D)), 2)[this_inds]

        # Evaluate the log likelihood
        lls[this_inds] = _multivariate_normal_logpdf(this_data, this_mus, this_Sigmas, Ls=this_Ls)

    # Reshape the output
    assert np.all(np.isfinite(lls))
    return np.reshape(lls, shp)


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
    assert K1 <= K2, "Can only find permutation from more states to fewer"

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
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
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
                      alpha0=1, beta0=1,
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
    alpha = alpha0
    beta = beta0 * np.ones(P)
    for X, y, weight in zip(Xs, ys, weights):
        yhat = np.dot(X, W.T) + b
        resid = y - yhat
        alpha += 0.5 * np.sum(weight)
        beta += 0.5 * np.sum(weight[:, None] * resid**2, axis=0)

    # Get MAP estimate of posterior mode of precision
    sigmasq = beta / (alpha + 1e-16)

    if fit_intercept:
        return W, b, sigmasq
    else:
        return W, sigmasq


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
def adam_step(value_and_grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
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


def _generic_sgd(method, loss, x0, callback=None, num_iters=200, step_size=0.1, mass=0.9, full_output=False):
    """
    Generic stochastic gradient descent step.
    """
    step = dict(adam=adam_step)[method]

    # Initialize outputs
    x, losses, grads, state = x0, [], [], None
    for itr in range(num_iters):
        x, val, g, state = step(value_and_grad(loss), x, itr, state)
        losses.append(val)
        grads.append(g)

    if full_output:
        return x, losses, grads
    else:
        return x


def _generic_minimize(method, loss, x0, verbose=False, num_iters=1000):
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
                      options=dict(maxiter=num_iters, disp=verbose))

    # if verbose:
    #     print("{} completed with message: \n{}".format(method, result.message))
    #
    # if not result.success:
    #     warn("{} failed with message:\n{}".format(method, result.message))

    return unflatten(result.x)


# Define optimizers
adam = partial(_generic_sgd, "adam")
bfgs = partial(_generic_minimize, "BFGS")

from autograd import numpy as np
from sds.utils import flatten_to_dim


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
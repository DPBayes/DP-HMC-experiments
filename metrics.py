import numpy as np
import numba
import timeit

def total_mean_error(samples, true_samples):
    """
    Return the Euclidean distance between the means of two given samples.
    """
    return np.sqrt(np.sum(component_mean_error(samples, true_samples)**2, axis=0))

def component_mean_error(samples, true_samples):
    """
    Return the difference between the means of the two given samples.
    """
    return np.mean(samples, axis=0) - np.mean(true_samples, axis=0).reshape(-1, 1)

def component_var_error(samples, true_samples):
    """
    Return the difference between the variances of the two given samples.
    """
    return np.var(samples, axis=0) - np.var(true_samples, axis=0).reshape(-1, 1)

def split_r_hat(chains):
    """
    Compute split-R-hat for the given chains.

    Parameters
    ----------
    chains : ndarray
        The chains as an array of shape (num_samples, num_dimensions, num_chains).
    """
    n_samples, dim, num_chains = chains.shape
    # If the number of samples if not even, discard the last sample
    if n_samples % 2 != 0:
        chains = chains[0:n_samples-1, :, :]
    return r_hat(np.concatenate(np.array_split(chains, 2, axis=0), axis=2))

def r_hat(chains):
    """
    Compute R-hat for the given chains.

    Parameters
    ----------
    chains : ndarray
        The chains as an array of shape (num_samples, num_dimensions, num_chains).
    """
    chains = np.transpose(chains, axes=(2, 0, 1))
    m, n, d = chains.shape
    chain_means = np.mean(chains, axis=1)
    total_means = np.mean(chain_means, axis=0)
    B = n / (m - 1) * np.sum((chain_means - total_means)**2, axis=0)
    s2s = np.var(chains, axis=1, ddof=1)
    W = np.mean(s2s, axis=0)
    var = (n - 1) / n * W + 1 / n * B
    r_hats = np.sqrt(var / W)
    return r_hats

def mmd(samples, true_samples):
    """
    Return MMD between two samples.

    Both arguments must be arrays either of shape
    (num_samples, num_dimensions, num_chains),
    or of shape (num_samples, num_dimensions), which is treated as if
    num_chains = 1.

    Returns
    -------
    ndarray
        MMD for each chain.
    """
    if len(samples.shape) == 2:
        n, dim = samples.shape
        chains = 1
    elif len(samples.shape) == 3:
        n, dim, chains = samples.shape
    else:
        raise ValueError("samples must be 2 or 3-dimensional")
    mmd = np.zeros(chains)
    for i in range(chains):
        mmd[i] = numba_mmd(np.asarray(samples[:, :, i]), np.asarray(true_samples))
    return mmd

@numba.njit
def kernel(x1, x2, sigma):
    return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))

@numba.njit
def numba_mmd(sample1, sample2):
    subset1 = sample1[np.random.choice(sample1.shape[0], 500, replace=True), :]
    subset2 = sample2[np.random.choice(sample2.shape[0], 500, replace=True), :]
    distances = np.sqrt(np.sum((subset1 - subset2)**2, axis=1))
    sigma = np.median(distances)

    n = sample1.shape[0]
    m = sample2.shape[0]

    term1 = 0.0
    for i in range(0, n):
        for j in range(i + 1, n):
            term1 += kernel(sample1[i, :], sample1[j, :], sigma)
    term2 = 0.0
    for i in range(0, m):
        for j in range(i + 1, m):
            term2 += kernel(sample2[i, :], sample2[j, :], sigma)
    term3 = 0.0
    for i in range(n):
        for j in range(m):
            term3 += kernel(sample1[i, :], sample2[j, :], sigma)
    return np.sqrt(np.abs(2 * term1 / (n * (n - 1)) + 2 * term2 / (m * (m - 1)) - 2 * term3 / (n * m)))

"""
DP-SGLD and DP-SGNHT implementations.

This module can be run as a script to test DP-SGLD on the banana distribution.
"""
import jax
import jax.numpy as np
import numpy as npa
import fourier_accountant as fa
import result

class SGLDParams:
    """
    Parameters for DP-SGLD.

    Parameters
    ----------
    subsample_size : int
    clip_bound : float
    eta : float
        Step size.
    """
    def __init__(self, subsample_size, clip_bound, eta):
        self.subsample_size = subsample_size
        self.clip_bound = clip_bound
        self.eta = eta

class SGNHTParams:
    """
    Parameters for DP-SGNHT.

    Parameters
    ----------
    subsample_size : int
    clip_bound : float
    eta : float
        Step size.
    A : float
        Controls the amount of noise added to the gradients. Larger values
        add more noise and allow additional iterations.
    """
    def __init__(self, subsample_size, clip_bound, eta, A):
        self.subsample_size = subsample_size
        self.clip_bound = clip_bound
        self.eta = eta
        self.A = A

def adp_delta(iters, epsilon, eta, sigma_mul, n, b, clip_bound):
    if iters <= 0:
        return 0

    print("Iters: {}".format(iters))

    sigma = b / (2 * clip_bound * np.sqrt(eta * n)) * sigma_mul**0.5
    q = b/n
    return fa.get_delta_S(target_eps=epsilon, sigma=sigma, q=q, ncomp=iters)

def adp_iters(epsilon, delta, eta, sigma_mul, n, b, clip_bound):
    """
    Compute the number of iterations DP-SGLD and DP-SGNHT can run for.

    Parameters
    ----------
    epsilon : float
    delta : float
    eta : float
        Step size.
    sigma_mul : float
        Multiplier for the noise standard deviation. 1 for DP-SGLD and 2 * A
        for DP-SGNHT.
    n : int
        Dataset size.
    b : int
        Subsample size.
    clip_bound : float
    """
    low_iters = 0
    up_iters = 1024
    while adp_delta(up_iters, epsilon, eta, sigma_mul, n, b, clip_bound) < delta:
        up_iters *= 2
    while int(up_iters) - int(low_iters) > 1:
        new_iters = (low_iters + up_iters) / 2
        new_delta = adp_delta(new_iters, epsilon, eta, sigma_mul, n, b, clip_bound)
        if new_delta > delta:
            up_iters = new_iters
        else:
            low_iters = new_iters

    if adp_delta(int(up_iters), epsilon, eta, sigma_mul, n, b, clip_bound) < delta:
        return int(up_iters)
    else:
        return int(low_iters)


def sgld(problem, theta0, epsilon, delta, params, chains,
         repeats=1, verbose=True, seed=4327467, replacement=False, thin_to=5000):
    """
    Run DP-SGLD.

    Implemenatation of the DP-SGLD algorithm. Unless the `seed` argument is given,
    a default value for the random number generator seed is used, so this function will always
    return the same results. Using using a non-default value for any of
    `repeats`, `replacement` will NOT provide
    the privacy bounds given by `epsilon` and `delta`.

    Parameters
    ----------
    problem : Problem
        The specification of the model and data to use.
    theta0 : ndarray
        The starting points each chain and repeat, as an ndarray with shape
        (problem.dim, repeats * chains).
    epsilon : float
    delta : float
    params : SGLDParams
        The parameters for DP-SGLD.
    chains : int
        The number of chains to run in pararrel. Each chain is run
    repeats : int, default 1
        The number of times to repeat the run.
    verbose : bool, default True
        If True, print the number of iterations the algorithm will run for
        before running the chains and print progress updates every 100 iterations.
    seed : int, optional
        Seed for the random number generator. By default, use an arbitrary, but
        fixed value.
    replacement : bool, default False
        If True, use subsampling with replacement, which is faster, but does
        not meet the advertised privacy bounds.
    thin_to : int, default 5000
        Thin the resulting chains to have approximately `thin_to` samples each.

    Returns
    -------
    MCMCResult or list of MCMCResult
        The results from running DP-SGLD. If `repeats` is set to 1, returns
        a single object, otherwise returns an MCMCResult object for each repeat
        as a list.
    """
    data = problem.data
    n, data_dim = data.shape
    total_chains = chains * repeats
    dim, num_theta0 = theta0.shape

    if num_theta0 != total_chains:
        raise ValueError("Expected {} theta0 values but got {}".format(chains * repeats, num_theta0))

    iters = adp_iters(epsilon, delta, params.eta, 1, n, params.subsample_size, params.clip_bound)
    iters = int(iters / chains)
    thinning = int(iters / thin_to) if iters > thin_to else 1
    if verbose:
        print("Iterations: {}".format(iters))

    chain = np.zeros((iters + 1, dim, total_chains))
    chain = jax.ops.index_update(chain, jax.ops.index[0,:, :], theta0)
    clipped_grads = np.zeros((iters, total_chains))

    rng_key = jax.random.PRNGKey(seed)

    def subsample_grads(clip_bound, theta, data, inds):
        subsample = data[inds]
        return problem.log_likelihood_grad_clipped(clip_bound, theta, subsample)
    grad_fun = jax.vmap(subsample_grads, (None, 1, None, 1), (1, 0))

    for i in range(iters):
        current = chain[i, :, :]
        rng_key, subsample_key, noise_key = jax.random.split(rng_key, 3)
        subsample_inds = jax.random.choice(
            subsample_key, n, (params.subsample_size, total_chains),
            replace=replacement
        )

        eta = params.eta / n
        noise = jax.random.normal(noise_key, shape=(dim, total_chains)) * eta**0.5
        pri_grad = jax.vmap(problem.log_prior_grad, 1, 1)(current)
        ll_grad, clipped = grad_fun(params.clip_bound, current, data, subsample_inds)
        clipped_grads = jax.ops.index_update(clipped_grads, jax.ops.index[i, :], clipped)

        proposal = current + eta * (pri_grad + n * ll_grad / params.subsample_size) + noise
        chain = jax.ops.index_update(chain, jax.ops.index[i + 1, :, :], proposal)

        if not np.isfinite(proposal).all():
            raise Exception("Iteration diverged.")
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))
            print("Eta: {}".format(eta))

    thinned_chain = chain[np.arange(1, iters + 1, thinning)]
    thinned_samples = thinned_chain.shape[0]
    return result.split_results(
        thinned_chain, np.repeat(thinned_samples - 1, repeats), np.zeros(repeats),
        np.sum(clipped_grads, axis=0) / iters / params.subsample_size,
        repeats, epsilon, delta, params
    )

def sgnht(problem, theta0, epsilon, delta, params, chains, repeats=1,
          verbose=True, seed=4327467, replacement=False, thin_to=5000, p_refresh=None):
    """
    Run DP-SGNHT.

    Implemenatation of the DP-SGNHT algorithm. Unless the `seed` argument is given,
    a default value for the random number generator seed is used, so this function will always
    return the same results. Using using a non-default value for any of
    `repeats`, `replacement` will NOT provide
    the privacy bounds given by `epsilon` and `delta`.

    Parameters
    ----------
    problem : Problem
        The specification of the model and data to use.
    theta0 : ndarray
        The starting points each chain and repeat, as an ndarray with shape
        (problem.dim, repeats * chains).
    epsilon : float
    delta : float
    params : SGNHTParams
        The parameters for DP-SGNHT.
    chains : int
        The number of chains to run in pararrel. Each chain is run
    repeats : int, default 1
        The number of times to repeat the run.
    verbose : bool, default True
        If True, print the number of iterations the algorithm will run for
        before running the chains and print progress updates every 100 iterations.
    seed : int, optional
        Seed for the random number generator. By default, use an arbitrary, but
        fixed value.
    replacement : bool, default False
        If True, use subsampling with replacement, which is faster, but does
        not meet the advertised privacy bounds.
    thin_to : int, default 5000
        Thin the resulting chains to have approximately `thin_to` samples each.
    p_refresh : int, optional
        If set, resample momentun every `p_refresh` iterations.

    Returns
    -------
    MCMCResult or list of MCMCResult
        The results from running DP-SGNHT. If `repeats` is set to 1, returns
        a single object, otherwise returns an MCMCResult object for each repeat
        as a list.
    """
    data = problem.data
    n, data_dim = data.shape
    dim, num_theta0 = theta0.shape
    total_chains = chains * repeats

    if num_theta0 != total_chains:
        raise ValueError("Expected {} theta0 values but got {}".format(chains * repeats, num_theta0))

    A = params.A

    iters = adp_iters(epsilon, delta, params.eta, 2*A, n, params.subsample_size, params.clip_bound)
    iters = int(iters / chains)
    thinning = int(iters / thin_to) if iters > thin_to else 1
    if verbose:
        print("Iterations: {}".format(iters))

    chain = np.zeros((iters + 1, dim, total_chains))
    chain = jax.ops.index_update(chain, jax.ops.index[0,:, :], theta0)
    clipped_grads = np.zeros((iters, total_chains))

    rng_key = jax.random.PRNGKey(seed)

    def subsample_grads(clip_bound, theta, data, inds):
        subsample = data[inds]
        return problem.log_likelihood_grad_clipped(clip_bound, theta, subsample)
    grad_fun = jax.vmap(subsample_grads, (None, 1, None, 1), (1, 0))

    rng_key, p_key = jax.random.split(rng_key, 2)

    p = jax.random.normal(p_key, (dim, total_chains))
    xi = np.repeat(A, total_chains)

    for i in range(iters):
        current = chain[i, :, :]
        rng_key, subsample_key, noise_key = jax.random.split(rng_key, 3)
        subsample_inds = jax.random.choice(
            subsample_key, n, (params.subsample_size, total_chains),
            replace=replacement
        )

        eta = params.eta / n
        noise = jax.random.normal(noise_key, shape=(dim, total_chains)) * (2 * eta * A)**0.5

        pri_grad = jax.vmap(problem.log_prior_grad, 1, 1)(current)
        ll_grad, clipped = grad_fun(params.clip_bound, current, data, subsample_inds)
        clipped_grads = jax.ops.index_update(clipped_grads, jax.ops.index[i, :], clipped)

        p_next = p - xi * eta * p + eta * (pri_grad + n * ll_grad / params.subsample_size) + noise
        proposal = current + eta * p
        xi = xi + eta * (np.sum(p**2, axis=0) / dim - 1)
        chain = jax.ops.index_update(chain, jax.ops.index[i + 1, :, :], proposal)
        p = p_next

        if not np.isfinite(proposal).all():
            raise Exception("Iteration diverged.")
        if p_refresh is not None and (i + 1) % p_refresh == 0:
            p_key, use_key = jax.random.split(p_key)
            p = jax.random.normal(p_key, (dim, total_chains))
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    thinned_chain = chain[np.arange(1, iters + 1, thinning)]
    thinned_samples = thinned_chain.shape[0]
    return result.split_results(
        thinned_chain, np.repeat(thinned_samples - 1, repeats), np.zeros(repeats),
        np.sum(clipped_grads, axis=0) / iters / params.subsample_size,
        repeats, epsilon, delta, params
    )

if __name__ == "__main__":
    import experiments
    from plot_summary import plot_chain_summary

    problem = experiments.experiments["banana"]
    epsilon = 12
    delta = 0.1 / problem.data.shape[0]
    chains = 4
    repeats = 1
    theta0 = np.vstack([problem.get_start_point(i) for i in range(chains * repeats)]).transpose()
    params = SGLDParams(
        subsample_size = 1000,
        clip_bound = 0.2,
        eta = 1
    )
    result = sgld(
        problem, theta0, epsilon, delta, params, chains,
        repeats=repeats, seed=4257757
    )
    # result = sgnht(
    #     problem, theta0, epsilon, delta, params, chains,
    #     repeats=repeats, seed=4257757, replacement=True, p_refresh=None
    # )

    metric_res = result.compute_metrics(problem.true_posterior)
    final_chain = result.get_final_chain()
    print(metric_res)
    plot_chain_summary(problem, result, theta0)

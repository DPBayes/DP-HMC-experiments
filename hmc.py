"""
DP-HMC implementation.

This module can be run as a standalone script to test the DP-HMC algorithm on
the banana distribution.
"""

import jax
import jax.numpy as np
import jax.scipy.special as spec
import result
import halton

class HMCParams:
    """
    DP-HMC parameters.

    Parameters
    ----------
    tau : float
        `tau` controls the tradeoff between more iterations and less noise
        for log-likelihood ratios: larger values add more noise, and allow
        more iterations.
    tau_g : float
        `tau_g` controls the tradeoff between more iterations and less noise
        for gradients: larger values add more noise, and allow more iterations.
    L : int
        Number of steps for the leapfrog simulation.
    eta : float
        Step size of the leapfrog simulation.
    mass : ndarray or float
        Mass of the particle simulated by HMC. Only scalar or vector
        values are supported. A vector value is treated as the diagonal
        of the full mass matrix.
    r_clip : float
        Log-likelihood ratio clip bound.
    grad_clip : float
        Gradient clip bound.
    """
    def __init__(self, tau, tau_g, L, eta, mass, r_clip, grad_clip):
        self.tau = tau 
        self.tau_g = tau_g 
        self.L = L
        self.eta = eta 
        self.mass = mass 
        self.r_clip = r_clip 
        self.grad_clip = grad_clip

class GradClipCounter:
    def __init__(self, total_chains):
        self.clipped_grad = np.zeros(total_chains)
        self.grad_accesses = 0

    def add_clips(self, clipped_grad):
        self.grad_accesses += 1
        self.clipped_grad += clipped_grad

def zcdp_iters(epsilon, delta, params, n, compute_less_grad=False):
    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    rho_l = 1 / (2 * params.tau**2 * n)
    rho_g = 1 / (2 * params.tau_g**2 * n)

    if compute_less_grad:
        iters = int((rho - rho_g) / (rho_l + params.L * rho_g))
    else:
        iters = int(rho / (rho_l + (params.L + 1) * rho_g))
    return iters

def adp_delta(k, epsilon, params, n, compute_less_grad=False):
    tau_l = params.tau
    tau_g = params.tau_g
    L = params.L
    grad_evals = k * L + 1 if compute_less_grad else k * (L + 1)
    mu = k / (2 * tau_l**2 * n) + grad_evals / (2 * tau_g**2 * n)
    term1 = spec.erfc((epsilon - mu) / (2 * np.sqrt(mu)))
    term2 = np.exp(epsilon) * spec.erfc((epsilon + mu) / (2 * np.sqrt(mu)))
    return (0.5 * (term1 - term2)).sum()

def adp_iters(epsilon, delta, params, n, compute_less_grad=False):
    """
    Compute the number of iteratios DP-HMC can run for.

    Parameters
    ----------
    epsilon : float
    delta : float
    params : HMCParams
        Parameters for DP-HMC.
    n : int
        The size of the dataset.
    compute_less_grad : bool, default False
        If set to True, compute the number of iterations when using
        kL + 1 gradient evaluations instead of k(L + 1) where k is the number
        of iterations and L is the number of leapfrog steps.

    Returns
    -------
    int
        The number of iterations DP-HMC can run for.
    """
    low_iters = zcdp_iters(epsilon, delta, params, n, compute_less_grad)
    up_iters = max(low_iters, 1)
    while adp_delta(up_iters, epsilon, params, n, compute_less_grad) < delta:
        up_iters *= 2
    while int(up_iters) - int(low_iters) > 1:
        new_iters = (low_iters + up_iters) / 2
        new_delta = adp_delta(new_iters, epsilon, params, n, compute_less_grad)
        if new_delta > delta:
            up_iters = new_iters
        else:
            low_iters = new_iters

    if adp_delta(int(up_iters), epsilon, params, n, compute_less_grad) < delta:
        return int(up_iters)
    else:
        return int(low_iters)

def hmc(problem, theta0, epsilon, delta, params, chains, repeats=1,
        verbose=True, use_adp=True, seed=42387742,
        no_ll_noise=False, no_grad_noise=False, iters=None):
    """
    Run DP-HMC.

    Implemenatation of the DP-HMC algorithm. Unless the `seed` argument is given,
    a default value for the random number generator seed is used, so this function will always
    return the same results. Using using a non-default value for any of
    `repeats`, `no_ll_noise`, `no_grad_noise` or `iters` will NOT provide
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
    params : HMCParams
        The parameters for DP-HMC.
    chains : int
        The number of chains to run in pararrel. Each chain is run
    repeats : int, default 1
        The number of times to repeat the run.
    verbose : bool, default True
        If True, print the number of iterations the algorithm will run for
        before running the chains and print progress updates every 100 iterations.
    use_adp : bool, default True
        If True, compute the number of iterations using the tight PLD based
        bound. Otherwise, compute the number of iterations using zCDP, which
        will give a smaller number of iterations.
    seed : int, optional
        Seed for the random number generator. By default, use an arbitrary, but
        fixed value.
    no_ll_noise : bool, default False
        Disable noise added to log-likelihood ratios.
    no_grad_noise : bool, default False
        Disable noise added to gradients.
    iters : int, optional
        If set, run for `iters` iterations instead of computing the number of
        iterations. Note that `epsilon` and `delta` must still be set when
        `iters` is set, but their values are not used.

    Returns
    -------
    MCMCResult or list of MCMCResult
        The results from running DP-HMC. If `repeats` is set to 1, returns
        a single object, otherwise returns an MCMCResult object for each repeat
        as a list.
    """
    data = problem.data
    n, data_dim = data.shape
    temp_scale = problem.temp_scale
    dim, num_theta0 = theta0.shape
    total_chains = chains * repeats
    if num_theta0 != total_chains:
        raise ValueError("Expected {} theta0 values but got {}".format(chains * repeats, num_theta0))

    tau = params.tau 
    tau_g = params.tau_g
    L = params.L
    eta = params.eta
    mass = params.mass
    r_clip = params.r_clip
    grad_clip = params.grad_clip

    if iters is None:
        if not use_adp:
            iters = zcdp_iters(epsilon, delta, params, n, False)
        else:
            iters = adp_iters(epsilon, delta, params, n, False)
        iters = int(iters / chains)
    if verbose:
        print("Iterations: {}".format(iters))

    sigma = tau * np.sqrt(n)
    if no_ll_noise:
        sigma = 0

    chain = np.zeros((iters + 1, dim, total_chains))
    chain = jax.ops.index_update(chain, jax.ops.index[0, :, :], theta0)
    # leapfrog_chain = np.zeros((iters * L, dim, total_chains))
    clipped_r = np.zeros((iters, total_chains))
    clipped_grad_counter = GradClipCounter(total_chains)
    accepts = np.zeros(total_chains)

    rng = jax.random.PRNGKey(seed)
    rng, halton_rng = jax.random.split(rng)
    halton_seq = halton.halton_sequence(iters, halton_rng)

    grad_noise_sigma = 2 * tau_g * np.sqrt(n) * grad_clip
    if no_grad_noise:
        grad_noise_sigma = 0

    vmap_ll_grad = jax.vmap(problem.log_likelihood_grad_clipped, (None, 1, None), (1, 0))
    vmap_ll = jax.vmap(problem.log_likelihood_no_sum, (1, None), 1)
    vmap_prior = jax.vmap(problem.log_prior, 1, 0)

    def grad_fun(theta, noise_key):
        ll_grad, clips = vmap_ll_grad(grad_clip, theta, data)
        clipped_grad_counter.add_clips(clips)

        pri_grad = jax.vmap(problem.log_prior_grad, 1, 1)(theta)
        noise = jax.random.normal(noise_key, shape=(dim, total_chains)) * grad_noise_sigma
        return temp_scale * (ll_grad + noise) + pri_grad

    llc = vmap_ll(theta0, data)
    for i in range(iters):
        current = chain[i, :]
        rng, proposal_key, gradient_key, ll_key, accept_key = jax.random.split(rng, 5)
        #TODO: this assumes diagonal M
        p = jax.random.normal(proposal_key, (dim, total_chains)) * np.sqrt(mass)
        p_orig = p.copy()
        prop = current.copy()
        gradient_key, use_key = jax.random.split(gradient_key, 2)
        grad_new = grad_fun(current, use_key)

        h = halton_seq[i]
        h_eta = h * eta

        for j in range(L):
            p += 0.5 * h_eta * (grad_new)# - 0.5 * grad_noise_sigma**2 * p / mass)
            prop += h_eta * p / mass
            # leapfrog_chain = jax.ops.index_update(leapfrog_chain, i * L + j, prop)
            gradient_key, use_key = jax.random.split(gradient_key, 2)
            grad_new = grad_fun(prop, use_key)
            p += 0.5 * h_eta * (grad_new)# - 0.5 * grad_noise_sigma**2 * p / mass)

        if not np.isfinite(prop).all(): print("Leapfrog diverged")
        llp = vmap_ll(prop, data)
        r = llp - llc

        d = np.sqrt(np.sum((current - prop)**2, axis=0))
        clip = d * r_clip
        clipped_r = jax.ops.index_update(clipped_r, jax.ops.index[i, :], np.sum(np.abs(r) > clip, axis=0))
        r = np.clip(r, -clip, clip)

        lpp = vmap_prior(prop)
        lpc = vmap_prior(current)

        s = jax.random.normal(ll_key, shape=(total_chains,)) * sigma * d * 2 * r_clip
        dp = 0.5 * np.sum(p_orig**2 / mass, axis=0) - 0.5 * np.sum(p**2 / mass, axis=0)
        dH = dp + temp_scale * (np.sum(r, axis=0) + s) + lpp - lpc
        u = np.log(jax.random.uniform(accept_key, (total_chains,)))

        accept = u < dH - 0.5 * (temp_scale * sigma * d * 2 * r_clip)**2
        for j in range(total_chains):
            if accept[j]:
                chain = jax.ops.index_update(chain, jax.ops.index[i+1,:, j], prop[:, j])
                llc = jax.ops.index_update(llc, jax.ops.index[:, j], llp[:, j])
                accepts = jax.ops.index_update(accepts, j, accepts[j] + 1)
            else:
                chain = jax.ops.index_update(chain, jax.ops.index[i+1,:, j], current[:, j])
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    return result.split_results(
        chain, accepts, np.sum(clipped_r, axis=0) / n / iters,
        clipped_grad_counter.clipped_grad / n / clipped_grad_counter.grad_accesses,
        repeats, epsilon, delta, params
    )

if __name__ == "__main__":
    import experiments
    from plot_summary import plot_chain_summary
    import metrics

    dim = 2
    problem = experiments.experiments["banana"]
    n, data_dim = problem.data.shape
    epsilon = 12
    delta = 0.1 / n
    chains = 4
    repeats = 1
    theta0 = np.vstack([problem.get_start_point(i) for i in range(chains * repeats)]).transpose()
    params = HMCParams(
        tau = 0.10,
        tau_g = 0.55,
        eta = 0.006,
        L = 25,
        mass = 1,
        r_clip= 0.1,
        grad_clip = 0.05
    )
    result = hmc(problem, theta0, epsilon, delta, params, chains,
                 repeats=repeats)

    # for res in result:
    #     print(res.compute_metrics(problem.true_posterior))
    #     print()

    metric_res = result.compute_metrics(problem.true_posterior)
    final_chain = result.get_final_chain()
    print(metric_res)
    plot_chain_summary(problem, result, theta0)

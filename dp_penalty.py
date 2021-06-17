"""
DP-penalty implementation.

This module can be run as a script to test the DP-penalty algorithm on the
banana distribution.
"""
import jax
import jax.numpy as np
import jax.scipy.special as spec
import result

class PenaltyParams:
    """
    Parameters for DP-penalty.

    Parameters
    ----------
    tau : float
        `tau` controls the tradeoff between more iterations and less noise:
        larger values add more noise, and allow more iterations.
    r_clip : float
        Log-likelihood ratio clip bound.
    prop_sigma : float
        Standard deviation of the proposal distribution.
    ocu : bool
        One-component updates: update only one component during each iteration.
    grw : bool
        Guided random walk: for each component, keep track of a direction.
        Only make proposals in the direction associated with a component,
        and reverse the direction on reject. Requires `ocu` to be True,
        has no effect otherwise.
    """
    def __init__(self, tau, r_clip, prop_sigma, ocu, grw):
        self.tau = tau
        self.r_clip= r_clip
        self.prop_sigma = prop_sigma
        self.ocu = ocu
        self.grw = grw

def zcdp_iters(epsilon, delta, tau, n):
    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    iters = int(2 * tau**2 * n * rho)
    return iters

def adp_delta(k, epsilon, tau, n):
    mu = 1 / (2 * tau**2 * n)
    divisor = 2 * np.sqrt(mu * k)
    term1 = spec.erfc((epsilon - k * mu) / divisor)
    term2 = np.exp(epsilon) * spec.erfc((epsilon + k * mu) / divisor)
    return (0.5 * (term1 - term2)).sum()

def adp_iters(epsilon, delta, tau, n):
    """
    Compute the number of iterations DP-penalty can run for.

    Parameters
    ----------
    epsilon : float
    delta : float
    tau : float
        `tau` controls the tradeoff between more iterations and less noise:
        larger values add more noise, and allow more iterations.
    n : int
        The size of the dataset.

    Returns
    -------
    int
        The number of iterations DP-penalty can run for.
    """
    low_iters = zcdp_iters(epsilon, delta, tau, n)
    up_iters = max(low_iters, 1)
    while adp_delta(up_iters, epsilon, tau, n) < delta:
        up_iters *= 2
    while int(up_iters) - int(low_iters) > 1:
        new_iters = (low_iters + up_iters) / 2
        new_delta = adp_delta(new_iters, epsilon, tau, n)
        if new_delta > delta:
            up_iters = new_iters
        else:
            low_iters = new_iters

    if adp_delta(int(up_iters), epsilon, tau, n) < delta:
        return int(up_iters)
    else:
        return int(low_iters)

def dp_penalty(
        problem, theta0, epsilon, delta, params, chains,
        repeats=1, seed=4237709, verbose=True, use_adp=True, no_ll_noise=False,
        iters=None
):
    """
    Run DP-penalty

    Implemenatation of the DP-penalty algorithm. Unless the `seed` argument is given,
    a default value for the random number generator seed is used, so this function will always
    return the same results. Using using a non-default value for any of
    `repeats`, `no_ll_noise`, or `iters` will NOT provide
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
    params : PenaltyParams
        The parameters for DP-penalty.
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
    iters : int, optional
        If set, run for `iters` iterations instead of computing the number of
        iterations. Note that `epsilon` and `delta` must still be set when
        `iters` is set, but their values are not used.

    Returns
    -------
    MCMCResult or list of MCMCResult
        The results from running DP-penalty. If `repeats` is set to 1, returns
        a single object, otherwise returns an MCMCResult object for each repeat
        as a list.
    """
    ocu = params.ocu
    if params.grw:
        ocu = True # GRW requires one component updates

    dim, num_theta0 = theta0.shape
    if num_theta0 != chains * repeats:
        raise ValueError("Expected {} theta0 values but got {}".format(chains * repeats, num_theta0))

    data = problem.data
    n, data_dim = data.shape
    temp_scale = problem.temp_scale
    total_chains = chains * repeats

    tau = params.tau
    r_clip_bound = params.r_clip
    prop_sigma = params.prop_sigma

    if iters is None:
        if use_adp:
            iters = adp_iters(epsilon, delta, tau, n)
        else:
            iters = zcdp_iters(epsilon, delta, tau, n)

        iters = int(iters / chains)

    if verbose:
        print("Iterations: {}, Chains: {}".format(iters, chains))

    rng = jax.random.PRNGKey(seed)
    rng, prop_init = jax.random.split(rng, 2)

    if params.grw:
        prop_dir = jax.random.choice(prop_init, np.array([-1, 1]), (dim, chains))

    sigma = tau * np.sqrt(n)
    if no_ll_noise:
        sigma = 0

    chain = np.zeros((iters + 1, dim, total_chains))
    chain = jax.ops.index_update(chain, jax.ops.index[0, :, :], theta0)
    clipped_r = np.zeros((iters, total_chains))
    accepts = np.zeros(total_chains)

    llc = jax.vmap(problem.log_likelihood_no_sum, (1, None), 1)(chain[0, :, :], data)
    for i in range(iters):
        current = chain[i, :, :]

        rng, prop_key, noise_key, accept_key, component_key = jax.random.split(rng, 5)
        if ocu:
            update_component = jax.random.randint(component_key, (), 0, dim)
            prop = current.copy()
            if params.grw:
                noise = jax.random.normal(prop_key, shape=(total_chains,))
                magnitude = np.abs(noise * params.prop_sigma[update_component])
                new_value = current[update_component, :] + prop_dir[update_component, :] * magnitude
                prop = jax.ops.index_update(current, jax.ops.index[update_component,:], new_value)
            else:
                noise = jax.random.normal(prop_key, shape=(total_chains,))
                rand_value = noise * params.prop_sigma[update_component]
                new_value = current[update_component, :] + rand_value
                prop = jax.ops.index_update(current, jax.ops.index[update_component, :], new_value)
        else:
            prop_noise = jax.random.normal(prop_key, shape=(dim,total_chains))
            mul_noise = prop_noise * params.prop_sigma.reshape((-1, 1))
            prop = current + mul_noise

        llp = jax.vmap(problem.log_likelihood_no_sum, (1, None), 1)(prop, data)
        r = llp - llc
        d = np.sqrt(np.sum((current - prop)**2, axis=0))
        clip = d * r_clip_bound
        clipped_r = jax.ops.index_update(clipped_r, jax.ops.index[i, :], np.sum(np.abs(r) > clip, axis=0))
        r = np.clip(r, -clip, clip)

        lpp = jax.vmap(problem.log_prior, 1, 0)(prop)
        lpc = jax.vmap(problem.log_prior, 1, 0)(current)

        s = jax.random.normal(noise_key, shape=(total_chains,)) * sigma * d * 2 * r_clip_bound
        lambd = temp_scale * (np.sum(r, axis=0) + s) + lpp - lpc
        u = np.log(jax.random.uniform(accept_key, (total_chains,)))

        accept = u < lambd - 0.5 * (temp_scale * sigma * d * 2 * r_clip_bound)**2
        for j in range(total_chains):
            if accept[j]:
                chain = jax.ops.index_update(chain, jax.ops.index[i + 1, :, j], prop[:, j])
                llc = jax.ops.index_update(llc, jax.ops.index[:, j], llp[:, j])
                accepts = jax.ops.index_update(accepts, j, accepts[j] + 1)
            else:
                chain = jax.ops.index_update(chain, jax.ops.index[i + 1, :, j], current[:, j])
                if params.grw:
                    prop_dir = jax.ops.index_update(
                        prop_dir, jax.ops.index[update_component, j],
                        -prop_dir[update_component, j]
                    )
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    return result.split_results(
        chain, accepts, np.sum(clipped_r, axis=0) / n / iters, np.repeat(np.nan, total_chains),
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
    params = PenaltyParams(
        tau = 0.17,
        prop_sigma = np.repeat(0.06, 2),
        r_clip = 0.15,
        ocu = False,
        grw = False,
    )
    result = dp_penalty(problem, theta0, epsilon, delta, params, chains, repeats=repeats)

    # for res in result:
    #     print(res.compute_metrics(problem.true_posterior))
    #     print()

    metric_res = result.compute_metrics(problem.true_posterior)
    final_chain = result.get_final_chain()
    print(metric_res)

    plot_chain_summary(problem, result, theta0)

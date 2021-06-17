import metrics
import jax.numpy as np
import pandas as pd

class MCMCMetrics:
    def __init__(self, result, true_posterior):
        chains = result.get_final_chain()
        self.samples, self.dim, self.num_chains = chains.shape
        self.epsilon = result.epsilon
        self.delta = result.delta

        self.indiv_acceptance = result.accepts / result.iters
        self.indiv_clipped_r = result.clipped_r
        self.indiv_clipped_grad = result.clipped_grad

        # The denominator for each percentage is the same, so taking the
        # mean results in the aggregate acceptance of the chain, and the
        # same holds for the clip fractions
        self.agg_acceptance = np.mean(self.indiv_acceptance).item()
        self.agg_clipped_r = np.mean(self.indiv_clipped_r).item()
        self.agg_clipped_grad = np.mean(self.indiv_clipped_grad).item()

        self.r_clip_bound = getattr(result.params, "r_clip", np.nan)

        if self.samples > 1:
            self.indiv_mmd = metrics.mmd(chains, true_posterior)
            self.indiv_total_mean_error = metrics.total_mean_error(chains, true_posterior)

            agg_chain = result.get_aggregate_final_chain()
            self.agg_mmd = metrics.mmd(agg_chain, true_posterior)[0]
            self.agg_total_mean_error = metrics.total_mean_error(agg_chain, true_posterior)[0]
            self.agg_component_mean_error = metrics.component_mean_error(agg_chain, true_posterior)
            self.agg_component_mean_error = self.agg_component_mean_error.reshape((-1,))
            self.agg_component_var_error = metrics.component_var_error(agg_chain, true_posterior)
            self.agg_component_var_error = self.agg_component_var_error.reshape((-1,))

            self.r_hat = metrics.split_r_hat(chains)
            self.max_r_hat = np.max(self.r_hat).item()
        else:
            raise Exception("1 iterations chains not supported")

    def as_pandas_row(self):
        data = {
            "agg_acceptance": [self.agg_acceptance],
            "agg_component_mean_error": [self.agg_component_mean_error],
            "agg_total_mean_error": [self.agg_total_mean_error],
            "agg_component_var_error": [self.agg_component_var_error],
            "agg_mmd": [self.agg_mmd],
            "r_hat": [self.r_hat],
            "max_r_hat": [self.max_r_hat],
            "agg_clipped_r": [self.agg_clipped_r],
            "agg_clipped_grad": [self.agg_clipped_grad],
            "epsilon": [self.epsilon],
            "delta": [self.delta],
            "samples": [self.samples],
            "clip_bound": [self.r_clip_bound]
        }
        return pd.DataFrame(data)

    def __str__(self):
        metric_str = lambda heading, value: "{}: {}".format(heading, value)
        metrics = [
            metric_str("Individual Acceptance", self.indiv_acceptance),
            metric_str("Aggregate Acceptance", self.agg_acceptance),
            "",
            metric_str("Indiv Total Mean Error", self.indiv_total_mean_error),
            metric_str("Aggregate Componentwise Mean Error", self.agg_component_mean_error),
            metric_str("Aggregate Total Mean Error", self.agg_total_mean_error),
            metric_str("Aggregate Componentwise Variance Error", self.agg_component_var_error),
            "",
            metric_str("Indiv MMD", self.indiv_mmd),
            metric_str("Aggregate MMD", self.agg_mmd),
            metric_str("R-hat", self.r_hat),
            "",
            metric_str("Indiv Clipped R", self.indiv_clipped_r),
            metric_str("Aggregate Clipped R", self.agg_clipped_r),
            metric_str("Indiv Clipped Grad", self.indiv_clipped_grad),
            metric_str("Aggregate Clipped Grad", self.agg_clipped_grad),
            "",
        ]
        return "\n".join(metrics)

def split_results(chain, accepts, clipped_r, clipped_grad, repeats, epsilon, delta, params):
    """
    Split multiple repeats into separate MCMCResult objects.

    Parameters
    ----------
    chain : ndarray
        The resulting chain as an array of shape (num_samples, num_dimensions, num_chains * repeats).
    accepts : ndarray
        The number of accepts for each chain.
    clipped_r : ndarray
        The number of clipped log-likelihood ratios for each chain.
    clipped_grad : ndarray
        The number of clipped gradients for each chains.
    repeats : int
        The number of repeats.
    epsilon : float
    delta : float
    params : object
        Parameters of the algorithm that produced the result.

    Returns
    -------
    MCMCResult or list of MCMCResult
        If `repeats` is 1, return a single MCMCResult, otherwise return
        a MCMCResult for each repeat as a list.
    """
    n_iters, dim, chains = chain.shape
    chains = int(chains / repeats)

    r_val = [
        MCMCResult(
            chain[:, :, i*chains:i*chains+chains], accepts[i*chains:(i+1)*chains],
            clipped_r[i*chains:i*chains+chains], clipped_grad[i*chains:i*chains+chains],
            epsilon, delta, params
        )
        for i in range(repeats)
    ]
    if repeats == 1:
        return r_val[0]
    else:
        return r_val

class MCMCResult:
    """
    Result of an MCMC run.

    Parameters
    ----------
    chain : ndarray
        The resulting chain as an array of shape (num_samples, num_dimensions, num_chains).
    accepts : ndarray
        The number of accepts for each chain.
    clipped_r : ndarray
        The number of clipped log-likelihood ratios for each chain.
    clipped_grad : ndarray
        The number of clipped gradients for each chains.
    epsilon : float
    delta : float
    params : object
        Parameters of the algorithm that produced the result.
    """
    def __init__(self, chain, accepts, clipped_r, clipped_grad, epsilon, delta, params):
        n_iters, dim, chains = chain.shape
        self.iters = n_iters - 1
        self.accepts = accepts
        self.chain = chain
        self.clipped_r = clipped_r
        self.clipped_grad = clipped_grad
        self.epsilon = epsilon
        self.delta = delta
        self.params = params

    def compute_metrics(self, posterior):
        """
        Compute evaluation metrics compared to a reference posterior.

        Parameters
        ----------
        posterior : ndarray
            The reference posterior as an array of shape (num_samples, num_dimensions).

        Returns
        -------
        MCMCMetrics
        """
        return MCMCMetrics(self, posterior)

    def get_final_chain(self):
        """
        Return each chain with the first half removed.
        """
        burn_in = 0.5
        return self.chain[int((self.iters - 1) * (1 - burn_in)) + 1:, :, :]

    def get_aggregate_final_chain(self):
        """
        Return the aggregate sample from all chains with the first half removed.
        """
        chains = self.get_final_chain()
        agg_chain = np.stack((np.concatenate(np.transpose(chains, axes=(2, 0, 1)), axis=0),), axis=2)
        return agg_chain

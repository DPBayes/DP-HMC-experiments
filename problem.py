import jax 
import jax.numpy as np
import metrics

class Problem:
    def __init__(
            self, log_likelihood_per_sample, log_prior, data, temp_scale,
            theta0, gen_true_posterior, plot_density_func, true_mean=None
    ):
        """Log likelihood per sample should have signature (theta, sample) -> float"""
        self.log_likelihood_per_sample = log_likelihood_per_sample
        self.log_prior = log_prior
        self.data = data
        self.temp_scale = temp_scale
        if gen_true_posterior is not None:
            self.gen_true_posterior = lambda n, key=None: gen_true_posterior(
                n, data, temp_scale, key
            )
            self.true_posterior = gen_true_posterior(1000, data, temp_scale)
        else:
            self.gen_true_posterior = None
            self.true_posterior = None
        self.true_mean = true_mean if true_mean is not None else np.mean(self.true_posterior, axis=0)
        self.theta0 = theta0
        self.dim = theta0.size
        self.plot_density_func = plot_density_func

        self.log_likelihood_no_sum = jax.jit(jax.vmap(self.log_likelihood_per_sample, in_axes=(None, 0)))
        self.log_likelihood = jax.jit(lambda theta, X: np.sum(self.log_likelihood_no_sum(theta, X)))

        self.log_likelihood_grads = jax.jit(
            jax.vmap(jax.grad(self.log_likelihood_per_sample, 0), in_axes=(None, 0))
        )
        self.log_likelihood_grad_clipped = clip_grad_fun(self.log_likelihood_grads)

        self.log_prior_grad = jax.jit(jax.grad(self.log_prior))

    def plot_density(self, ax):
        self.plot_density_func(self, ax)

    def get_start_point(self, i):
        key = jax.random.PRNGKey(4236482 + i)
        if self.true_posterior is not None:
            mul = self.true_posterior.std(axis=0).mean() * 1
        else:
            mul = 1
        return self.theta0 + jax.random.normal(key, shape=(self.dim,)) * mul

def clip_grad_fun(grad_fun):
    def return_fun(clip, *args):
        grads = grad_fun(*args)
        grads, did_clip = jax.vmap(clip_norm, in_axes=(0, None))(grads, clip)
        clipped_grad = np.sum(did_clip)
        return (np.sum(grads, axis=0), clipped_grad)
    return jax.jit(return_fun)

@jax.jit
def clip_norm(x, bound):
    norm = np.sqrt(np.sum(x**2))
    clipped_norm = np.min(np.array((norm, bound)))
    return (x / norm * clipped_norm, norm > bound)

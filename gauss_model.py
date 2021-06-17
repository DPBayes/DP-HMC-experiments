import jax.numpy as np
import jax.scipy.stats as stats
import jax.scipy.linalg as linalg
import jax
import problem
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

@jax.jit
def log_prior(theta, sigma0):
    return np.sum(stats.norm.logpdf(theta, scale=sigma0))

@jax.jit
def log_likelihood_per_sample(theta, x, cov):
    return stats.multivariate_normal.logpdf(theta, mean=x, cov=cov)

@jax.jit
def log_likelihood_per_sample_fast(theta, x, chol):
    dim = theta.size
    # const = -0.5 * dim * np.log(2 * np.pi)
    # logdet = -np.sum(np.log(chol.diagonal()))
    z = linalg.solve_triangular(chol, x - theta, lower=True)
    return -0.5 * np.sum(z**2) #+ const + logdet

class GaussModel:
    def __init__(self, dim, gamma_shape=0.5):
        self.dim = dim
        self.sigma0 = 100
        self.tau0 = 1 / self.sigma0**2

        rng = jax.random.PRNGKey(43235667)
        rng, eigval_key, eigvec_key = jax.random.split(rng, 3)
        d = np.diag(jax.random.gamma(eigval_key, gamma_shape, shape=(dim,)))
        mat = jax.random.uniform(eigvec_key, shape=(dim, dim))
        q, r = np.linalg.qr(mat)
        self.cov = q @ d @ np.linalg.inv(q)

        self.L = np.linalg.cholesky(self.cov)
        self.true_mean = np.hstack((np.array((0, 3)), np.zeros(dim - 2)))

    def log_prior(self, theta):
        return log_prior(theta, self.sigma0)

    def log_likelihood_per_sample(self, theta, x):
        return log_likelihood_per_sample_fast(theta, x, self.L)

    def generate_data(self, n):
        key = jax.random.PRNGKey(4627290)
        return jax.random.multivariate_normal(key, self.true_mean, self.cov, (n,))

    def compute_posterior_params(self, data):
        n, d = data.shape
        # See Bayesian Data Analysis, section 3.5
        # or https://stats.stackexchange.com/q/28744
        sigma0_mat = np.eye(self.dim) * self.sigma0
        term1 = sigma0_mat @ np.linalg.inv(sigma0_mat + self.cov / n)
        mu_post = term1 @ np.mean(data, axis=0).reshape((-1, 1))
        cov_post = term1 @ self.cov / n
        return (mu_post.reshape(-1,), cov_post)

    def generate_true_posterior(self, samples, data, temp_scale=1, key=None):

        if temp_scale != 1:
            raise Exception("GaussModel does not support sampling tempered posteriors")
        if key is None:
            key = jax.random.PRNGKey(434927)
        mu_post, cov_post = self.compute_posterior_params(data)

        return jax.random.multivariate_normal(key, mu_post, cov_post, (samples,))

    def plot_posterior(self, ax, data):
        mu_post, cov_post = self.compute_posterior_params(data)

        xs = np.linspace(-0.01, 0.01, 1000) + mu_post[0]
        ys = np.linspace(-0.01, 0.01, 1000) + mu_post[1]
        X, Y = np.meshgrid(xs, ys)
        #TODO: figure out why this plots a mirrored version of the posterior
        Z = jax.vmap(jax.vmap(lambda x: stats.multivariate_normal.pdf(x, mean=mu_post, cov=cov_post),
            in_axes=1
        ), in_axes=2)(np.stack((X, Y)))
        ax.contour(X, Y, Z, levels=20)

    def plot_marginals(self, axes, data):
        mu_post, cov_post = self.compute_posterior_params(data)
        xs = np.linspace(-0.01, 0.01, 1000) + mu_post[0]
        ys = np.linspace(-0.01, 0.01, 1000) + mu_post[1]
        axes[0].plot(xs, stats.norm.pdf(xs, loc=mu_post[0], scale=cov_post[0, 0]**0.5))
        axes[1].plot(ys, stats.norm.pdf(ys, loc=mu_post[1], scale=cov_post[1, 1]**0.5))

def get_problem(dim, n, gamma_shape=0.5):
    model = GaussModel(dim, gamma_shape)
    data = model.generate_data(n)
    prob = problem.Problem(
        model.log_likelihood_per_sample, model.log_prior,
        data, 1, model.true_mean, model.generate_true_posterior,
        lambda problem, ax: model.plot_posterior(ax, problem.data)
    )
    return prob

if __name__ == "__main__":
    dim = 2
    n = 100000

    # prob = get_problem(dim, n)
    # data = prob.data
    # posterior = prob.true_posterior

    model = GaussModel(dim)

    # n = 1000
    # mu = np.zeros(2)
    # # cov = model.cov
    # cov = np.array(((1, 1.22), (1.22, 1.5)))
    # print(cov)

    # key = jax.random.PRNGKey(4237842)
    # samples = jax.random.multivariate_normal(key, mu, cov, (n,))
    # sample_cov = np.cov(samples, rowvar=False)
    # print(sample_cov)

    # scale = 300
    # xs = np.linspace(-0.01, 0.01, 1000) * scale
    # ys = np.linspace(-0.01, 0.01, 1000) * scale
    # X, Y = np.meshgrid(xs, ys)
    # Z = jax.vmap(jax.vmap(
    #     lambda x: stats.multivariate_normal.pdf(x, mean=mu, cov=sample_cov), in_axes=1
    # ), in_axes=2)(np.stack((X, Y)))
    # Z2 = jax.vmap(jax.vmap(
    #     lambda x: stats.multivariate_normal.pdf(x, mean=mu, cov=cov), in_axes=1
    # ), in_axes=2)(np.stack((Y, X)))
    # plt.contour(X, Y, Z, levels=20)
    # # plt.contour(X, Y, Z2, levels=20)
    # plt.scatter(samples[:, 0], samples[:, 1])
    # plt.plot(xs, xs, color="red")
    # plt.show()

    data = model.generate_data(100000)
    posterior = model.generate_true_posterior(40000, data)

    mean, cov = model.compute_posterior_params(data)
    print("Posterior covariance condition number: {:.3e}".format(np.linalg.cond(cov)))
    print("Model covariance condition number: {:.3e}".format(np.linalg.cond(model.cov)))

    fig, ax = plt.subplots()
    if dim == 2:
        model.plot_posterior(ax, data)
        # prob.plot_density(ax)

    plt.scatter(posterior[:, 0], posterior[:, 1])

    fig, ax = plt.subplots(2)
    model.plot_marginals(ax, data)
    ax[0].hist(posterior[:, 0], density=True, bins=50)
    ax[1].hist(posterior[:, 1], density=True, bins=50)
    plt.show()

    for i in range(dim):
        print(np.quantile(posterior[:, i], np.array([0.01, 0.99])))

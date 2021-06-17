import jax.numpy as np
import jax
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import problem
jax.config.update("jax_enable_x64", True)

@jax.jit
def log_prior(theta, a, b, m, tau0):
    return (
        stats.norm.logpdf(theta[0], scale=1 / np.sqrt(tau0))
        + stats.norm.logpdf(
            theta[1] + a * (theta[0] - m)**2 + b, scale=1 / np.sqrt(tau0)
        )
        + np.sum(stats.norm.logpdf(theta[2:], scale=1 / np.sqrt(tau0)))
    )

@jax.jit
def log_likelihood_per_sample(theta, x, a, b, m, sigma1, sigma2, sigma3, dim):
    logc1 = -np.log(sigma1 * np.sqrt(2 * np.pi))
    logc2 = -np.log(sigma2 * np.sqrt(2 * np.pi))
    logc3 = -np.log(sigma3 * np.sqrt(2 * np.pi))

    theta1 = theta[0]
    theta2 = theta[1]
    theta_rest = theta[2:]
    x1 = x[0]
    x2 = x[1]
    xrest = x[2:]
    term1 = -0.5 * (x1 - theta1)**2 / sigma1**2 + logc1
    term2 = -0.5 * (x2 - (theta2 + a * (theta1 - m)**2) + b)**2 / sigma2**2 + logc2
    term3 = -0.5 * np.sum((xrest - theta_rest)**2) / sigma3**2 + logc3 * (dim - 2)
    return term1 + term2 + term3

class BananaModel:
    def __init__(self, dim=2, a=20, tau0=0.001, tau1=0.05, tau2=0.04, tau3=1):
        self.tau0 = tau0
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3

        self.sigma1 = 1 / np.sqrt(self.tau1)
        self.sigma2 = 1 / np.sqrt(self.tau2)
        self.sigma3 = 1 / np.sqrt(self.tau3)

        self.a = a
        self.b = 0.0
        self.m = 0.0
        self.dim = dim

    def generate_test_data(self, n=100000):
        theta1 = 0
        theta2 = 3
        theta_rest = np.zeros(self.dim - 2)

        key = jax.random.PRNGKey(43247)
        keys = jax.random.split(key, 3)

        x1s = jax.random.normal(keys[0], (n,1)) * self.sigma1 + theta1
        x2s = jax.random.normal(keys[1], (n,1)) * self.sigma2 + theta2 + self.a * (theta1 - self.m)**2 + self.b
        xrest = jax.random.normal(keys[2], (n, self.dim - 2)) * self.sigma3 + theta_rest
        return np.hstack((x1s, x2s, xrest))

    def log_likelihood_per_sample(self, theta, data):
        return log_likelihood_per_sample(
            theta, data, self.a, self.b, self.m, self.sigma1, self.sigma2, self.sigma3, self.dim
        )

    def log_prior(self, theta):
        return log_prior(theta, self.a, self.b, self.m, self.tau0)

    def get_problem(self, n=100000, n0=None):
        data = self.generate_test_data(n)
        n, d = data.shape
        if n0 is None:
            temp_scale = 1
        else:
            temp_scale = n0 / n
        # true_posterior = self.generate_posterior_samples(1000, data, temp_scale)
        theta0 = np.zeros(self.dim)
        theta0 = jax.ops.index_update(theta0, 1, 3)
        return problem.Problem(
            self.log_likelihood_per_sample, self.log_prior, data,
            temp_scale, theta0, self.generate_posterior_samples,
            lambda problem, ax: self.plot_posterior(problem.data, ax, problem.temp_scale)
        )

    def banana_density(self, theta1, theta2, mu1, mu2, sigma1, sigma2, a, b, m):
        return (
            stats.norm.pdf(theta1, loc=mu1, scale=sigma1)
            * stats.norm.pdf(theta2 + a * (theta1 - m)**2 + b, loc=mu2, scale=sigma2)
        )

    def banana_g2(self, x1, x2, m):
        return x2 - self.a * (x1 - m)**2 - self.b

    def compute_posterior_params(self, X, T):
        n = X.shape[0]
        mu1 = (T * n * self.tau1 * X[..., 0].mean()) / (T * n * self.tau1 + self.tau0)
        mu2 = (T * n * self.tau2 * X[..., 1].mean()) / (T * n * self.tau2 + self.tau0)
        murest = (T * n * self.tau3 * X[...,2:].mean(axis=0)) / (T * n * self.tau3 + self.tau0)
        sigma1_p = 1 / np.sqrt(T * n * self.tau1 + self.tau0)
        sigma2_p = 1 / np.sqrt(T * n * self.tau2 + self.tau0)
        sigma3_p = 1 / np.sqrt(T * n * self.tau3 + self.tau0)
        return (mu1, mu2, murest, sigma1_p, sigma2_p, sigma3_p)

    def generate_posterior_samples(self, n, X, T, key=None):
        mu1, mu2, murest, sigma1_p, sigma2_p, sigma3_p = self.compute_posterior_params(X, T)

        if key is None:
            key = jax.random.PRNGKey(56437)
        keys = jax.random.split(key, 3)

        s1s = jax.random.normal(keys[0], (n,1)) * sigma1_p + mu1
        s2s = self.banana_g2(s1s, jax.random.normal(keys[1], (n,1)) * sigma2_p + mu2, self.m)
        srest = jax.random.normal(keys[2], (n, self.dim - 2)) * sigma3_p + murest
        return np.hstack((s1s, s2s, srest))

    def scatterplot_posterior(self, X, ax, T):
        post = self.generate_posterior_samples(1000, X, T)
        ax.scatter(post[:, 0], post[:, 1], alpha=0.5)

    def plot_posterior(self, X, ax, T):
        mu1, mu2, _, sigma1_p, sigma2_p, __ = self.compute_posterior_params(X, T)
        samples = self.generate_posterior_samples(40, X, T)
        spread_samples = samples.mean(axis=0) + (samples - samples.mean(axis=0)) * 2.5
        mi = np.min(spread_samples, axis=0)
        ma = np.max(spread_samples, axis=0)
        xs = np.linspace(mi[0], ma[0], 1000)
        ys = np.linspace(mi[1], ma[1], 1000)
        X, Y = np.meshgrid(xs, ys)
        Z = self.banana_density(X, Y, mu1, mu2, sigma1_p, sigma2_p, self.a, self.b, self.m)
        ax.contour(X, Y, Z)

def get_problem(dim=2, n0=None, a=20, n=100000, tau1=0.05, tau2=0.04, tau3=1, tau0=0.001):
    return BananaModel(dim, a, tau0, tau1, tau2).get_problem(n=n, n0=n0)

if __name__ == "__main__":
    T = 1000 / 100000

    banana = BananaModel(a=100, tau1=0.0025, tau2=0.0025)
    X = banana.generate_test_data()

    fig, ax = plt.subplots()
    banana.scatterplot_posterior(X, ax, 1)
    banana.plot_posterior(X, ax, 1)
    plt.show()

    # fig, ax = plt.subplots()
    # banana.scatterplot_posterior(X, ax, T)
    # banana.plot_posterior(X, ax, T)
    # plt.show()

    post = banana.generate_posterior_samples(10000, X, 1)
    print(np.quantile(post[:, 0], np.array([0.01, 0.99])))
    print(np.quantile(post[:, 1], np.array([0.01, 0.99])))
    for i in range(banana.dim - 2):
        print(np.quantile(post[:, i + 2], np.array((0.01, 0.99))))

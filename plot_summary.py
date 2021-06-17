import matplotlib.pyplot as plt

def plot_chain_summary(problem, result, theta0, filename=None):
    if filename is not None:
        tracefile = filename + "_trace.pdf"
        posteriorfile = filename + "_posterior.pdf"
    else:
        tracefile = None
        posteriorfile = None

    plot_trace(result, tracefile)
    plot_2d_posterior(problem, result, theta0, posteriorfile)

def show_or_save(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def plot_trace(result, filename=None):
    n_samples, dim, chains = result.chain.shape
    fig, axes = plt.subplots(min(dim, 10), squeeze=False)
    for i in range(min(dim, 10)):
        for j in range(chains):
            axes[i, 0].plot(result.chain[:, i, j])
    show_or_save(filename)

def plot_2d_posterior(problem, result, theta0, filename):
    final_chain = result.get_final_chain()
    n_samples, dim, chains = final_chain.shape
    posterior = problem.true_posterior
    fig, ax = plt.subplots()
    ax.scatter(posterior[:, 0], posterior[:, 1], color="gray")
    for i in range(chains):
        ax.plot(final_chain[:, 0, i], final_chain[:, 1, i], linestyle='', marker='.')
    plt.plot(theta0[0], theta0[1], color="red", linestyle="", marker=".")
    show_or_save(filename)

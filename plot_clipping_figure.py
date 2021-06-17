import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str)
parser.add_argument("experiment")
parser.add_argument("input")
args = parser.parse_args()

numpy_converter = lambda s: np.fromstring(s[1:-1], sep=" ")
df = pd.read_csv(
    args.input, converters={
        "agg_component_mean_error": numpy_converter,
        "agg_component_var_error": numpy_converter,
        "r_hat": numpy_converter,
    }
)

column_name_replacements = {
    "agg_mmd": "MMD",
    "agg_total_mean_error": "Mean Error",
    "agg_component_mean_error": "Mean Error",
    "agg_component_var_error": "Variance Error",
    "max_r_hat": "R-hat",
    "agg_acceptance": "Acceptance",
    "agg_clipped_r": "Clipped Log-likelihood ratios",
    "agg_clipped_grad": "Clipped Gradients",
    "clip_bound": "Clip Bound",
}
algorithm_name_replacements = {
    "dp-hmc": "HMC",
    "dp-penalty": "RWMH"
}
figure_titles = {
    "banana": "Banana Experiment",
    "gauss": "Gaussian Experiment"
}
y_limits = {
    "banana": 0.5,
    "gauss": 0.6
}
df["algorithm"] = df["algorithm"].map(algorithm_name_replacements)

def plot_scalar(x, y, df, ax, plot_fun, legend):
    plot_fun(x=x, y=y, data=df, hue="algorithm", ax=ax)
    ax.set_ylabel(column_name_replacements[y])
    ax.set_xlabel(column_name_replacements[x])
    if y == "agg_mmd":
        ax.set_ylim((0, y_limits[args.experiment]))
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.get_legend().remove()

fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
fig.suptitle(figure_titles[args.experiment])
plot_scalar("clip_bound", "agg_mmd", df, axes[0], sns.boxplot, False)
plot_scalar("agg_clipped_r", "agg_mmd", df, axes[1], sns.scatterplot, True)
# plot_scalar("clip_bound", "max_r_hat", df, axes[2], sns.stripplot, True)
plt.tight_layout()
plt.savefig(args.output)
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
fig.suptitle(figure_titles[args.experiment])
plot_scalar("clip_bound", "max_r_hat", df, axes[0], sns.stripplot, False)
plot_scalar("clip_bound", "agg_clipped_grad", df, axes[1], sns.stripplot, True)
plt.tight_layout()
plt.savefig(args.output[:-4] + "-diagnostic.pdf")
# plt.show()

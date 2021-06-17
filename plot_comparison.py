import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str)
parser.add_argument("output_full", type=str)
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
}
figure_titles = {
    "banana": "Banana Experiment",
    "gauss": "Gaussian Experiment"
}

def plot_scalar(y, df, ax, legend=True):
    sns.boxplot(x="epsilon", y=y, data=df, hue="algorithm", ax=ax)
    ax.set_ylabel(column_name_replacements[y])
    ax.set_xlabel("Epsilon")
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.get_legend().remove()

fig, axes = plt.subplots(3, 2, figsize=(10, 2.5 * 3))
fig.suptitle(figure_titles[args.experiment])
plot_ys_full = ["agg_mmd", "agg_total_mean_error", "max_r_hat", "agg_acceptance",
           "agg_clipped_r", "agg_clipped_grad"]
for i, y in enumerate(plot_ys_full):
    plot_scalar(y, df, axes[i % 3, i // 3])

plt.tight_layout()
plt.savefig(args.output_full)

fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
fig.suptitle(figure_titles[args.experiment])
plot_ys = ["agg_mmd", "agg_total_mean_error"]
for i, y in enumerate(plot_ys):
    plot_scalar(y, df, axes[i], i==1)

plt.tight_layout()
plt.savefig(args.output)

# def plot_vector(y, df, axes):
#     dim = df[y][0].size
#     for i in range(dim):
#         col_name = "{} {}".format(column_name_replacements[y], i + 1)
#         updated_df = df.assign(**{col_name: df[y].map(lambda x: x[i])})
#         sns.stripplot(x="epsilon", y=col_name, data=updated_df, hue="algorithm", ax=axes[i])
#         axes[i].axhline(0, color="gray")
#         mi, ma = axes[i].get_ylim()
#         new_lim = max(np.abs(mi), np.abs(ma))
#         axes[i].set_ylim(-new_lim, new_lim)

# fig, axes = plt.subplots(2, 2, figsize=(6, 6))
# plot_vector("agg_component_mean_error", df, axes[0, :])
# plot_vector("agg_component_var_error", df, axes[1, :])
# plt.tight_layout()
# plt.savefig("../latex/figures/componentwise_comparison_figure.pdf")
# # plt.show()

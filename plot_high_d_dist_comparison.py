import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import experiments

plt.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument("--chains", nargs="+")
parser.add_argument("--result_tables", nargs="+")
parser.add_argument("--experiment")
parser.add_argument("--algorithms", nargs="+")
parser.add_argument("--epsilon", type=float)
parser.add_argument("--output")
args = parser.parse_args()

def plot_one_comparison(chain_file, result_table_file, experiment, algorithm, epsilon, axes, legend):

    with open(chain_file, "rb") as file:
        results = pickle.load(file)

    chains = [result.get_aggregate_final_chain()[:, :, 0] for result in results]
    res_df = pd.read_csv(result_table_file)
    res_df.sort_values("agg_mmd", inplace=True)
    best_index = res_df.iloc[0]["repeat_index"]
    worst_index = res_df.iloc[-1]["repeat_index"]
    median_index = res_df.iloc[res_df.shape[0] // 2]["repeat_index"]

    problem = experiments.experiments[experiment]
    posterior = problem.true_posterior
    n, dim = posterior.shape

    def posterior_to_df(posterior, repeat_index):
        df = pd.DataFrame(posterior, columns=list(map(str, range(dim))))
        df["Index"] = repeat_index
        return df

    df = pd.concat(
        [posterior_to_df(posterior, "true posterior")]
        + [posterior_to_df(chains[best_index], "best sample")]
        + [posterior_to_df(chains[median_index], "median sample")]
        + [posterior_to_df(chains[worst_index], "worst sample")]
    )

    for i in range(dim):
        sns.kdeplot(
            data=df, x=str(i), hue="Index", common_norm=False, fill=True,
            ax=axes[i], legend=legend
        )
        axes[i].set_xlabel(["x", "y"][i])

    axes[0].set_title(algorithm)

    # representative_2d = chains[median_index]
    # axes[2].scatter(posterior[:, 0], posterior[:, 1], alpha=0.2)
    # axes[2].scatter(representative_2d[:, 0], representative_2d[:, 1], alpha=0.2)

    ax = sns.kdeplot(data=df, x="0", y="1", hue="Index", common_norm=False, ax=axes[2], legend=legend, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if legend:
        for ax in axes:
            # Hack to set legend with sns.kdeplot
            # see https://github.com/mwaskom/seaborn/issues/2280
            old_leg = ax.legend_
            handles = old_leg.legendHandles
            labels = [t.get_text() for t in old_leg.get_texts()]
            # title = old_leg.get_title().get_text()
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1))
            # ax.legend(handles, labels, loc="lower left", title=title)

fig, axes = plt.subplots(3, len(args.chains), figsize=(10, 5.5))
for i, (chain, result_table, algorithm) in enumerate(zip(args.chains, args.result_tables, args.algorithms)):
    plot_one_comparison(chain, result_table, args.experiment, algorithm, args.epsilon, axes[:,i], i==3)

plt.tight_layout()
plt.savefig(args.output)
# plt.show()

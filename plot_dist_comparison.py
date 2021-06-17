import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import experiments

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("experiment")
parser.add_argument("algorithm")
parser.add_argument("epsilon", type=float)
parser.add_argument("output")
args = parser.parse_args()

with open(args.input, "rb") as file:
    results = pickle.load(file)

chains = [result.get_aggregate_final_chain()[:, :, 0] for result in results]

problem = experiments.experiments[args.experiment]
posterior = problem.true_posterior

def posterior_to_df(posterior, repeat_index):
    df = pd.DataFrame(posterior, columns=["x", "y"])
    df["Index"] = repeat_index
    return df

df = pd.concat(
    [posterior_to_df(posterior, "true posterior")]
    + [posterior_to_df(chain, i) for i, chain in enumerate(chains)]
)

grid = sns.JointGrid(data=df, x="x", y="y", hue="Index")
grid.plot_joint(sns.kdeplot, alpha=1.0)
grid.plot_marginals(sns.kdeplot, fill=True, common_norm=False)
plt.savefig(args.output)
# plt.show()

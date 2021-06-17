import pickle
import argparse
import pandas as pd
import result
import experiments

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str)
parser.add_argument("algorithm", type=str)
parser.add_argument("chain", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

with open(args.chain, "rb") as file:
    results = pickle.load(file)

problem = experiments.experiments[args.experiment]
metrics = [result.compute_metrics(problem.true_posterior) for result in results]

df = pd.concat([
    metric.as_pandas_row().assign(experiment=args.experiment, algorithm=args.algorithm, repeat_index=i)
    for (i, metric) in enumerate(metrics)
])
df.to_csv(args.output, header=True, index=False)

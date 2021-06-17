import jax.numpy as np
import argparse
import pickle
import experiments
import params
import dp_penalty
import hmc
import sgld

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str)
parser.add_argument("algorithm", type=str)
parser.add_argument("params", type=str)
parser.add_argument("epsilon", type=float)
parser.add_argument("repeats", type=int)
parser.add_argument("output", type=str)
args = parser.parse_args()

problem = experiments.experiments[args.experiment]
n, data_dim = problem.data.shape

seed_base = args.algorithm + args.experiment + str(args.epsilon)
seed = int(seed_base.encode("utf8").hex(), 16) % 2**32

algorithms = {
    "dp-penalty": dp_penalty.dp_penalty,
    "dp-hmc": hmc.hmc,
    "dp-sgld": sgld.sgld,
    "dp-sgnht": sgld.sgnht,
}
par = params.__dict__[args.params].params(epsilon=args.epsilon)
chains = 4
theta0 = np.vstack([problem.get_start_point(i) for i in range(chains * args.repeats)]).transpose()
results = algorithms[args.algorithm](
    problem, theta0, args.epsilon, 0.1 / n, par, chains,
    args.repeats, verbose=False, seed=seed
)

with open(args.output, "wb") as file:
    pickle.dump(results, file)

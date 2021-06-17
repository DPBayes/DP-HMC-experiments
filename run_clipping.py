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
parser.add_argument("bound", type=float)
parser.add_argument("repeats", type=int)
parser.add_argument("output", type=str)
args = parser.parse_args()

problem = experiments.experiments[args.experiment]
n, data_dim = problem.data.shape

seed_base = args.algorithm + args.experiment + str(args.bound)
seed = int(seed_base.encode("utf8").hex(), 16) % 2**32

params = {
    "banana": {
        "dp-hmc": (hmc.HMCParams(
            tau = 0.10,
            tau_g = 0.55,
            eta = 0.01,
            L = 100,
            mass = 1,
            r_clip= 10.0,
            grad_clip = 10
        ), 1000),
        "dp-penalty": (dp_penalty.PenaltyParams(
            tau = 0.17,
            prop_sigma = np.repeat(0.3, 2),
            r_clip = 1.0,
            ocu = False,
            grw = False,
        ), 12000)
    },
    "gauss":{
        "dp-hmc": (hmc.HMCParams(
            tau = 0.1,
            tau_g = 0.25,
            eta = 0.0005,
            L = 12,
            mass = 1,
            r_clip= 60,
            grad_clip = 100
        ), 1000),
        "dp-penalty": (dp_penalty.PenaltyParams(
            tau = 0.05,
            prop_sigma = np.repeat(0.0007, 10),
            r_clip = 100,
            ocu = False,
            grw = False,
        ), 25000)
    }
}
par = params[args.experiment][args.algorithm][0]
iters = params[args.experiment][args.algorithm][1]
par.r_clip = args.bound

chains = 4
theta0 = np.vstack([problem.get_start_point(i) for i in range(chains * args.repeats)]).transpose()

if args.algorithm == "dp-hmc":
    results = hmc.hmc(
        problem, theta0, None, None, par, chains, args.repeats, verbose=False,
        seed=seed, no_ll_noise=True, no_grad_noise=True, iters=iters
    )

elif args.algorithm == "dp-penalty":
    results = dp_penalty.dp_penalty(
        problem, theta0, None, None, par, chains, args.repeats, verbose=False,
        seed=seed, no_ll_noise=True, iters=iters
    )

with open(args.output, "wb") as file:
    pickle.dump(results, file)

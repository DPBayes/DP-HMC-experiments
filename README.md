# Differentially Private Hamiltonian Monte Carlo

## Installing dependencies

We use [Anaconda](https://www.anaconda.com/), specifically
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
to manage our dependencies. Once you have Anaconda installed,
run
``` sh
conda env create -f environment.yml
```
to create the Anaconda environment named `dp-hmc-env` with the dependencies. Next, activate
the environment with
``` sh
conda activate dp-hmc-env
```

## Running the Experiments

The experiments are run with 
[Snakemake](https://snakemake.readthedocs.io/en/stable/),
which is installed as part of the dependencies.

The command
``` sh
snakemake -j 4
```
runs all of the experiments on the CPU and produces the all of the figures to
`../latex/figures`. Intermediate results are placed in `results/`. 
The number after `-j` sets the number of concurrent jobs that are run, and 
can be set according to the number of available CPU cores.


## Using the GPU

To run the experiments on the GPU, first reinstall JAX using
[the installations instructions for GPU](https://github.com/google/jax#installation)
into the `dp-hmc-env` Anaconda environment. The command to run the experiments
is the same
``` sh
snakemake -j 1
```
but you must use `-j 1`, as JAX 
allocates 90% of GPU memory per job, so running more than one job 
concurrently will crash any extra jobs. 

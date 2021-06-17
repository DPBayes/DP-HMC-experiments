epsilons = [2, 4, 6, 8, 10, 12, 15]
experiments = ["banana", "gauss"]
algorithms = ["dp-penalty", "dp-hmc", "dp-sgld", "dp-sgnht"]
num_repeats = 10

result_dir = "results/"
fig_dir = "figures/"

rule all:
    input:
        dist_comparison = fig_dir + "dist_comparison_banana.pdf",
        comparison_figure = expand(
            fig_dir + "{experiment}_comparison_figure.pdf", experiment=experiments
        ),
        comparison_figure_full = expand(
            fig_dir + "{experiment}_comparison_figure_full.pdf", experiment=experiments
        ),
        clip_fig = expand(fig_dir + "clip-{experiment}_figure.pdf", experiment=experiments)

banana_clip_bounds = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 10]
gauss_clip_bounds = [0.1, 0.2, 1, 5, 10, 15, 20, 50, 100]

rule run_clip_chain:
    input:
        py = "run_clipping.py"
    output: result_dir + "chains/clip-{experiment}_{algo}_{bound}.p"
    shell: "python {input.py} {wildcards.experiment} {wildcards.algo} {wildcards.bound} {num_repeats} {output}"

rule run_chain:
    params:
        py = "run_chain.py",
    input:
        par = "params/{algo}_{experiment}.py"
    output: result_dir + "chains/{experiment}_{algo}_{eps}.p"
    shell: "python {params.py} {wildcards.experiment} {wildcards.algo} {wildcards.algo}_{wildcards.experiment} {wildcards.eps} {num_repeats} {output}"

rule run_result_table:
    input:
        py = "compute_run_result_table.py",
        chain = result_dir + "chains/{experiment}_{algo}_{eps}.p"
    output: result_dir + "run_tables/{experiment}_{algo}_{eps}.csv"
    shell: "JAX_PLATFORM_NAME=cpu python {input.py} {wildcards.experiment} {wildcards.algo} {input.chain} {output}"

rule experiment_result_table:
    input:
        results = [
            result_dir + "run_tables/{card}_{algo}_{eps}.csv".format(card="{experiment}", algo=algo, eps=eps)
            for algo in algorithms for eps in epsilons
        ]
    output: result_dir + "{experiment}_result_table.csv"
    shell: "{{ head -n 1 {input.results[0]}; tail -q -n +2 {input.results}; }} > {output}"

rule clip_result_table_banana:
    input:
        results = [
            result_dir + "run_tables/clip-banana_{algo}_{bound}.csv".format(algo=algo, bound=bound)
            for algo in ["dp-hmc", "dp-penalty"] for bound in banana_clip_bounds
        ]
    output: result_dir + "clip-banana_result_table.csv"
    shell: "{{ head -n 1 {input.results[0]}; tail -q -n +2 {input.results}; }} > {output}"

rule clip_result_table_gauss:
    input:
        results = [
            result_dir + "run_tables/clip-gauss_{algo}_{bound}.csv".format(algo=algo, bound=bound)
            for algo in ["dp-hmc", "dp-penalty"] for bound in gauss_clip_bounds
        ]
    output: result_dir + "clip-gauss_result_table.csv"
    shell: "{{ head -n 1 {input.results[0]}; tail -q -n +2 {input.results}; }} > {output}"

rule comparison_figure:
    input:
        py = "plot_comparison.py",
        csv = result_dir + "{experiment}_result_table.csv"
    output:
        fig = fig_dir + "{experiment}_comparison_figure.pdf",
        full_fig = fig_dir + "{experiment}_comparison_figure_full.pdf"
    shell: "JAX_PLATFORM_NAME=cpu python {input.py} {output.fig} {output.full_fig} {wildcards.experiment} {input.csv}"

rule clipping_figure:
    input:
        py = "plot_clipping_figure.py",
        csv = result_dir + "clip-{experiment}_result_table.csv"
    output: fig_dir + "clip-{experiment}_figure.pdf",
    shell: "JAX_PLATFORM_NAME=cpu python {input.py} {output} {wildcards.experiment} {input.csv}"

max_eps = max(epsilons)
dist_comp_shell_str = (
    "JAX_PLATFORM_NAME=cpu "
    "python {input.py} --chains {input.chains} --result_tables {input.result_tables} "
    "--experiment=banana --algorithms {algorithms} "
    "--epsilon={max_eps} --output={output} "
)
rule dist_comparison_figure:
    input:
        py = "plot_high_d_dist_comparison.py",
        chains = expand(
            result_dir + "chains/{experiment}_{algo}_{eps}.p",
            experiment=["banana"], algo=algorithms, eps=max(epsilons)
        ),
        result_tables = expand(
            result_dir + "run_tables/{experiment}_{algo}_{eps}.csv",
            experiment=["banana"], algo=algorithms, eps=max(epsilons)
        )
    output: fig_dir + "dist_comparison_banana.pdf"
    shell: dist_comp_shell_str

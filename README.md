## dXPP: Penalty-Based Differentiation Through Black-Box QP Solvers

This repository introduces the **dXPP** framework for differentiation through quadratic programming (QP) solvers. The dXPP approach is penalty-based: it employs any black-box QP solver in the forward pass, and, in the backward pass, performs differentiation through a penalty-based reformulation of the original problem.




Linghu, Yuxuan, Zhiyuan Liu, and Qi Deng. **"A Penalty Approach for Differentiation Through Black-Box Quadratic Programming Solvers."** arXiv preprint arXiv:2602.14154, 2026.

- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Citing](#citing)

## Installation

Create a Conda environment with Python 3.9, then install dependencies in order: core packages (PyTorch CPU, SciPy, NumPy, QP solvers), optional QP solver backends, PyTorch Geometric, differentiable QP layers (optnet, qpth, cvxpylayers, proxsuite), and extras for running examples and experiments.

```bash
conda create -y --name dXPP python=3.9
conda activate dXPP
pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html scipy numpy qpsolvers
pip install clarabel cvxopt daqp ecos gurobipy highspy mosek osqp piqp proxsuite qpalm quadprog scs
pip install qdldl pypardiso
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install optnet qpth cvxpylayers proxsuite
pip install matplotlib tensorboard pandas
pip install setproctitle
pip install libigl polyscope shapely robust_laplacian torchvision==0.18
conda install -c conda-forge ffmpeg
```

**Quick Checks.** Run a minimal example to verify the solver and autograd pipeline:

```bash
python example_script.py
```

## Experiments

### Part 1: Gradient Comparison

Run the following script to verify gradient accuracy:

```bash
python layer/gradient_accuracy.py
```

### Part 2: Mega Test (Large QP Benchmark)

Scripts live in `experiments/mega_test/`. See [experiments/mega_test/README.md](./experiments/mega_test/README.md) for details.

The workflow is:

1. Generate random data:

```bash
python experiments/mega_test/create_random_data.py [data_name]
```

2. Run benchmarks:

```bash
python experiments/mega_test/mega_exp.py [data_name] [model_name] --nolarge/--onlylarge
```

3. Plot results:

```bash
python experiments/mega_test/plot_backward_ratio_panels.py
```

### Part 3: Portfolio Experiment

Scripts live in `experiments/portfolio/`. See [experiments/portfolio/README.md](./experiments/portfolio/README.md) for details.

- Single run (choose one optimizer and horizon `H`):

```bash
bash experiments/portfolio/run_portfolio.sh
```

- Benchmark across optimizers and horizons (generates timing CSVs in `experiments/portfolio/logs/`):

```bash
bash experiments/portfolio/run_benchmark.sh
```

- Plot summary (requires `experiments/portfolio/logs/portfolio_summary.csv`):

```bash
python experiments/portfolio/plot_portfolio_summary.py
```

## Citing

If you use CVXPYlayers for research, please cite our accompanying paper:

```latex
@article{linghu2026penalty,
  title={A Penalty Approach for Differentiation Through Black-Box Quadratic Programming Solvers},
  author={Linghu, Yuxuan and Liu, Zhiyuan and Deng, Qi},
  journal={arXiv preprint arXiv:2602.14154},
  year={2026}
}
```
## dXPP: Penalty-Based Differentiation Through Black-Box QP Solvers

This repository introduces the **dXPP** framework for differentiation through quadratic programming (QP) solvers. The dXPP approach is penalty-based: it employs any black-box QP solver in the forward pass, and, in the backward pass, performs differentiation through a penalty-based reformulation of the original problem.

Linghu, Yuxuan, Zhiyuan Liu, and Qi Deng. **"A Penalty Approach for Differentiation Through Black-Box Quadratic Programming Solvers."** arXiv preprint arXiv:2602.14154, 2026.

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Citing](#citing)

## Installation

Create a Conda environment with Python 3.9, then install dependencies in order: core packages (PyTorch CPU, SciPy, NumPy, QP solvers), optional QP solver backends, PyTorch Geometric, differentiable QP layers (optnet, qpth, cvxpylayers, proxsuite), and extras for running examples.

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
python examples/cross.py
```

## Usage

dXPP solves the following QP and computes gradients through the solver via penalty smoothing:

$$
\min_x \;\tfrac{1}{2}x^\top Q x + q^\top x \quad \text{s.t.}\; Gx \le h,\; Ax = b
$$

### Solver Parameters

| Option            | Type      | Description                                                                                                      | Default     |
| ----------------- | --------- | ---------------------------------------------------------------------------------------------------------------- | ----------- |
| `beta`          | `float` | Smoothing parameter for the penalty reformulation.                                                               | `1e-4`    |
| `penalty_coeff` | `float` | Penalty strength multiplier.                                                                                     | `10`      |
| `eps_abs`       | `float` | Absolute tolerance passed to the QP solver.                                                                      | `1e-6`    |
| `eps_rel`       | `float` | Relative tolerance passed to the QP solver.                                                                      | `0`       |
| `solve_type`    | `str`   | `"dense"`, `"sparse"`, or `"auto"` (auto-detect from input layout).                                        | `"auto"`  |
| `qp_solver`     | `str`   | QP solver backend name (e.g.`"gurobi"`, `"cvxopt"`, `"osqp"`, `"piqp"`). Auto-selected if not specified. | `"auto"`  |
| `lin_solver`    | `str`   | Linear solver for backward pass:`"pardiso"`, `"qdldl"`, `"cholmod"`, or `"scipy"`.                       | `"scipy"` |
| `warm_start`    | `bool`  | Reuse previous primal solution as initialization.                                                                | `True`    |
| `verbose`       | `bool`  | Print debug information.                                                                                         | `False`   |

### Test

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dXPP import dXPPLayer

n, m = 2, 3
A_data = torch.tensor([[1.0, 2.0],
                       [-1.0, 1.0],
                       [0.5, -0.5]], dtype=torch.float64, requires_grad=True)
b_data = torch.tensor([1.0, -0.5, 0.2], dtype=torch.float64, requires_grad=True)

Q = A_data.T @ A_data + 1e-6 * torch.eye(n, dtype=torch.float64)
q = -(A_data.T @ b_data)
G = -torch.eye(n, dtype=torch.float64)
h = torch.zeros(n, dtype=torch.float64)

layer = dXPPLayer(beta=1e-6, penalty_coeff=10.0, eps_abs=1e-8, solve_type="dense")

x_star, mu_star, nu_star = layer(Q, q, G, h)
x_star.sum().backward()

print("x* =", x_star)
print("d(loss)/d(A_data) =", A_data.grad)
print("d(loss)/d(b_data) =", b_data.grad)
```

## Examples

- **Sparse QP diagnostic**: [`examples/cross.py`](./examples/cross.py) — loads a sparse QP from file and verifies forward solve and backward gradient.
- **Geometry**: [`examples/geometry/`](./examples/geometry/) — differentiable harmonic mapping with cone constraints. See [examples/geometry/README.md](./examples/geometry/README.md) for details.
- **Sudoku**: [`examples/sudoku/`](./examples/sudoku/) — learning Sudoku via differentiable QP layers. See [examples/sudoku/README.md](./examples/sudoku/README.md) for details.

## Citing

If you use dXPP for research, please cite our accompanying paper:

```latex
@article{linghu2026penalty,
  title={A Penalty Approach for Differentiation Through Black-Box Quadratic Programming Solvers},
  author={Linghu, Yuxuan and Liu, Zhiyuan and Deng, Qi},
  journal={arXiv preprint arXiv:2602.14154},
  year={2026}
}
```

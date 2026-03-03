# Portfolio Optimization Experiment

This experiment demonstrates multi-period portfolio optimization using different QP solvers, including our **dXPP** and several baselines.

## Overview

The goal is to optimize portfolio weights over a multi-period horizon $H$ while considering transaction costs and risk .

The core model is implemented in `model/poptimizer.py`, which constructs a large QP for the entire horizon $H$ and solves it using a differentiable layer.

## Data

The experiments use daily ETF price data.  
**Due to data licensing restrictions, we do not distribute the dataset in this repository.**

### Expected format

Please place the dataset at `data/etfs.csv` with the following **CSV schema** (no header):

- `RIC` (string)
- `Date` in `MM/DD/YYYY`
- `Price` (float)

You may download daily close prices and convert it to the format above.

## Running the Experiment

### Single Run

To run the experiment for a specific optimizer and horizon $H$, you can use the provided shell script:

```bash
bash run_portfolio.sh
```

You can modify the script to change parameters like `H` and `OPTIMIZER`. Available optimizers are: `dXPP`, `dQP`, `OptNet`, `Cvxpy`, and `SCQPTH`.

### Benchmark Run

To run a full benchmark across multiple horizons and all optimizers:

```bash
bash run_benchmark.sh
```

This script will:

1. Iterate through horizons $H \in \{10, 20, 50, 100, 150, 200\}$.
2. Run each optimizer for 100 epochs.
3. Extract timing information (forward, backward, and total time) and save it to `logs/[optimizer]_timing.csv`.

**Note on `train_len`**: The `train_len` parameter specifies the number of time steps (days) used for training in each rolling window. A larger `train_len` means more training data per window, resulting in longer training time. In the benchmark script (`run_benchmark.sh`), we set `train_len=1` for faster execution. For more realistic training, increase this value (e.g., `train_len=120` in `run_portfolio.sh`).

## Visualizing Results

After running the benchmark, you can plot the total time vs. horizon $H$ using:

```bash
python plot_portfolio_summary.py
```

This will generate a summary plot at `logs/portfolio_summary.pdf`. Note that the plotting script maps `dXPP` to `dQP` and the original `dQP` to `dQP (original)` as specified in the user requirements for consistent naming with the paper.

## Directory Structure

- `run.py`: Entry point for training and backtesting.
- `exp.py`: Orchestrates the backtesting process.
- `model/`:
  - `poptimizer.py`: The portfolio optimizer model with the differentiable QP layer.
  - `CNNLSTM.py`, `Linear.py`: Forecasting models for asset returns.
- `logs/`: Stores logs, timing CSVs, and generated plots.
- `data/`: Contains the ETF dataset.

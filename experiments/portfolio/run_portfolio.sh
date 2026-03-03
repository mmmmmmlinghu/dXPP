#!/usr/bin/env bash
set -euo pipefail

# Available optimizers: dXPP, dQP, OptNet, Cvxpy, SCQPTH
# Note: all solvers run in dense mode
# Change to the script directory (experiments/portfolio)
cd "$(dirname "$0")"

# Default example: use the ETF data provided in the repository
# Adjustable stage count H, keep horizon and output_dim consistent
H=10
OPTIMIZER="OptNet"  # dXPP, dQP, OptNet, Cvxpy, SCQPTH

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp and optimizer name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/${OPTIMIZER}_H${H}.log"

echo "Logging to: $LOGFILE"

python run.py \
  --is_training 1 \
  --max_steps 0 \
  --model Linear \
  --optimizer "$OPTIMIZER" \
  --data ETF \
  --data_path "$(pwd)/data/etfs.csv" \
  --device cpu \
  --itr 1 \
  --num_epochs 10 \
  --train_len 120 \
  --test_len 20 \
  --window 30 \
  --horizon "$H" \
  --output_dim "$H" \
  --batch_size 64 \
  --learning_rate 1e-2 \
  2>&1 | tee "$LOGFILE"

echo ""
echo "Log saved to: $LOGFILE"
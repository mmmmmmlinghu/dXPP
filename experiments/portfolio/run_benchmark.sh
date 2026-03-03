#!/usr/bin/env bash
set -euo pipefail

# Benchmark script for timing across optimizers and horizons.
# Keeps the same arguments as run_portfolio.sh (23-39), only varies H, optimizer, and num_epochs.

cd "$(dirname "$0")"

H_LIST=(10 20 50 100 150 200)
OPTIMIZERS=(dXPP dQP OptNet Cvxpy SCQPTH)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_one() {
  local optimizer="$1"
  local H="$2"
  local epochs="$3"
  local log_file="${LOG_DIR}/${optimizer}_H${H}.log"

  echo ""
  echo "[run] optimizer=${optimizer}, H=${H}, epochs=${epochs}"
  echo "Logging to: ${log_file}"

  python run.py \
    --is_training 1 \
    --model Linear \
    --optimizer "${optimizer}" \
    --data ETF \
    --data_path "$(pwd)/data/etfs.csv" \
    --device cpu \
    --itr 1 \
    --train_len 1 \
    --test_len 20 \
    --window 30 \
    --horizon "${H}" \
    --output_dim "${H}" \
    --batch_size 64 \
    --learning_rate 1e-2 \
    --num_epochs "${epochs}" \
    2>&1 | tee "${log_file}"

  python - "${log_file}" "${LOG_DIR}/${optimizer}_timing.csv" "${H}" "${epochs}" <<'PY'
import csv
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
H = int(sys.argv[3])
epochs = int(sys.argv[4])

pattern = re.compile(r'forward:\s*([0-9.]+)s,\s*backward:\s*([0-9.]+)s,\s*total:\s*([0-9.]+)s')
rows = []
for line in log_path.read_text().splitlines():
    m = pattern.search(line)
    if m:
        f, b, t = map(float, m.groups())
        rows.append((f, b, t))

if not rows:
    raise SystemExit(f"No epoch timing found in {log_path}")

if len(rows) > 1:
    max_idx = max(range(len(rows)), key=lambda i: rows[i][2])
    rows = [r for i, r in enumerate(rows) if i != max_idx]

avg_f = sum(r[0] for r in rows) / len(rows)
avg_b = sum(r[1] for r in rows) / len(rows)
avg_t = sum(r[2] for r in rows) / len(rows)
avg_f_ms = avg_f * 1000.0
avg_b_ms = avg_b * 1000.0
avg_t_ms = avg_t * 1000.0

write_header = not csv_path.exists()
with csv_path.open("a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["H", "epochs", "avg_forward_ms", "avg_backward_ms", "avg_total_ms", "log_file"])
    writer.writerow([H, epochs, f"{avg_f_ms:.2f}", f"{avg_b_ms:.2f}", f"{avg_t_ms:.2f}", str(log_path)])

print(f"[info] {log_path.name}: avg_forward={avg_f_ms:.2f}ms avg_backward={avg_b_ms:.2f}ms avg_total={avg_t_ms:.2f}ms")
PY
}

for optimizer in "${OPTIMIZERS[@]}"; do
  epochs=100
  for H in "${H_LIST[@]}"; do
    run_one "${optimizer}" "${H}" "${epochs}"
  done
done

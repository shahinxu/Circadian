#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/unites2/home/zhx/Circadian"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/zhx/miniconda3/envs/tempo310/bin/python" ]]; then
  PY="/home/zhx/miniconda3/envs/tempo310/bin/python"
else
  PY="$(command -v python || true)"
fi
TEMPO_DIR="$ROOT/tempo"

H5AD="$ROOT/ad427_compressed.h5ad"
TEMPO_OUT="$ROOT/ad427_tempo_out"
TEMPO_CONFIG="$TEMPO_DIR/test_data/tempo_test_out/config.txt"
TEMPO_RUNTIME_CONFIG="$TEMPO_OUT/config.runtime.txt"
COGS="$ROOT/02_AD223/Cyclops2/ROSMAP_CYCLOPS_phases_409.csv"
SUMMARY_OUT="$ROOT/ad427_tempo_vs_cyclops_summary.csv"
TEMPO_CELL_POST="$TEMPO_OUT/tempo_results/opt/cell_posterior.tsv"

# CPU limiting knobs (override at runtime, e.g. CPU_THREADS=6 TORCH_INTEROP_THREADS=1)
CPU_THREADS="${CPU_THREADS:-8}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
CPU_CORES="${CPU_CORES:-}"

# Scheduling knobs for lower system interference.
# NICE_LEVEL range: [-20, 19], larger means lower CPU priority.
# IONICE_CLASS: 2 (best-effort) with IONICE_LEVEL 0-7, larger means lower I/O priority.
NICE_LEVEL="${NICE_LEVEL:-10}"
USE_IONICE="${USE_IONICE:-1}"
IONICE_CLASS="${IONICE_CLASS:-2}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"

# Tempo config knobs.
HVG_STD_RESIDUAL_THRESHOLD="${HVG_STD_RESIDUAL_THRESHOLD:-0.5}"
TEMPO_H5AD_LOAD_MODE="${TEMPO_H5AD_LOAD_MODE:-auto}"

mkdir -p "$TEMPO_OUT"

export PYTHONPATH="$TEMPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"
export TORCH_NUM_THREADS="$CPU_THREADS"
export TORCH_INTEROP_THREADS="$TORCH_INTEROP_THREADS"
export TEMPO_H5AD_LOAD_MODE
export PYTHONUNBUFFERED=1

if [[ -z "$PY" ]]; then
  echo "python not found in current environment." >&2
  exit 1
fi

echo "Using python: $PY"
echo "CPU thread limit: $CPU_THREADS (torch interop: $TORCH_INTEROP_THREADS)"
if [[ -n "$CPU_CORES" ]]; then
  echo "CPU core pinning: $CPU_CORES"
fi
echo "CPU scheduling nice level: $NICE_LEVEL"
if [[ "$USE_IONICE" == "1" ]]; then
  echo "I/O scheduling ionice class/level: $IONICE_CLASS/$IONICE_LEVEL"
fi
echo "HVG std residual threshold: $HVG_STD_RESIDUAL_THRESHOLD"
echo "H5AD load mode: $TEMPO_H5AD_LOAD_MODE"

RUNNER=()
if [[ "$USE_IONICE" == "1" ]]; then
  if command -v ionice >/dev/null 2>&1; then
    RUNNER+=(ionice -c "$IONICE_CLASS" -n "$IONICE_LEVEL")
  else
    echo "Warning: ionice not found; I/O priority control ignored." >&2
  fi
fi

if command -v nice >/dev/null 2>&1; then
  RUNNER+=(nice -n "$NICE_LEVEL")
else
  echo "Warning: nice not found; CPU priority control ignored." >&2
fi

if [[ -n "$CPU_CORES" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    RUNNER+=(taskset -c "$CPU_CORES")
  else
    echo "Warning: taskset not found; CPU_CORES pinning ignored." >&2
  fi
fi

# Build a runtime config with absolute paths for files referenced by Tempo.
$PY - <<'PY'
import ast
import os
from pathlib import Path

root = Path('/mnt/unites2/home/zhx/Circadian')
tempo_dir = root / 'tempo'
config_in = tempo_dir / 'test_data' / 'tempo_test_out' / 'config.txt'
config_out = root / 'ad427_tempo_out' / 'config.runtime.txt'

cfg = ast.literal_eval(config_in.read_text())
for key in ['gene_acrophase_prior_path', 'core_clock_gene_path', 'cell_phase_prior_path']:
  if key in cfg and isinstance(cfg[key], str) and cfg[key]:
    p = Path(cfg[key])
    if not p.is_absolute():
      candidates = [tempo_dir / 'test_data' / p, tempo_dir / p, root / p]
      for c in candidates:
        if c.exists():
          cfg[key] = str(c)
          break

hvg = os.environ.get('HVG_STD_RESIDUAL_THRESHOLD')
if hvg:
  cfg['hv_std_residual_threshold'] = float(hvg)

config_out.write_text(str(cfg))
print(f'Runtime config: {config_out}')
print(f"Runtime hv_std_residual_threshold: {cfg.get('hv_std_residual_threshold')}")
PY

# Run Tempo on the full ad427 dataset.
# This may take a long time and require substantial RAM.
if [[ -s "$TEMPO_CELL_POST" && "${FORCE_TEMPO:-0}" != "1" ]]; then
  echo "Tempo final output already exists, skipping tempo.run_tempo: $TEMPO_CELL_POST"
else
  "${RUNNER[@]}" "$PY" -m tempo.run_tempo -f "$H5AD" -o "$TEMPO_OUT" -c "$TEMPO_RUNTIME_CONFIG"
fi

# Compare Tempo against the 220 overlapping samples in the CYCLOPS reference.
if [[ -s "$SUMMARY_OUT" && "${FORCE_COMPARE:-0}" != "1" ]]; then
  echo "Summary already exists, skipping compare step: $SUMMARY_OUT"
else
  "${RUNNER[@]}" "$PY" "$ROOT/compare_tempo_vs_cyclops.py" \
    --h5ad "$H5AD" \
    --cell-posterior "$TEMPO_CELL_POST" \
    --cyclops "$COGS" \
    --out "$SUMMARY_OUT"
fi

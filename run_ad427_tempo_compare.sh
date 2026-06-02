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

mkdir -p "$TEMPO_OUT"

export PYTHONPATH="$TEMPO_DIR:${PYTHONPATH:-}"

if [[ -z "$PY" ]]; then
  echo "python not found in current environment." >&2
  exit 1
fi

echo "Using python: $PY"

# Build a runtime config with absolute paths for files referenced by Tempo.
$PY - <<'PY'
import ast
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

config_out.write_text(str(cfg))
print(f'Runtime config: {config_out}')
PY

# Run Tempo on the full ad427 dataset.
# This may take a long time and require substantial RAM.
$PY -m tempo.run_tempo -f "$H5AD" -o "$TEMPO_OUT" -c "$TEMPO_RUNTIME_CONFIG"

# Compare Tempo against the 220 overlapping samples in the CYCLOPS reference.
$PY "$ROOT/compare_tempo_vs_cyclops.py" \
  --h5ad "$H5AD" \
  --cell-posterior "$TEMPO_OUT/tempo_results/opt/cell_posterior.tsv" \
  --cyclops "$COGS" \
  --out "$SUMMARY_OUT"

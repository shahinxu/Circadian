# CYCLOPS-2.0

Circadian phase prediction.

## Usage

```bash
./pipeline.sh <data_path> [output_dir]
./batch_process.sh [data_dir]
```

## Data

Each dataset needs:
- `expression.csv` (Gene_Symbol + samples)
- `seed_genes.txt` (one per line)
- `metadata.csv` (optional)

## Output

`results/<dataset>/` contains:
- `Fit_Output_*.csv`
- `clock_genes*.png`
- `phase_distribution.png`

## Files

- `CYCLOPS.jl` - Core algorithm
- `run_cyclops.jl` - Main script
- `pipeline.sh` - Run single
- `batch_process.sh` - Run all

See `README_ORIGINAL.md` for algorithm details.

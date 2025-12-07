#!/bin/bash
set -e

DATA_DIR="${1:-/home/rzh/zhenx/circadian/data}"
RESULTS_DIR="./results"

[ ! -d "$DATA_DIR" ] && echo "Error: $DATA_DIR not found" && exit 1

mkdir -p "$RESULTS_DIR"

find "$DATA_DIR" -name "expression.csv" | while read expr_file; do
    dataset_dir=$(dirname "$expr_file")
    [ ! -f "$dataset_dir/seed_genes.txt" ] && continue
    
    dataset_name=$(basename "$dataset_dir")
    parent_dir=$(basename "$(dirname "$dataset_dir")")
    
    [ "$parent_dir" != "$(basename "$DATA_DIR")" ] && full_name="${parent_dir}_${dataset_name}" || full_name="$dataset_name"
    
    echo "Processing: $full_name"
    ./pipeline.sh "$dataset_dir" "$RESULTS_DIR/${full_name}" && echo "[OK] $full_name" || echo "[FAIL] $full_name"
done

echo "Done!"

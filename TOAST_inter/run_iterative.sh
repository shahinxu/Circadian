#!/bin/bash
# Iterative TOAST Training Pipeline
# Example: iteratively expand gene set from seed genes

# Example dataset
DATASET="../data/GSE54651/adrenal_gland"

python iterative_toast.py \
  --data_path "${DATASET}" \
  --num_iterations 5 \
  --genes_per_iteration 100 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --device cuda \
  --output_dir ./results_iterative

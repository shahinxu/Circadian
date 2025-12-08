#!/bin/bash

# Example run script for iterative TOAST training

DATASET_PATH="GSE54651/adrenal_gland"
SEED_GENES="../data/${DATASET_PATH}/seed_genes.txt"

# Default parameters
N_ITERATIONS=5
GENES_PER_ITER=100
EPOCHS=1000
N_COMPONENTS=5
LR=0.001
DROPOUT=0.1
DEVICE="cuda"
RHYTHMICITY_METHOD="correlation"  # or "anova"

# Run iterative training
python train_iterative.py \
    --dataset_path "${DATASET_PATH}" \
    --seed_genes "${SEED_GENES}" \
    --n_iterations ${N_ITERATIONS} \
    --genes_per_iteration ${GENES_PER_ITER} \
    --num_epochs ${EPOCHS} \
    --n_components ${N_COMPONENTS} \
    --lr ${LR} \
    --dropout ${DROPOUT} \
    --device ${DEVICE} \
    --rhythmicity_method ${RHYTHMICITY_METHOD} \
    --random_seed 42

echo "Iterative training complete!"

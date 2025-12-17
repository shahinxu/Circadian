#!/bin/bash
# Run script for Set Transformer - batch processing multiple subdirectories

# Single dataset example
# python train.py \
#     --dataset_path "GSE54651/aorta" \
#     --num_epochs 1000 \
#     --lr 0.001 \
#     --device cuda \
#     --d_model 128 \
#     --use_isab \
#     --num_inducing_points 32

DATASET_FOLDER="GSE54652"

for subdir in $(ls ../data/${DATASET_FOLDER}); do
    if [ ! -d "../data/${DATASET_FOLDER}/${subdir}" ]; then
        continue
    fi
    
    echo "========================================"
    echo "Processing: ${DATASET_FOLDER}/${subdir}"
    echo "========================================"
    
    python train.py \
        --dataset_path "${DATASET_FOLDER}/${subdir}" \
        --num_epochs 1000 \
        --lr 0.001 \
        --device cuda \
        --d_model 128 \
        # --use_isab \
        # --num_inducing_points 32 \
        # --use_seed_genes False
    
    echo "Finished: ${subdir}"
    echo ""
done

# echo "All subdirectories processed!"


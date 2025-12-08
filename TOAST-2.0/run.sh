#!/bin/bash
# Pathway-based TOAST training script
# wandb is enabled by default, use --no_wandb to disable

# Single dataset example
python train.py \
  --dataset_path "GSE54651_all_tissues" \
  --num_epochs 500 \
  --lr 5e-4 \
  --device cuda

# Batch processing example
# folder="GSE54651"

# for subdir in $(ls ../data/${folder}); do
# 	echo "Processing ${subdir}..."
# 	python train.py \
# 		--dataset_path "${folder}/${subdir}" \
# 		--num_epochs 2000 \
# 		--lr 5e-4 \
# 		--device cuda \
# 		--wandb_project "circadian-${folder}"
# 	echo "Finished ${subdir}"
# done
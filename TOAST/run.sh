# python train.py \
# --dataset_path "GTEx/GTEx_brain_amygdala/" \
# --n_components 5 \
# --num_epochs 2000 \
# --lr 0.001 \
# --device cuda

folder="Zhang_CancerCell_2025_sub"

for subdir in $(ls ../data/${folder}); do
	echo "Processing ${subdir}..."
	python train.py \
		--dataset_path "${folder}/${subdir}" \
		--n_components 5 \
		--num_epochs 2000 \
		--lr 0.001 \
		--device cuda
	echo "Finished training ${subdir}"
	
	# Always generate plots after training
	echo "Generating plots for ${subdir}..."
	python plot_all_from_predictions.py --results_base "results/${folder}/${subdir}" || echo "Warning: Plot generation failed for ${subdir}"
	echo "Finished ${subdir}"
done

# Generate comparison plot across all cell types
echo "Generating comparison plot across all cell types..."
python plot_all_from_predictions.py --results_base "results/${folder}" --compare_mode || echo "Warning: Comparison plot failed"
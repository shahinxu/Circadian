python train.py \
--dataset_path "GSE54651/adrenal_gland" \
--n_components 5 \
--num_epochs 1000 \
--lr 0.001 \
--device cuda

# folder="GSE261698"

# for subdir in $(ls ../data/${folder}); do
# 	echo "Processing ${subdir}..."
# 	python train.py \
# 		--dataset_path "${folder}/${subdir}" \
# 		--n_components 5 \
# 		--num_epochs 2000 \
# 		--lr 0.001 \
# 		--device cuda
# 	echo "Finished ${subdir}"
# done
#!/bin/bash
# Batch plot tumor core genes for all transfer learning results

echo "========================================"
echo "Plotting Tumor Core Genes for All Datasets"
echo "========================================"

# Create unified output directory
output_dir="./results/tumor_core_genes_all"
mkdir -p "$output_dir"
echo "Output directory: $output_dir"

# Array of result directories
results=(
    "./results/GTEx_Zhang_Transfer_Bcell_20251229_134750"
    "./results/GTEx_Zhang_Transfer_CD4Tcell_20251229_140010"
    "./results/GTEx_Zhang_Transfer_CD8Tcell_20251229_140334"
    "./results/GTEx_Zhang_Transfer_GSE176078_20251229_142755"
    "./results/GTEx_Zhang_Transfer_Myeloid_20251229_140650"
    "./results/GTEx_Zhang_Transfer_NKcell_20251229_140833"
)

# Loop through each result directory
for result_dir in "${results[@]}"; do
    if [ -d "$result_dir" ]; then
        echo ""
        echo "Processing: $result_dir"
        python plot_tumor_core_genes.py "$result_dir" "$output_dir"
    else
        echo "Warning: Directory not found: $result_dir"
    fi
done

echo ""
echo "========================================"
echo "All plots saved to: $output_dir"
echo "========================================"
ls -lh "$output_dir"

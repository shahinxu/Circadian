#!/bin/bash
# Clean up non-essential files from transfer learning results
# Keep: Fit_Output (phases), transfer_learning_summary, tumor core gene plots
# Remove: Trained models, intermediate CSVs, atlas plots

echo "========================================"
echo "Cleaning up non-essential result files"
echo "========================================"

results=(
    "./results/GTEx_Zhang_Transfer_Bcell_20251229_134750"
    "./results/GTEx_Zhang_Transfer_CD4Tcell_20251229_140010"
    "./results/GTEx_Zhang_Transfer_CD8Tcell_20251229_140334"
    "./results/GTEx_Zhang_Transfer_GSE176078_20251229_142755"
    "./results/GTEx_Zhang_Transfer_Myeloid_20251229_140650"
    "./results/GTEx_Zhang_Transfer_NKcell_20251229_140833"
)

for result_dir in "${results[@]}"; do
    if [ -d "$result_dir" ]; then
        echo ""
        echo "Processing: $result_dir"
        
        # Files to remove (less critical for analysis)
        rm -v "$result_dir"/Trained_Model_*.csv 2>/dev/null
        rm -v "$result_dir"/Trained_Parameter_Dictionary_*.csv 2>/dev/null
        rm -v "$result_dir"/Metric_Correlation_to_Eigengenes_*.csv 2>/dev/null
        rm -v "$result_dir"/Genes_of_Interest_Aligned_Cosine_Fit_*.csv 2>/dev/null
        rm -v "$result_dir"/Mouse_Atlas_Aligned_Cosine_Fit_*.csv 2>/dev/null
        rm -v "$result_dir"/Genes_of_Interest_Aligned_Acrophase_Plot_*.png 2>/dev/null
        rm -v "$result_dir"/Mouse_Atlas_Aligned_Acrophase_Plot_*.png 2>/dev/null
        
        # Keep:
        # - Fit_Output_*.csv (sample phases - IMPORTANT)
        # - transfer_learning_summary.txt (metadata)
        # - tumor_core_genes_*.png/pdf (analysis plots)
        
        echo "Kept essential files:"
        ls -lh "$result_dir"/ | grep -E "(Fit_Output|summary|tumor_core)"
    fi
done

echo ""
echo "========================================"
echo "Cleanup complete!"
echo "========================================"

# Show new disk usage
echo ""
echo "New disk usage:"
du -sh "${results[@]}"

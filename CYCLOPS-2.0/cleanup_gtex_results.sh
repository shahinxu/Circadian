#!/bin/bash
# Clean up non-essential files from GTEx and GSE54651 result directories
# Keep: Fit_Output (phases), plots (png)
# Remove: Trained models, intermediate CSVs, parameter dictionaries

echo "========================================"
echo "Cleaning GTEx and GSE54651 result files"
echo "========================================"

# Clean GTEx directories
echo ""
echo "Cleaning GTEx results..."
for result_dir in /playpen-shared/zhenx/Circadian/CYCLOPS-2.0/results/GTEx/GTEx_GTEx_*/; do
    if [ -d "$result_dir" ]; then
        dir_name=$(basename "$result_dir")
        echo "  $dir_name"
        
        rm -f "$result_dir"/Trained_Model_*.csv 2>/dev/null
        rm -f "$result_dir"/Trained_Parameter_Dictionary_*.csv 2>/dev/null
        rm -f "$result_dir"/Metric_Correlation_to_Eigengenes_*.csv 2>/dev/null
        rm -f "$result_dir"/Mouse_Atlas_Aligned_Cosine_Fit_*.csv 2>/dev/null
        rm -f "$result_dir"/Mouse_Atlas_Aligned_Acrophase_Plot_*.png 2>/dev/null
        rm -rf "$result_dir"/phase_vs_metadata 2>/dev/null
    fi
done

# Clean GSE54651 directories
echo ""
echo "Cleaning GSE54651 results..."
for result_dir in /playpen-shared/zhenx/Circadian/CYCLOPS-2.0/results/GSE54651/GSE54651_*/; do
    if [ -d "$result_dir" ]; then
        dir_name=$(basename "$result_dir")
        echo "  $dir_name"
        
        rm -f "$result_dir"/Trained_Model_*.csv 2>/dev/null
        rm -f "$result_dir"/Trained_Parameter_Dictionary_*.csv 2>/dev/null
        rm -f "$result_dir"/Metric_Correlation_to_Eigengenes_*.csv 2>/dev/null
        rm -f "$result_dir"/Mouse_Atlas_Aligned_Cosine_Fit_*.csv 2>/dev/null
        rm -f "$result_dir"/Mouse_Atlas_Aligned_Acrophase_Plot_*.png 2>/dev/null
        rm -rf "$result_dir"/phase_vs_metadata 2>/dev/null
    fi
done

echo ""
echo "========================================"
echo "Cleanup complete!"
echo "========================================"

# Show disk usage summary
echo ""
echo "Disk usage after cleanup:"
du -sh /playpen-shared/zhenx/Circadian/CYCLOPS-2.0/results/GTEx/
du -sh /playpen-shared/zhenx/Circadian/CYCLOPS-2.0/results/GSE54651/

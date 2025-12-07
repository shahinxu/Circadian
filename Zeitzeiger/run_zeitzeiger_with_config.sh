#!/bin/bash
# 使用配置文件批量运行 Zeitzeiger
# Usage: bash run_zeitzeiger_with_config.sh [DATASET1 DATASET2 ...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/zeitzeiger_config.sh"

# 加载配置
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Warning: Config file not found, using defaults"
    DATA_ROOT="../data"
    OUTPUT_ROOT="./results"
    VERBOSE=""
fi

# 获取数据集参数
get_param() {
    local dataset=$1
    local param_name=$2
    local default_value=$3
    local var_name="${dataset}_${param_name}"
    echo "${!var_name:-$default_value}"
}

# 运行单个数据集
run_dataset() {
    local dataset=$1
    
    echo "=========================================="
    echo "Processing: $dataset"
    echo "=========================================="
    
    local sample_col=$(get_param "$dataset" "SAMPLE_COL" "Sample")
    local time_col=$(get_param "$dataset" "TIME_COL" "Time_Phase")
    local time_format=$(get_param "$dataset" "TIME_FORMAT" "auto")
    
    python "$SCRIPT_DIR/run_zeitzeiger_auto.py" \
        --dataset "$dataset" \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --sample-col "$sample_col" \
        --time-col "$time_col" \
        --time-format "$time_format" \
        $VERBOSE
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $dataset"
    else
        echo "✗ Failed: $dataset"
        return 1
    fi
}

# 主程序
if [ $# -eq 0 ]; then
    datasets=("GSE54651" "GSE54652")
else
    datasets=("$@")
fi

success=0
fail=0

for dataset in "${datasets[@]}"; do
    if run_dataset "$dataset"; then
        ((success++))
    else
        ((fail++))
    fi
done

echo "=========================================="
echo "SUMMARY: $success successful, $fail failed"
echo "=========================================="

exit $fail

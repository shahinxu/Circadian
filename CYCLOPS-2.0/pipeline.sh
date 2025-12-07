#!/bin/bash

JULIA_CMD="./julia-1.6.7/bin/julia"

DATA_INPUT="${1}"
OUTPUT_DIR="${2}"

if [ -z "$DATA_INPUT" ]; then
    echo "Usage: $0 <dataset_name_or_path> [output_dir]"
    exit 1
fi

# Auto-detect if input is just a name or full path
if [ -d "$DATA_INPUT" ]; then
    DATA_PATH="$DATA_INPUT"
elif [ -d "/home/rzh/zhenx/circadian/data/$DATA_INPUT" ]; then
    DATA_PATH="/home/rzh/zhenx/circadian/data/$DATA_INPUT"
else
    echo "Error: Cannot find dataset '$DATA_INPUT'"
    exit 1
fi

# Check if this is a direct dataset or has subdatasets
if [ -f "$DATA_PATH/expression.csv" ] && [ -f "$DATA_PATH/seed_genes.txt" ]; then
    # Direct dataset - run once
    DATASET_NAME=$(basename "$DATA_PATH")
    
    if [ -z "$OUTPUT_DIR" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="./results/${DATASET_NAME}_${TIMESTAMP}"
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    echo "Running CYCLOPS: $DATASET_NAME"
    echo "Output: $OUTPUT_DIR"
    
    sed "s|DATA_PATH_PLACEHOLDER|$DATA_PATH|g; s|OUTPUT_PATH_PLACEHOLDER|$OUTPUT_DIR|g" \
        run_cyclops.jl > temp_run.jl
    
    $JULIA_CMD temp_run.jl
    
    rm -f temp_run.jl
else
    # Has subdatasets - run each subdirectory
    echo "No expression.csv found. Processing subdatasets in $DATA_PATH"
    
    PARENT_NAME=$(basename "$DATA_PATH")
    
    for SUBDIR in "$DATA_PATH"/*/; do
        [ ! -d "$SUBDIR" ] && continue
        
        SUBNAME=$(basename "$SUBDIR")
        
        if [ ! -f "$SUBDIR/expression.csv" ] || [ ! -f "$SUBDIR/seed_genes.txt" ]; then
            echo "Skipping $SUBNAME (missing files)"
            continue
        fi
        
        if [ -z "$OUTPUT_DIR" ]; then
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            SUB_OUTPUT="./results/${PARENT_NAME}_${SUBNAME}_${TIMESTAMP}"
        else
            SUB_OUTPUT="${OUTPUT_DIR}/${SUBNAME}"
        fi
        
        mkdir -p "$SUB_OUTPUT"
        
        echo "Running CYCLOPS: $PARENT_NAME/$SUBNAME"
        echo "Output: $SUB_OUTPUT"
        
        sed "s|DATA_PATH_PLACEHOLDER|${SUBDIR%/}|g; s|OUTPUT_PATH_PLACEHOLDER|$SUB_OUTPUT|g" \
            run_cyclops.jl > temp_run.jl
        
        $JULIA_CMD temp_run.jl && echo "✓ Completed: $SUBNAME" || echo "✗ Failed: $SUBNAME"
        
        rm -f temp_run.jl
    done
    
    echo "All subdatasets processed!"
fi

echo "Completed: $OUTPUT_DIR"

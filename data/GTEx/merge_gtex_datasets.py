#!/usr/bin/env python3
"""
合并所有GTEx子数据集到一个完整的数据集
包括metadata.csv和expression.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def merge_gtex_datasets(gtex_dir, output_dir):
    """
    合并所有GTEx子目录中的metadata和expression文件
    
    Args:
        gtex_dir: GTEx数据根目录
        output_dir: 输出目录
    """
    gtex_path = Path(gtex_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有子目录
    subdirs = [d for d in gtex_path.iterdir() if d.is_dir() and d.name.startswith('GTEx_')]
    subdirs = sorted(subdirs)
    
    print(f"Found {len(subdirs)} GTEx tissue subdirectories")
    
    # 合并metadata
    print("\n=== Merging metadata.csv ===")
    metadata_list = []
    
    for subdir in subdirs:
        metadata_file = subdir / 'metadata.csv'
        if not metadata_file.exists():
            print(f"  [WARN] {subdir.name}: metadata.csv not found, skipping")
            continue
        
        df = pd.read_csv(metadata_file)
        tissue_name = subdir.name
        
        # 确保有tissue列
        if 'tissue_D' not in df.columns:
            df['tissue_D'] = tissue_name
        
        metadata_list.append(df)
        print(f"  {tissue_name}: {len(df)} samples")
    
    if metadata_list:
        merged_metadata = pd.concat(metadata_list, ignore_index=True)
        output_metadata = output_path / 'metadata.csv'
        merged_metadata.to_csv(output_metadata, index=False)
        print(f"\n[OK] Merged metadata saved: {output_metadata}")
        print(f"  Total samples: {len(merged_metadata)}")
        print(f"  Columns: {list(merged_metadata.columns)}")
    else:
        print("  [ERROR] No metadata files found!")
        return
    
    # 合并expression
    print("\n=== Merging expression.csv ===")
    
    # 首先读取第一个expression文件获取基因列表
    first_expr_file = None
    for subdir in subdirs:
        expr_file = subdir / 'expression.csv'
        if expr_file.exists():
            first_expr_file = expr_file
            break
    
    if first_expr_file is None:
        print("  [ERROR] No expression files found!")
        return
    
    print(f"  Reading gene list from {first_expr_file.parent.name}...")
    first_df = pd.read_csv(first_expr_file, low_memory=False)
    
    if 'Gene_Symbol' not in first_df.columns:
        print("  [ERROR] Gene_Symbol column not found!")
        return
    
    # Handle duplicate Gene_Symbols by keeping first occurrence
    if first_df['Gene_Symbol'].duplicated().any():
        n_dup = first_df['Gene_Symbol'].duplicated().sum()
        print(f"  [WARN] Found {n_dup} duplicate gene symbols, keeping first occurrence")
        first_df = first_df.drop_duplicates(subset='Gene_Symbol', keep='first')
    
    # Use Gene_Symbol as index
    gene_symbols = first_df['Gene_Symbol'].values
    print(f"  Found {len(gene_symbols)} unique genes")
    
    # 创建合并后的DataFrame，从Gene_Symbol列开始
    merged_expr = pd.DataFrame({'Gene_Symbol': gene_symbols})
    
    # 逐个读取并合并expression文件
    all_sample_columns = []
    
    for subdir in subdirs:
        expr_file = subdir / 'expression.csv'
        if not expr_file.exists():
            print(f"  [WARN] {subdir.name}: expression.csv not found, skipping")
            continue
        
        df = pd.read_csv(expr_file, low_memory=False)
        
        # Handle duplicate Gene_Symbols
        if 'Gene_Symbol' in df.columns and df['Gene_Symbol'].duplicated().any():
            df = df.drop_duplicates(subset='Gene_Symbol', keep='first')
        
        # Ensure gene order matches using merge instead of reindex to avoid duplicate label errors
        if 'Gene_Symbol' in df.columns:
            # Merge on Gene_Symbol to align genes
            gene_df = pd.DataFrame({'Gene_Symbol': gene_symbols})
            df = gene_df.merge(df, on='Gene_Symbol', how='left')
        
        # Get sample columns (exclude Gene_Symbol)
        sample_cols = [col for col in df.columns if col != 'Gene_Symbol']
        
        # 检查样本列是否与metadata中的Sample列匹配
        tissue_metadata = merged_metadata[merged_metadata['tissue_D'] == subdir.name]
        if len(tissue_metadata) > 0:
            metadata_samples = set(tissue_metadata['Sample'].astype(str))
            expr_samples = set(sample_cols)
            matched = metadata_samples & expr_samples
            print(f"  {subdir.name}: {len(sample_cols)} expression columns, {len(matched)} matched with metadata")
        else:
            print(f"  {subdir.name}: {len(sample_cols)} expression columns")
        
        # 添加样本列到合并的DataFrame
        for col in sample_cols:
            if col not in merged_expr.columns:
                merged_expr[col] = df[col].values
                all_sample_columns.append(col)
    
    # Save merged expression file
    output_expression = output_path / 'expression.csv'
    print(f"\n  Saving merged expression file...")
    merged_expr.to_csv(output_expression, index=False)
    print(f"[OK] Merged expression saved: {output_expression}")
    print(f"  Total genes: {len(gene_symbols)}")
    print(f"  Total sample columns: {len(all_sample_columns)}")
    
    # 检查metadata和expression的样本是否匹配
    print("\n=== Validation ===")
    metadata_samples = set(merged_metadata['Sample'].astype(str))
    expr_samples = set(all_sample_columns)
    
    matched_samples = metadata_samples & expr_samples
    metadata_only = metadata_samples - expr_samples
    expr_only = expr_samples - metadata_samples
    
    print(f"  Samples in both metadata and expression: {len(matched_samples)}")
    print(f"  Samples only in metadata: {len(metadata_only)}")
    print(f"  Samples only in expression: {len(expr_only)}")
    
    if len(metadata_only) > 0:
        print(f"  [WARN] Some samples in metadata not found in expression")
    if len(expr_only) > 0:
        print(f"  [WARN] Some samples in expression not found in metadata")
    
    print("\n=== Summary ===")
    print(f"  Output directory: {output_path}")
    print(f"  Merged {len(subdirs)} tissue types")
    print(f"  Total samples: {len(merged_metadata)}")
    print(f"  Total genes: {len(gene_symbols)}")
    print(f"  Expression matrix shape: {merged_expr.shape}")

if __name__ == '__main__':
    gtex_dir = r'D:\CriticalFile\projects\Circadian\data\GTEx'
    output_dir = r'D:\CriticalFile\projects\Circadian\data\GTEx_merged'
    
    print("GTEx Dataset Merger")
    print("=" * 60)
    
    merge_gtex_datasets(gtex_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Done!")

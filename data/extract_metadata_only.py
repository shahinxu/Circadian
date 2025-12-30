#!/usr/bin/env python3
"""
Extract comprehensive metadata from h5ad file for existing pseudo-bulk samples
Only reads metadata, no expression matrix processing
"""
import h5py
import pandas as pd
import numpy as np
from collections import Counter
import sys

def read_categorical_column(h5_file, column_name):
    """Read a categorical column from h5ad obs"""
    categories = h5_file[f'obs/{column_name}/categories'][:]
    codes = h5_file[f'obs/{column_name}/codes'][:]
    if categories.dtype == 'O':
        categories = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in categories]
    return [categories[code] if code >= 0 else 'NA' for code in codes]

h5_path = sys.argv[1] if len(sys.argv) > 1 else 'ad427_compressed.h5ad'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'pseudobulk_by_sample/metadata_full.csv'

print("="*80)
print("Extracting metadata from h5ad file")
print("="*80)
print(f"Input: {h5_path}")
print(f"Output: {output_file}")

with h5py.File(h5_path, 'r') as f:
    print("\n1. Reading Sample grouping...")
    group_labels = read_categorical_column(f, 'Sample')
    n_cells = len(group_labels)
    print(f"   Total cells: {n_cells:,}")
    
    # Count samples
    from collections import Counter
    sample_counts = Counter(group_labels)
    print(f"   Unique samples: {len(sample_counts):,}")
    
    print("\n2. Reading all metadata columns...")
    obs_keys = [k for k in f['obs'].keys() if not k.endswith('/categories') and not k.endswith('/codes')]
    
    sample_metadata = {}
    skipped = []
    
    for key in obs_keys:
        try:
            if f'obs/{key}/categories' in f:
                # Categorical
                sample_metadata[key] = read_categorical_column(f, key)
            else:
                # Numeric
                data = f[f'obs/{key}'][:]
                sample_metadata[key] = data
        except Exception as e:
            skipped.append(f"{key}: {e}")
    
    print(f"   Loaded {len(sample_metadata)} columns")
    if skipped:
        print(f"   Skipped {len(skipped)} columns")
    
    print("\n3. Aggregating metadata by sample...")
    metadata_records = []
    
    for i, (sample, count) in enumerate(sorted(sample_counts.items())):
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(sample_counts)} samples...")
        
        mask = np.array([g == sample for g in group_labels])
        
        meta_record = {
            'Sample': sample,
            'n_cells': count
        }
        
        # Aggregate each metadata column
        for key, values in sample_metadata.items():
            if key in ['Sample_Barcode', 'Sample_barcode', 'index']:
                continue  # Skip cell-specific IDs
            
            group_values = [values[j] for j in range(len(values)) if mask[j]]
            
            # Check if numeric
            if isinstance(group_values[0], (int, float, np.integer, np.floating)):
                # Numeric: use mean, ignore NaN
                valid_vals = [v for v in group_values if not (isinstance(v, float) and np.isnan(v))]
                if valid_vals:
                    meta_record[key] = np.mean(valid_vals)
                else:
                    meta_record[key] = np.nan
            else:
                # Categorical: use most common
                counter = Counter(group_values)
                meta_record[key] = counter.most_common(1)[0][0]
        
        metadata_records.append(meta_record)
    
    print(f"   Completed {len(metadata_records)} samples")

print("\n4. Creating DataFrame...")
metadata_df = pd.DataFrame(metadata_records)

# Clean column names to avoid special characters
metadata_df.columns = [col.replace('.', '_').replace(' ', '_') for col in metadata_df.columns]

print(f"\nMetadata shape: {metadata_df.shape}")
print(f"Columns ({len(metadata_df.columns)}): {list(metadata_df.columns)[:20]}...")

print("\n5. Saving...")
# Convert all columns to simple types to avoid numpy.rec issue
for col in metadata_df.columns:
    if metadata_df[col].dtype == 'object':
        metadata_df[col] = metadata_df[col].astype(str)

metadata_df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")

print("\n" + "="*80)
print("Sample metadata preview:")
print("="*80)
print(metadata_df.head())

print("\n" + "="*80)
print("Complete!")
print("="*80)

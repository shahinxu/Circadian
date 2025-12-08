#!/usr/bin/env python3
"""
Calculate DIP (Donor Internal Phase) from TIP (Tissue/Cell-type Internal Phase)
Based on the CYCLOPS phase predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def circular_mean(phases):
    """Calculate circular mean of phases (in radians)"""
    if len(phases) == 0:
        return np.nan
    sin_sum = np.sum(np.sin(phases))
    cos_sum = np.sum(np.cos(phases))
    mean_phase = np.arctan2(sin_sum, cos_sum)
    # Convert to 0-2π range
    if mean_phase < 0:
        mean_phase += 2 * np.pi
    return mean_phase

def circular_std(phases):
    """Calculate circular standard deviation"""
    if len(phases) == 0:
        return np.nan
    sin_sum = np.sum(np.sin(phases))
    cos_sum = np.sum(np.cos(phases))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(phases)
    # R ranges from 0 (uniform) to 1 (concentrated)
    # Circular std: sqrt(-2*ln(R))
    if R < 1e-10:
        return np.pi  # Maximum dispersion
    return np.sqrt(-2 * np.log(R))

def circular_distance(phase1, phase2):
    """Calculate shortest circular distance between two phases"""
    diff = np.abs(phase1 - phase2)
    return np.minimum(diff, 2*np.pi - diff)

def calculate_dip_with_outlier_removal(donor_data, outlier_threshold=np.pi/2):
    """
    Calculate DIP with iterative outlier removal
    outlier_threshold: distance threshold in radians (default π/2 = 3 hours)
    """
    phases = donor_data['Phase'].values
    celltypes = donor_data['CellType'].values
    
    if len(phases) == 0:
        return np.nan, [], []
    
    # Initial DIP
    current_dip = circular_mean(phases)
    included_mask = np.ones(len(phases), dtype=bool)
    
    # Iteratively remove outliers
    max_iterations = 5
    for iteration in range(max_iterations):
        # Calculate distance of each TIP to current DIP
        distances = np.array([circular_distance(p, current_dip) for p in phases])
        
        # Find outliers
        outliers = (distances > outlier_threshold) & included_mask
        
        if not np.any(outliers):
            break  # No more outliers
        
        # Remove outliers
        included_mask = included_mask & ~outliers
        
        # Recalculate DIP
        if np.sum(included_mask) > 0:
            current_dip = circular_mean(phases[included_mask])
        else:
            break
    
    included_celltypes = celltypes[included_mask].tolist()
    excluded_celltypes = celltypes[~included_mask].tolist()
    
    return current_dip, included_celltypes, excluded_celltypes

def load_and_process_data(result_dir):
    """Load CYCLOPS results and calculate DIP"""
    result_path = Path(result_dir)
    
    # Load predicted phases
    phase_file = list(result_path.glob("Fit_Output_*.csv"))[0]
    df = pd.read_csv(phase_file)
    df = df.rename(columns={'ID': 'Sample'})
    df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
    
    # Remove rows with NaN phases
    df = df[df['Phase'].notna()].copy()
    
    # Extract Donor ID and CellType from Sample ID
    # Format: P002_Pre.Bcell -> Donor: P002, Treatment: Pre, CellType: Bcell
    df['Donor'] = df['Sample'].str.extract(r'^(P\d+)_')[0]
    df['Treatment'] = df['Sample'].str.extract(r'_(Pre|Post)\.')[0]
    df['CellType'] = df['Sample'].str.extract(r'\.(.*?)$')[0]
    
    # Remove rows where extraction failed
    df = df.dropna(subset=['Donor', 'Treatment', 'CellType'])
    
    return df

def calculate_all_dips(df, remove_outliers=True, outlier_threshold=np.pi/2):
    """Calculate DIP for all donors"""
    results = []
    
    for donor in sorted(df['Donor'].unique()):
        donor_data = df[df['Donor'] == donor].copy()
        
        # Calculate overall DIP (across all treatments and cell types)
        if remove_outliers:
            dip, included, excluded = calculate_dip_with_outlier_removal(
                donor_data, outlier_threshold
            )
        else:
            dip = circular_mean(donor_data['Phase'].values)
            included = donor_data['CellType'].tolist()
            excluded = []
        
        dip_std = circular_std(donor_data['Phase'].values)
        n_samples = len(donor_data)
        n_included = len(included)
        n_excluded = len(excluded)
        
        results.append({
            'Donor': donor,
            'DIP': dip,
            'DIP_std': dip_std,
            'N_samples': n_samples,
            'N_included': n_included,
            'N_excluded': n_excluded,
            'Included_celltypes': ','.join(included) if included else '',
            'Excluded_celltypes': ','.join(excluded) if excluded else ''
        })
        
        # Also calculate DIP per treatment (Pre/Post)
        for treatment in sorted(donor_data['Treatment'].unique()):
            treat_data = donor_data[donor_data['Treatment'] == treatment]
            
            if remove_outliers:
                dip_treat, incl, excl = calculate_dip_with_outlier_removal(
                    treat_data, outlier_threshold
                )
            else:
                dip_treat = circular_mean(treat_data['Phase'].values)
                incl = treat_data['CellType'].tolist()
                excl = []
            
            dip_treat_std = circular_std(treat_data['Phase'].values)
            
            results.append({
                'Donor': f"{donor}_{treatment}",
                'DIP': dip_treat,
                'DIP_std': dip_treat_std,
                'N_samples': len(treat_data),
                'N_included': len(incl),
                'N_excluded': len(excl),
                'Included_celltypes': ','.join(incl) if incl else '',
                'Excluded_celltypes': ','.join(excl) if excl else ''
            })
    
    return pd.DataFrame(results)

def plot_dip_results(df, dip_df, output_dir):
    """Visualize TIP vs DIP"""
    
    # Plot 1: Donor-level comparison
    donors = [d for d in dip_df['Donor'].values if '_' not in d]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: DIP distribution
    ax = axes[0]
    dip_values = dip_df[dip_df['Donor'].isin(donors)]['DIP'].values
    
    # Convert to hours for better readability
    dip_hours = dip_values * 24 / (2 * np.pi)
    
    ax.hist(dip_hours, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('DIP (hours)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of Donor Internal Phase (n={len(donors)})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 24])
    ax.grid(True, alpha=0.3)
    
    # Right: TIP vs DIP scatter
    ax = axes[1]
    
    for donor in donors[:10]:  # Show first 10 donors to avoid clutter
        donor_data = df[df['Donor'] == donor]
        dip = dip_df[dip_df['Donor'] == donor]['DIP'].values[0]
        
        tips = donor_data['Phase'].values * 24 / (2 * np.pi)
        dip_h = dip * 24 / (2 * np.pi)
        
        # Plot TIPs
        ax.scatter([donor] * len(tips), tips, alpha=0.5, s=50)
        # Plot DIP
        ax.scatter([donor], [dip_h], color='red', marker='*', s=200, 
                  edgecolors='black', linewidths=1.5, zorder=10)
    
    ax.set_xlabel('Donor', fontsize=12)
    ax.set_ylabel('Phase (hours)', fontsize=12)
    ax.set_title('TIP (dots) vs DIP (red stars)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 24])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'DIP_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Plot 2: Circular plot of DIPs
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    dip_df_donors = dip_df[dip_df['Donor'].isin(donors)]
    
    # Plot each DIP
    for _, row in dip_df_donors.iterrows():
        dip = row['DIP']
        std = row['DIP_std']
        
        # Plot point
        ax.scatter(dip, 1, s=100, alpha=0.6, edgecolors='black', linewidths=1)
        
        # Plot uncertainty arc
        if not np.isnan(std):
            arc = np.linspace(dip - std, dip + std, 20)
            ax.plot(arc, [1]*len(arc), linewidth=3, alpha=0.3)
    
    ax.set_ylim([0, 1.2])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set labels in hours
    hour_labels = [f'{int(h)}h' for h in np.linspace(0, 24, 13)[:-1]]
    ax.set_xticklabels(hour_labels)
    ax.set_yticks([])
    
    ax.set_title(f'Donor Internal Phase Distribution (n={len(donors)})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'DIP_circular.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = "/home/rzh/zhenx/Circadian/CYCLOPS-2.0/results/Zhang_CancerCell_2025_all_20251208_170431"
    
    print(f"Loading data from: {result_dir}")
    df = load_and_process_data(result_dir)
    
    print(f"\nLoaded {len(df)} samples")
    print(f"Donors: {df['Donor'].nunique()}")
    print(f"Cell types: {df['CellType'].nunique()}")
    print(f"Treatments: {df['Treatment'].nunique()}")
    
    # Calculate DIP with outlier removal
    print("\n" + "="*60)
    print("Calculating DIP with outlier removal (threshold = π/2 ≈ 3 hours)...")
    print("="*60)
    dip_df = calculate_all_dips(df, remove_outliers=True, outlier_threshold=np.pi/2)
    
    # Save results
    output_file = Path(result_dir) / 'DIP_results.csv'
    dip_df.to_csv(output_file, index=False)
    print(f"\nSaved DIP results to: {output_file}")
    
    # Show summary
    print("\n" + "="*60)
    print("DIP Summary (Donor level):")
    print("="*60)
    donor_only = dip_df[~dip_df['Donor'].str.contains('_')].copy()
    donor_only['DIP_hours'] = donor_only['DIP'] * 24 / (2 * np.pi)
    donor_only['DIP_std_hours'] = donor_only['DIP_std'] * 24 / (2 * np.pi)
    
    print(donor_only[['Donor', 'DIP_hours', 'DIP_std_hours', 'N_samples', 
                      'N_included', 'N_excluded']].to_string(index=False))
    
    if donor_only['N_excluded'].sum() > 0:
        print("\n" + "="*60)
        print("Excluded cell types (outliers):")
        print("="*60)
        for _, row in donor_only.iterrows():
            if row['Excluded_celltypes']:
                print(f"{row['Donor']}: {row['Excluded_celltypes']}")
    
    # Plot results
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    plot_dip_results(df, dip_df, result_dir)
    
    print("\nDone!")

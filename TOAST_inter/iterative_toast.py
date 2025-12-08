"""
Iterative TOAST Training Pipeline
==================================
Reuses data loading logic from TOAST/data_load.py to ensure consistency.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import stats
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from datetime import datetime

# Add TOAST directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TOAST'))
from AE import SetPhaseAutoEncoder
from compare_with_reference import compare_predictions


def load_expression_data(data_path: str, gene_list: List[str] = None):
    """
    Load and preprocess expression data compatible with TOAST format.
    
    Args:
        data_path: Path to dataset folder
        gene_list: Optional list of genes to filter (will take intersection with available genes)
    
    Returns:
        expression_df: DataFrame with samples as rows, genes as columns
        metadata_df: DataFrame with sample metadata
        available_genes: List of genes actually present in the data
    """
    expr_file = os.path.join(data_path, 'expression.csv')
    metadata_file = os.path.join(data_path, 'metadata.csv')
    
    # Load expression data
    df = pd.read_csv(expr_file, low_memory=False)
    
    # Identify sample columns (all except Gene_Symbol)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    
    # Filter out special rows
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    
    # If gene_list provided, take intersection with available genes
    if gene_list is not None:
        # Case-insensitive matching
        gene_df['Gene_Symbol_Upper'] = gene_df['Gene_Symbol'].str.upper()
        gene_list_upper = [g.upper() for g in gene_list]
        
        # Find intersection
        available_in_data = set(gene_df['Gene_Symbol_Upper'].values)
        requested_upper = set(gene_list_upper)
        intersection = available_in_data & requested_upper
        
        # Filter to intersection
        gene_df = gene_df[gene_df['Gene_Symbol_Upper'].isin(intersection)].copy()
        gene_df = gene_df.drop(columns=['Gene_Symbol_Upper'])
        
        print(f"  Gene intersection: requested {len(gene_list)}, available {len(available_in_data)}, using {len(gene_df)}")
        
        if len(gene_df) == 0:
            raise ValueError("No genes remain after intersection!")
    
    # Get gene symbols
    gene_symbols = gene_df['Gene_Symbol'].values
    
    # Extract expression values and transpose (samples as rows, genes as columns)
    sample_df = gene_df[sample_columns].copy()
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    expression_data = sample_df.values.T  # [n_samples, n_genes]
    
    # Create DataFrame
    expression_df = pd.DataFrame(expression_data, index=sample_columns, columns=gene_symbols)
    
    # Load metadata if exists
    if os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file, index_col=0)
    else:
        metadata_df = pd.DataFrame(index=sample_columns)
    
    return expression_df, metadata_df, gene_symbols.tolist()


def calculate_gene_rhythmicity(
    expression: np.ndarray,
    phases: np.ndarray,
    method: str = 'correlation'
) -> np.ndarray:
    """
    Calculate rhythmicity score for each gene based on predicted phases.
    
    Args:
        expression: [n_samples, n_genes] expression matrix
        phases: [n_samples] predicted phase for each sample (in radians)
        method: 'correlation' or 'anova'
    
    Returns:
        rhythmicity_scores: [n_genes] score for each gene (higher = more rhythmic)
    """
    n_samples, n_genes = expression.shape
    scores = np.zeros(n_genes)
    
    if method == 'correlation':
        # Circular-linear correlation between gene expression and phase
        for i in range(n_genes):
            expr = expression[:, i]
            
            # Compute correlation with sin and cos components of phase
            sin_phase = np.sin(phases)
            cos_phase = np.cos(phases)
            
            # Pearson correlation with both components
            r_sin = np.abs(np.corrcoef(expr, sin_phase)[0, 1])
            r_cos = np.abs(np.corrcoef(expr, cos_phase)[0, 1])
            
            # Combined score (max of both)
            scores[i] = max(r_sin, r_cos)
    
    elif method == 'anova':
        # Bin samples by phase and compute ANOVA F-statistic
        n_bins = 8  # Divide 24h into 8 bins (3h each)
        phase_bins = np.digitize(phases, bins=np.linspace(0, 2*np.pi, n_bins+1))
        
        for i in range(n_genes):
            expr = expression[:, i]
            groups = [expr[phase_bins == b] for b in range(1, n_bins+1)]
            groups = [g for g in groups if len(g) > 0]  # Remove empty bins
            
            if len(groups) > 1:
                f_stat, _ = stats.f_oneway(*groups)
                scores[i] = f_stat if not np.isnan(f_stat) else 0.0
            else:
                scores[i] = 0.0
    
    return scores


def select_top_rhythmic_genes(
    expression: np.ndarray,
    phases: np.ndarray,
    gene_names: List[str],
    current_genes: List[str],
    top_k: int = 500,
    method: str = 'correlation'
) -> List[str]:
    """
    Select top-k most rhythmic genes that are not already in current_genes.
    
    Returns:
        new_genes: List of gene names to add
    """
    # Calculate rhythmicity for all genes
    rhythmicity = calculate_gene_rhythmicity(expression, phases, method=method)
    
    # Create dataframe for easier manipulation
    gene_df = pd.DataFrame({
        'gene': gene_names,
        'rhythmicity': rhythmicity
    })
    
    # Remove genes already selected
    gene_df = gene_df[~gene_df['gene'].isin(current_genes)]
    
    # Sort by rhythmicity and select top-k
    gene_df = gene_df.sort_values('rhythmicity', ascending=False)
    new_genes = gene_df.head(top_k)['gene'].tolist()
    
    print(f"  Selected {len(new_genes)} new genes")
    print(f"  Rhythmicity range: {gene_df.head(top_k)['rhythmicity'].min():.4f} - {gene_df.head(top_k)['rhythmicity'].max():.4f}")
    
    return new_genes


def train_toast_model(
    expression: np.ndarray,
    num_epochs: int = 500,
    lr: float = 1e-3,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: str = 'cuda'
) -> Tuple[nn.Module, np.ndarray]:
    """
    Train TOAST model on given expression data.
    
    Returns:
        model: Trained model
        predicted_phases: [n_samples] predicted phases in radians
    """
    n_samples, n_genes = expression.shape
    
    # Initialize model
    model = SetPhaseAutoEncoder(
        input_dim=n_genes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    # Prepare data
    X = torch.FloatTensor(expression).to(device)
    X = F.normalize(X, p=2, dim=1)
    X = X.unsqueeze(0)  # [1, n_samples, n_genes]
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        _, pred_phases, recon = model(X)
        loss = 1 - F.cosine_similarity(recon, X, dim=2).mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                phase_std = pred_phases.squeeze(0).std().item()
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Phase Std: {phase_std:.4f}")
    
    # Get final predictions
    model.eval()
    with torch.no_grad():
        _, pred_phases, _ = model(X)
        pred_phases = pred_phases.squeeze(0).cpu().numpy()
    
    return model, pred_phases


def iterative_toast_pipeline(
    data_path: str,
    num_iterations: int = 5,
    genes_per_iteration: int = 500,
    num_epochs: int = 500,
    lr: float = 1e-3,
    device: str = 'cuda',
    output_dir: str = './results_iterative'
):
    """
    Main iterative training pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"iter_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("ITERATIVE TOAST TRAINING PIPELINE")
    print("="*80)
    print(f"Data: {data_path}")
    print(f"Iterations: {num_iterations}")
    print(f"Genes added per iteration: {genes_per_iteration}")
    print(f"Epochs per iteration: {num_epochs}")
    print(f"Output: {save_dir}")
    print("="*80)
    
    # Load seed genes
    print("\n[1] Loading seed genes...")
    seed_genes_file = os.path.join(data_path, 'seed_genes.txt')
    if os.path.exists(seed_genes_file):
        with open(seed_genes_file, 'r') as f:
            seed_genes = [line.strip() for line in f if line.strip()]
        print(f"  Found {len(seed_genes)} seed genes in file")
    else:
        seed_genes = None
        print(f"  No seed_genes.txt found")
    
    # Load expression data with seed genes (taking intersection)
    print("\n[2] Loading expression data...")
    if seed_genes:
        expression_df, metadata_df, available_genes = load_expression_data(data_path, gene_list=seed_genes)
        current_genes = available_genes  # Start with intersection
    else:
        # Load all genes and select top variance as seeds
        expression_df, metadata_df, all_genes = load_expression_data(data_path, gene_list=None)
        gene_var = expression_df.var(axis=0)
        current_genes = gene_var.nlargest(100).index.tolist()
        available_genes = all_genes
        print(f"  Using top 100 variance genes as seed")
    
    print(f"  Expression shape: {expression_df.shape[0]} samples x {expression_df.shape[1]} genes")
    print(f"  Starting with {len(current_genes)} genes")
    
    # For rhythmicity calculation, we need all genes - reload without filter
    full_expression_df, _, all_gene_names = load_expression_data(data_path, gene_list=None)
    full_expression = full_expression_df.values  # [n_samples, n_all_genes]
    
    sample_names = expression_df.index.tolist()
    n_samples = len(sample_names)
    
    # Get true phases and tissue labels (if available)
    true_phases = metadata_df['Time_Phase'].values if 'Time_Phase' in metadata_df.columns else (
        metadata_df['phase'].values if 'phase' in metadata_df.columns else None
    )
    tissue_labels = metadata_df['tissue'].values if 'tissue' in metadata_df.columns else (
        metadata_df['organ'].values if 'organ' in metadata_df.columns else np.array(['unknown'] * n_samples)
    )
    iteration_results = []
    
    # Iterative training
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*80}")
        print(f"Current gene set size: {len(current_genes)}")
        
        # Get expression for current genes (reload with current gene list)
        current_expr_df, _, _ = load_expression_data(data_path, gene_list=current_genes)
        current_expression = current_expr_df.values  # [n_samples, n_current_genes]
        
        print(f"  Loaded expression: {current_expression.shape[0]} samples x {current_expression.shape[1]} genes")
        
        # Train model
        print(f"\n[{iteration}.1] Training TOAST model...")
        model, pred_phases = train_toast_model(
            expression=current_expression,
            num_epochs=num_epochs,
            lr=lr,
            device=device
        )
        
        # Save model
        model_path = os.path.join(save_dir, f'model_iter{iteration}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'sample': sample_names,
            'pred_phase_rad': pred_phases,
            'pred_phase_hours': (pred_phases / (2 * np.pi)) * 24
        })
        if true_phases is not None:
            pred_df['true_phase_rad'] = true_phases
            pred_df['true_phase_hours'] = (true_phases / (2 * np.pi)) * 24
        
        pred_path = os.path.join(save_dir, f'predictions_iter{iteration}.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved: {pred_path}")
        
        # Generate comparison plot using TOAST's compare_with_reference
        if true_phases is not None:
            print(f"\n[{iteration}.2] Generating comparison plot...")
            
            # Prepare metadata for comparison (same format as TOAST's metadata.csv)
            reference_df = pd.DataFrame({
                'Sample': sample_names,
                'Time_Phase': true_phases,
                'Hour_in_24': (true_phases / (2 * np.pi)) * 24
            })
            reference_path = os.path.join(save_dir, f'reference_iter{iteration}.csv')
            reference_df.to_csv(reference_path, index=False)
            
            # Use TOAST's compare_predictions function
            plot_dir = os.path.join(save_dir, f'comparison_iter{iteration}')
            try:
                compare_predictions(pred_path, reference_path, plot_dir)
                print(f"  Comparison plot saved: {plot_dir}/")
            except Exception as e:
                print(f"  Warning: Failed to generate comparison plot: {e}")
        
        # Store results
        iteration_results.append({
            'iteration': iteration,
            'n_genes': len(current_genes),
            'model_path': model_path,
            'pred_path': pred_path,
            'genes': current_genes.copy()
        })
        
        # Select new genes (except for last iteration)
        if iteration < num_iterations:
            print(f"\n[{iteration}.3] Selecting new rhythmic genes...")
            new_genes = select_top_rhythmic_genes(
                expression=full_expression,
                phases=pred_phases,
                gene_names=all_gene_names,
                current_genes=current_genes,
                top_k=genes_per_iteration,
                method='correlation'
            )
            
            current_genes.extend(new_genes)
            
            # Save current gene list
            genes_path = os.path.join(save_dir, f'genes_iter{iteration+1}.txt')
            with open(genes_path, 'w') as f:
                f.write('\n'.join(current_genes))
            print(f"  Gene list saved: {genes_path}")
    
    # Save summary
    summary_df = pd.DataFrame([
        {'iteration': r['iteration'], 'n_genes': r['n_genes']} 
        for r in iteration_results
    ])
    summary_path = os.path.join(save_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("ITERATIVE TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"All results saved to: {save_dir}")
    print(f"\nGene count progression:")
    for r in iteration_results:
        print(f"  Iteration {r['iteration']}: {r['n_genes']} genes")
    
    return iteration_results, save_dir


def main():
    parser = argparse.ArgumentParser(description='Iterative TOAST Training')
    parser.add_argument('--data_path', required=True, help='Path to dataset folder')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations')
    parser.add_argument('--genes_per_iteration', type=int, default=500, 
                       help='Number of genes to add each iteration')
    parser.add_argument('--num_epochs', type=int, default=500, help='Epochs per iteration')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', default='./results_iterative', help='Output directory')
    
    args = parser.parse_args()
    
    iterative_toast_pipeline(
        data_path=args.data_path,
        num_iterations=args.num_iterations,
        genes_per_iteration=args.genes_per_iteration,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

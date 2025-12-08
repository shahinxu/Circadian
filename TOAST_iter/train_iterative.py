#!/usr/bin/env python3
"""
Iterative TOAST training with gene selection
Train TOAST → predict phases → select rhythmic genes → add to seed genes → repeat
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from AE import SetPhaseAutoEncoder
from data_load import load_and_preprocess_train_data
from utils import align_predictions_to_gene_acrophases, plot_comparsion
from datetime import datetime
import torch.nn.functional as F
from scipy import stats


def train_model_iteration(
    model: nn.Module,
    train_dataset,
    num_epochs=1000,
    lr=1e-3,
    device='cuda'
):
    """Train model for one iteration"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    
    all_expressions = [train_dataset[i]['expression'] for i in range(len(train_dataset))]
    X_all = torch.stack(all_expressions).to(device)
    X_all = X_all / (torch.norm(X_all, dim=1, keepdim=True) + 1e-8)
    X_input = X_all.unsqueeze(0)
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        _, pred_phases, recon = model(X_input)
        loss = 1 - F.cosine_similarity(recon, X_input, dim=2).mean()
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        scheduler.step(loss_val)
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4f}")
    
    # Get final predictions
    model.eval()
    with torch.no_grad():
        _, pred_phases, recon = model(X_input)
    pred_phases_np = pred_phases.squeeze(0).detach().cpu().numpy()
    
    return model, pred_phases_np


def calculate_gene_rhythmicity(expr_matrix, phases, method='correlation'):
    """
    Calculate rhythmicity score for each gene
    
    Args:
        expr_matrix: (n_samples, n_genes) expression matrix
        phases: (n_samples,) predicted phases in radians
        method: 'correlation' (circular-linear correlation) or 'anova' (F-statistic)
    
    Returns:
        rhythmicity_scores: (n_genes,) rhythmicity score for each gene
    """
    n_samples, n_genes = expr_matrix.shape
    scores = np.zeros(n_genes)
    
    if method == 'correlation':
        # Circular-linear correlation: max of |corr(expr, sin(phase))| and |corr(expr, cos(phase))|
        sin_phase = np.sin(phases)
        cos_phase = np.cos(phases)
        
        for g in range(n_genes):
            expr = expr_matrix[:, g]
            if np.std(expr) < 1e-8:
                scores[g] = 0
                continue
            
            # Correlation with sin and cos
            r_sin = abs(np.corrcoef(expr, sin_phase)[0, 1])
            r_cos = abs(np.corrcoef(expr, cos_phase)[0, 1])
            
            # Use max as rhythmicity score
            scores[g] = max(r_sin, r_cos)
    
    elif method == 'anova':
        # ANOVA F-statistic: test if expression differs across phase bins
        n_bins = 8
        phase_bins = np.linspace(0, 2*np.pi, n_bins + 1)
        bin_indices = np.digitize(phases, phase_bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        for g in range(n_genes):
            expr = expr_matrix[:, g]
            if np.std(expr) < 1e-8:
                scores[g] = 0
                continue
            
            # Group expression by phase bins
            groups = [expr[bin_indices == b] for b in range(n_bins)]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                scores[g] = 0
                continue
            
            # One-way ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            scores[g] = f_stat if not np.isnan(f_stat) else 0
    
    return scores


def select_top_rhythmic_genes(
    expr_file,
    phases,
    current_genes,
    top_k=100,
    method='correlation'
):
    """
    Select top-k most rhythmic genes not already in current set
    
    Args:
        expr_file: path to expression.csv
        phases: predicted phases for samples
        current_genes: set of genes already selected
        top_k: number of new genes to add
        method: rhythmicity calculation method
    
    Returns:
        new_genes: list of newly selected gene symbols
        all_scores_df: DataFrame with all genes and their scores
    """
    # Load full expression matrix
    expr_df = pd.read_csv(expr_file, index_col=0)
    
    # Get all gene symbols and expression
    all_genes = expr_df.index.tolist()
    expr_matrix = expr_df.values.T  # Transpose to (n_samples, n_genes)
    
    # Calculate rhythmicity for all genes
    print(f"  Calculating rhythmicity for {len(all_genes)} genes...")
    scores = calculate_gene_rhythmicity(expr_matrix, phases, method=method)
    
    # Create DataFrame with results
    scores_df = pd.DataFrame({
        'Gene': all_genes,
        'Rhythmicity_Score': scores,
        'Already_Selected': [g in current_genes for g in all_genes]
    })
    
    # Sort by score (descending)
    scores_df = scores_df.sort_values('Rhythmicity_Score', ascending=False)
    
    # Select top-k genes not already in current set
    available = scores_df[~scores_df['Already_Selected']]
    new_genes = available.head(top_k)['Gene'].tolist()
    
    print(f"  Top rhythmicity score: {scores_df['Rhythmicity_Score'].iloc[0]:.4f}")
    print(f"  Selected {len(new_genes)} new genes (top scores: {available.head(3)['Rhythmicity_Score'].tolist()})")
    
    return new_genes, scores_df


def filter_expression_by_genes(expr_file, gene_list, output_file):
    """Filter expression.csv to only include specified genes"""
    expr_df = pd.read_csv(expr_file, index_col=0)
    
    # Find intersection of requested genes and available genes
    available_genes = set(expr_df.index.str.upper())
    requested_genes_upper = set([g.upper() for g in gene_list])
    
    # Match genes (case-insensitive)
    gene_mapping = {g.upper(): g for g in expr_df.index}
    found_genes = []
    for req_gene in gene_list:
        req_upper = req_gene.upper()
        if req_upper in available_genes:
            found_genes.append(gene_mapping[req_upper])
    
    print(f"  Requested {len(gene_list)} genes, found {len(found_genes)} in data")
    
    if len(found_genes) == 0:
        raise ValueError("No requested genes found in expression data!")
    
    # Filter and save
    filtered_df = expr_df.loc[found_genes]
    filtered_df.to_csv(output_file)
    
    return found_genes


def iterative_toast_pipeline(
    dataset_path,
    seed_genes_file,
    n_iterations=5,
    genes_per_iteration=100,
    num_epochs=1000,
    n_components=5,
    lr=0.001,
    dropout=0.1,
    device='cuda',
    rhythmicity_method='correlation',
    random_seed=42
):
    """
    Main iterative training pipeline
    
    Args:
        dataset_path: path to dataset folder (relative to ../data/)
        seed_genes_file: path to initial seed genes file
        n_iterations: number of iterations
        genes_per_iteration: number of genes to add each iteration
        num_epochs: training epochs per iteration
        n_components: number of PCA components
        lr: learning rate
        dropout: dropout rate
        device: 'cuda' or 'cpu'
        rhythmicity_method: 'correlation' or 'anova'
        random_seed: random seed
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    base_data = f"../data/{dataset_path}"
    expr_file = os.path.join(base_data, "expression.csv")
    metadata_file = os.path.join(base_data, "metadata.csv")
    
    # Create output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", dataset_path, f"iterative_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load initial seed genes
    with open(seed_genes_file, 'r') as f:
        seed_genes = [line.strip() for line in f if line.strip()]
    
    print(f"\n{'='*60}")
    print(f"Starting Iterative TOAST Training")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Initial seed genes: {len(seed_genes)}")
    print(f"Iterations: {n_iterations}")
    print(f"Genes per iteration: {genes_per_iteration}")
    print(f"Epochs per iteration: {num_epochs}")
    print(f"{'='*60}\n")
    
    # Mouse reference for alignment
    mouse_acrophases = [0, 0.0790637050481884, 0.151440116812406, 2.29555301890004, 2.90900605826091,
                       2.98706493493206, 2.99149022777511, 3.00769248308471, 3.1219769314524,
                       3.3058682224604, 3.31357155959037, 3.42557704861225, 3.50078722833753,
                       3.88658015146741, 4.99480367551318, 5.04951134876313, 6.00770260397838]
    mouse_gene_symbol = ["ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2",
                        "DBP", "CIART", "PER1", "PER3", "TEF", "HLF",
                        "CRY2", "PER2", "CRY1", "RORC", "NFIL3"]
    
    current_genes = set(seed_genes)
    all_iteration_results = []
    
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        print(f"Current gene set size: {len(current_genes)}")
        
        iter_dir = os.path.join(save_dir, f"iteration_{iteration+1}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Filter expression file for current genes
        filtered_expr = os.path.join(iter_dir, "expression_filtered.csv")
        found_genes = filter_expression_by_genes(expr_file, list(current_genes), filtered_expr)
        
        # Load and preprocess data
        print(f"  Loading data...")
        train_dataset, preprocessing_info = load_and_preprocess_train_data(
            filtered_expr, n_components=n_components
        )
        
        # Create and train model
        print(f"  Training model...")
        model = SetPhaseAutoEncoder(
            input_dim=preprocessing_info['n_components'],
            dropout=dropout
        )
        
        model, pred_phases = train_model_iteration(
            model, train_dataset, num_epochs=num_epochs, lr=lr, device=device
        )
        
        # Save predictions
        sample_names = preprocessing_info.get('sample_columns', [])
        pred_hours = (pred_phases / (2 * np.pi)) * 24.0
        
        results_df = pd.DataFrame({
            'Sample_ID': sample_names,
            'pred_phase_rad': pred_phases,
            'Predicted_Phase_Hours': pred_hours
        })
        
        # Align to mouse reference
        try:
            aligned_df, shift_rad, per_gene_df = align_predictions_to_gene_acrophases(
                results_df=results_df,
                test_expr_file=filtered_expr,
                gene_symbols=mouse_gene_symbol,
                ref_acrophases_rad=mouse_acrophases
            )
            results_df['Aligned_Phase_Hours'] = (aligned_df['pred_phase_rad'] / (2 * np.pi)) * 24.0
            print(f"  Alignment shift: {shift_rad:.4f} rad")
        except Exception as e:
            print(f"  Warning: Alignment failed: {e}")
            results_df['Aligned_Phase_Hours'] = results_df['Predicted_Phase_Hours']
        
        # Save iteration results
        results_path = os.path.join(iter_dir, 'predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        # Calculate phase statistics
        phase_std = np.std(pred_phases)
        phase_range = np.max(pred_phases) - np.min(pred_phases)
        
        print(f"  Phase std: {phase_std:.4f} rad ({phase_std * 24 / (2*np.pi):.2f} hours)")
        print(f"  Phase range: {phase_range:.4f} rad ({phase_range * 24 / (2*np.pi):.2f} hours)")
        
        # Plot comparison with metadata if available
        if os.path.exists(metadata_file):
            print(f"\n  Plotting comparison with metadata...")
            try:
                comparison_result = plot_comparsion(results_df, metadata_file, iter_dir)
                if comparison_result:
                    _, r, r2, spearman_R = comparison_result
                    print(f"  Comparison: Pearson r={r:.4f}, Spearman ρ={spearman_R:.4f}")
            except Exception as e:
                print(f"  Warning: Comparison plot failed: {e}")
        
        # Select new genes for next iteration starting from iteration 2
        if iteration >= 1 and iteration < n_iterations - 1:
            print(f"\n  Selecting top {genes_per_iteration} rhythmic genes...")
            
            new_genes, all_scores_df = select_top_rhythmic_genes(
                expr_file=expr_file,
                phases=pred_phases,
                current_genes=current_genes,
                top_k=genes_per_iteration,
                method=rhythmicity_method
            )
            
            # Save rhythmicity scores
            scores_path = os.path.join(iter_dir, 'gene_rhythmicity_scores.csv')
            all_scores_df.to_csv(scores_path, index=False)
            
            # Update gene set
            current_genes.update(new_genes)
            
            # Save updated gene list
            genes_path = os.path.join(iter_dir, 'selected_genes.txt')
            with open(genes_path, 'w') as f:
                for gene in sorted(current_genes):
                    f.write(f"{gene}\n")
        elif iteration == n_iterations - 1:
            # Last iteration: still calculate rhythmicity but don't add genes
            print(f"\n  Calculating rhythmicity scores (final iteration)...")
            _, all_scores_df = select_top_rhythmic_genes(
                expr_file=expr_file,
                phases=pred_phases,
                current_genes=current_genes,
                top_k=0,  # Don't select new genes
                method=rhythmicity_method
            )
            scores_path = os.path.join(iter_dir, 'gene_rhythmicity_scores.csv')
            all_scores_df.to_csv(scores_path, index=False)
        
        # Record iteration summary
        all_iteration_results.append({
            'Iteration': iteration + 1,
            'N_genes': len(found_genes),
            'Phase_std_rad': phase_std,
            'Phase_std_hours': phase_std * 24 / (2*np.pi),
            'Phase_range_rad': phase_range,
            'Phase_range_hours': phase_range * 24 / (2*np.pi)
        })
    
    # Save summary
    summary_df = pd.DataFrame(all_iteration_results)
    summary_path = os.path.join(save_dir, 'iteration_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Iterative Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}")
    print(f"\nIteration Summary:")
    print(summary_df.to_string(index=False))
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Iterative TOAST training with gene selection')
    parser.add_argument("--dataset_path", required=True, help="Dataset path relative to ../data/")
    parser.add_argument("--seed_genes", required=True, help="Path to initial seed genes file")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--genes_per_iteration", type=int, default=100, help="Genes to add per iteration")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Training epochs per iteration")
    parser.add_argument("--n_components", type=int, default=5, help="Number of PCA components")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", default='cuda', help="Device: cuda or cpu")
    parser.add_argument("--rhythmicity_method", default='correlation', 
                       choices=['correlation', 'anova'], 
                       help="Method to calculate gene rhythmicity")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, using CPU")
    
    iterative_toast_pipeline(
        dataset_path=args.dataset_path,
        seed_genes_file=args.seed_genes,
        n_iterations=args.n_iterations,
        genes_per_iteration=args.genes_per_iteration,
        num_epochs=args.num_epochs,
        n_components=args.n_components,
        lr=args.lr,
        dropout=args.dropout,
        device=args.device,
        rhythmicity_method=args.rhythmicity_method,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()

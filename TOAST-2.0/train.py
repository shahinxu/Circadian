import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
from pathway_model import PathwayAutoencoderWithTissue
from data_load import load_and_preprocess_train_data, load_and_preprocess_test_data
from utils import align_predictions_to_gene_acrophases
from datetime import datetime
import torch.nn.functional as F
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed. Run: pip install wandb")


def train_model(
    model: nn.Module, 
    train_dataset, 
    preprocessing_info,
    num_epochs=100, 
    lr=1e-3, 
    device='cuda',
    save_dir='./model_checkpoints',
    use_wandb=False
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    
    os.makedirs(save_dir, exist_ok=True)
    train_losses = []
    all_expressions = [train_dataset[i]['expression'] for i in range(len(train_dataset))]
    X_all = torch.stack(all_expressions).to(device)
    X_all = X_all / (torch.norm(X_all, dim=1, keepdim=True) + 1e-8)
    X_input = X_all.unsqueeze(0)
    
    batch_size, n_samples = 1, X_all.shape[0]
    tissue_idx = torch.zeros(batch_size, n_samples, dtype=torch.long).to(device)

    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        _, pred_phases, recon = model(X_input, tissue_idx)
            
        loss = 1 - F.cosine_similarity(recon, X_input, dim=2).mean()
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        train_losses.append(loss_val)
        scheduler.step(loss_val)
        
        # Compute metrics
        with torch.no_grad():
            phases_np = pred_phases.squeeze(0).cpu().numpy()
            phase_std = np.std(phases_np)
            phase_range = phases_np.max() - phases_np.min()
            
            # Circular variance (0=all same, 1=uniform)
            complex_phases = np.exp(1j * phases_np)
            circular_mean = np.abs(complex_phases.mean())
            circular_variance = 1 - circular_mean
            
            # Phase coherence (consecutive phase differences)
            sorted_phases = np.sort(phases_np)
            phase_diffs = np.diff(sorted_phases)
            phase_coherence = 1.0 / (1.0 + np.std(phase_diffs))
            
            # Reconstruction MSE
            mse = F.mse_loss(recon, X_input).item()
            
        metrics = {
            'epoch': epoch + 1,
            'loss/cosine': loss_val,
            'loss/mse': mse,
            'phase/std': phase_std,
            'phase/range': phase_range,
            'phase/circular_variance': circular_variance,
            'phase/coherence': phase_coherence,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Cosine Loss: {loss_val:.4f}, "
                  f"MSE: {mse:.4f}, Phase Std: {phase_std:.4f}, Circ Var: {circular_variance:.4f}")

    return model


def evaluate_test_set(
    model, 
    test_file, 
    preprocessing_info, 
    save_dir, 
    device='cuda', 
    metadata_path=None
):
    if not os.path.isfile(test_file):
        print("No test file provided; skipping test set evaluation.")
        return None

    print("Loading test set for Set Transformer evaluation...")
    
    test_dataset, test_preprocessing_info = load_and_preprocess_test_data(
        test_file, preprocessing_info
    )

    all_expressions = [test_dataset[i]['expression'] for i in range(len(test_dataset))]
    X_test = torch.stack(all_expressions).to(device)
    X_test = X_test / (torch.norm(X_test, dim=1, keepdim=True) + 1e-8)
    X_input = X_test.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        batch_size, n_samples = 1, X_test.shape[0]
        tissue_idx = torch.zeros(batch_size, n_samples, dtype=torch.long).to(device)
        
        _, pred_phases, _ = model(X_input, tissue_idx)
            
    pred_phases_np = pred_phases.squeeze(0).cpu().numpy()
    
    period = preprocessing_info.get('period_hours', 24.0)
    pred_hours = (pred_phases_np / (2 * np.pi)) * period
    
    sample_names = test_preprocessing_info.get('sample_columns', [])
    
    if len(sample_names) != len(pred_hours):
        print(f"[WARN] Sample names count ({len(sample_names)}) != Predictions count ({len(pred_hours)}). Generating dummy IDs.")
        sample_names = [f"Sample_{i}" for i in range(len(pred_hours))]

    results_df = pd.DataFrame({
        'Sample_ID': sample_names,
        'pred_phase_rad': pred_phases_np,
        'Predicted_Phase_Hours': pred_hours
    })

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Test predictions saved to: {results_path}")

    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--align_gene_symbols", type=str, default=None)
    parser.add_argument("--align_acrophases", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--pathway_csv", type=str, default="../pathway_dataset/dataset.csv",
                        help="Path to pathway dataset CSV")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dimension for pathway model")
    parser.add_argument("--wandb", action='store_true', default=True,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no_wandb", action='store_false', dest='wandb',
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="circadian-pathway",
                        help="W&B project name")
    args = parser.parse_args()

    base_data = f"../data/{args.dataset_path}"
    train_file = os.path.join(base_data, "expression.csv")
    metadata = os.path.join(base_data, "metadata.csv")
    ts = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    save_dir = os.path.join("results", args.dataset_path, ts)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    else:
        if args.device == 'cuda':
            print(f"Using device: {args.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"Using device: {args.device}")

    os.makedirs(save_dir, exist_ok=True)

    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        train_file,
        pathway_csv=args.pathway_csv
    )

    preprocessing_info['period_hours'] = args.period_hours

    pathway_info = preprocessing_info.get('pathway_info', None)
    
    if pathway_info is None:
        raise ValueError("Pathway information not found! Must use --use_pathway_model with --pathway_csv")
    
    print("="*50)
    print("Pathway-Based Attention Model")
    print("="*50)
    
    model = PathwayAutoencoderWithTissue(
        input_dim=preprocessing_info['input_dim'],
        pathway_map=pathway_info['pathway_indices'],
        num_tissues=1,
        embed_dim=args.embed_dim,
        dropout=args.dropout
    )
    print(f"Initialized: {len(pathway_info['pathway_names'])} pathways, {args.embed_dim}D embeddings")

    if args.wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.dataset_path.replace('/', '_')}",
            config={
                'dataset': args.dataset_path,
                'num_epochs': args.num_epochs,
                'lr': args.lr,
                'embed_dim': args.embed_dim,
                'dropout': args.dropout,
                'num_pathways': len(pathway_info['pathway_names']),
                'num_genes': preprocessing_info['input_dim']
            }
        )
        print("\n" + "="*70)
        print(f"📊 W&B Dashboard: {run.get_url()}")
        print("="*70 + "\n")

    model = train_model(
        model,
        train_dataset=train_dataset,
        preprocessing_info=preprocessing_info,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        save_dir=save_dir,
        use_wandb=args.wandb
    )
    
    # Save model checkpoint
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessing_info': preprocessing_info,
        'config': {
            'embed_dim': args.embed_dim,
            'dropout': args.dropout,
            'num_pathways': len(pathway_info['pathway_names']),
            'num_genes': preprocessing_info['input_dim']
        }
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("Generating final predictions and comparison plot in main save_dir...")
    results_df = None
    try:
        results_df = evaluate_test_set(
            model=model,
            test_file=train_file,
            preprocessing_info=preprocessing_info,
            save_dir=save_dir,
            device=args.device,
            metadata_path=metadata
        )
    except Exception as e:
        print(f"[WARN] Failed to generate final prediction plot: {e}")
    
    # Generate comparison plot with metadata
    predictions_csv = os.path.join(save_dir, 'predictions.csv')
    if os.path.exists(predictions_csv) and os.path.exists(metadata):
        print("\nGenerating phase comparison plots...")
        try:
            from utils import plot_comparsion
            import pandas as pd
            results_df = pd.read_csv(predictions_csv)
            summary_df = plot_comparsion(results_df, metadata, save_dir)
            if summary_df is not None:
                print("\n=== Comparison Summary ===")
                print(summary_df[['Tissue', 'N', 'Pearson_R', 'Spearman_R', 'Circular_R']].to_string(index=False))
        except Exception as e:
            print(f"[WARN] Failed to generate comparison plots: {e}")
            import traceback
            traceback.print_exc()

    if not os.path.isfile(metadata):
        try:
            mouse_acrophases = [0, 0.0790637050481884, 0.151440116812406, 2.29555301890004, 2.90900605826091, 
                                2.98706493493206, 2.99149022777511, 3.00769248308471, 3.1219769314524, 
                                3.3058682224604, 3.31357155959037, 3.42557704861225, 3.50078722833753, 
                                3.88658015146741, 4.99480367551318, 5.04951134876313, 6.00770260397838]
            mouse_gene_symbol = ["Arntl", "Clock", "Npas2", "Nr1d1", "Bhlhe41", "Nr1d2", 
                                "Dbp", "Ciart", "Per1", "Per3", "Tef", "Hlf", 
                                "Cry2", "Per2", "Cry1", "Rorc", "Nfil3"]
            preds_csv = os.path.join(save_dir, 'predictions.csv')
            if os.path.isfile(preds_csv):
                try:
                    aligned_df, shift_rad, per_gene_df = align_predictions_to_gene_acrophases(
                        results_df=pd.read_csv(preds_csv),
                        test_expr_file=train_file,
                        gene_symbols=mouse_gene_symbol,
                        ref_acrophases_rad=mouse_acrophases
                    )
                    aligned_csv = os.path.join(save_dir, 'predictions_aligned.csv')
                    aligned_df.to_csv(aligned_csv, index=False)
                    per_gene_csv = os.path.join(save_dir, 'alignment_gene_summary.csv')
                    per_gene_df.to_csv(per_gene_csv, index=False)
                    print(f"No metadata present — applied mouse-based alignment (shift={shift_rad:.4f} rad). Saved: {aligned_csv}")
                except Exception as e:
                    print(f"[WARN] Failed to perform automatic mouse-based alignment: {e}")
            else:
                # If predictions_df already returned from evaluate_test_set, use that
                if results_df is not None:
                    try:
                        aligned_df, shift_rad, per_gene_df = align_predictions_to_gene_acrophases(
                            results_df=results_df,
                            test_expr_file=train_file,
                            gene_symbols=mouse_gene_symbol,
                            ref_acrophases_rad=mouse_acrophases
                        )
                        aligned_csv = os.path.join(save_dir, 'predictions_aligned.csv')
                        aligned_df.to_csv(aligned_csv, index=False)
                        per_gene_csv = os.path.join(save_dir, 'alignment_gene_summary.csv')
                        per_gene_df.to_csv(per_gene_csv, index=False)
                        print(f"No metadata present — applied mouse-based alignment (shift={shift_rad:.4f} rad). Saved: {aligned_csv}")
                    except Exception as e:
                        print(f"[WARN] Failed to perform automatic mouse-based alignment: {e}")
        except Exception as e:
            print(f"[WARN] Unexpected error while attempting automatic alignment: {e}")

    if args.align_gene_symbols and args.align_acrophases:
        try:
            gene_list = [g.strip() for g in args.align_gene_symbols.split(',') if g.strip()]
            acro_list = [float(x) for x in args.align_acrophases.split(',') if x.strip()]
            if len(gene_list) != len(acro_list):
                print('[WARN] align_gene_symbols and align_acrophases lengths differ; skipping alignment')
            else:
                preds_csv = os.path.join(save_dir, 'predictions.csv')
                if os.path.isfile(preds_csv):
                    preds_df = pd.read_csv(preds_csv)
                elif results_df is not None:
                    preds_df = results_df
                else:
                    preds_df = None
                if preds_df is None:
                    print(f"[WARN] No predictions available to align; skipping")
                else:
                    aligned_df, shift_rad, per_gene_df = align_predictions_to_gene_acrophases(preds_df, train_file, gene_list, acro_list)
                    aligned_csv = os.path.join(save_dir, 'predictions_aligned.csv')
                    aligned_df.to_csv(aligned_csv, index=False)
                    per_gene_csv = os.path.join(save_dir, 'alignment_gene_summary.csv')
                    per_gene_df.to_csv(per_gene_csv, index=False)
                    print(f"Alignment applied (shift={shift_rad:.4f} rad). Aligned predictions: {aligned_csv}; per-gene summary: {per_gene_csv}")
        except Exception as e:
            print(f"[WARN] Failed to perform gene-based alignment: {e}")

    print("Training completed. Final ordering obtained from last epoch.")


if __name__ == '__main__':
    main()

from utils import plot_comparsion
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
from AE import SetPhaseAutoEncoder
from data_load import load_and_preprocess_train_data, load_and_preprocess_test_data
from torch.utils.data import DataLoader
from utils import predict_and_save_phases
from utils import align_predictions_to_gene_acrophases
from datetime import datetime
import torch.nn.functional as F

def plot_express(components, phases, save_path, n_plot=None):
    order = np.argsort(phases)
    phases_sorted = phases[order]
    comp_sorted = components[order]
    n_components = comp_sorted.shape[1]
    if n_plot is None:
        n_plot = n_components
    plt.figure(figsize=(8, 6))
    for i in range(n_plot):
        plt.plot(phases_sorted, comp_sorted[:, i], label=f'PC{i+1}')
    plt.xlabel('Predicted phase')
    plt.title('Expression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_model(
    model: nn.Module, 
    train_dataset, 
    preprocessing_info,
    num_epochs=100, 
    lr=1e-3, 
    device='cuda',
    save_dir='./model_checkpoints'
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

    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        _, pred_phases, recon = model(X_input)
        loss = 1 - F.cosine_similarity(recon, X_input, dim=2).mean()
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        train_losses.append(loss_val)
        scheduler.step(loss_val)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Cosine Loss: {loss_val:.4f}")

    model.eval()
    with torch.no_grad():
        _, pred_phases, recon = model(X_input)
    pred_phases_np = pred_phases.squeeze(0).detach().cpu().numpy()
    pred_order_relative = np.argsort(pred_phases_np)
    
    return model, pred_order_relative


def evaluate_order_plot(
        pred_order, 
        preprocessing_info, 
        metadata_path, 
        save_dir, 
        period_hours=24.0
    ):
    if not os.path.isfile(metadata_path):
        return None

    n_samples = len(pred_order)
    pred_hours = (np.arange(n_samples) / n_samples) * period_hours
    sample_names = preprocessing_info.get('sample_columns', [])
    order_df = pd.DataFrame({
        'Sample_ID': sample_names,
        'Predicted_Order': pred_order,
        'Predicted_Phase_Hours': pred_hours
    })
    return plot_comparsion(order_df, metadata_path, save_dir)


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
        _, pred_phases, recon = model(X_input)
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

    if metadata_path and os.path.isfile(metadata_path):
        plot_comparsion(results_df, metadata_path, save_dir)

    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--align_gene_symbols", type=str, default=None)
    parser.add_argument("--align_acrophases", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    base_data = f"../data/{args.dataset_path}"
    train_file = os.path.join(base_data, "expression.csv")
    metadata = os.path.join(base_data, "metadata.csv")
    ts = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    save_dir = os.path.join("results", args.dataset_path, ts)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    os.makedirs(save_dir, exist_ok=True)

    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        train_file, n_components=args.n_components
    )

    preprocessing_info['period_hours'] = args.period_hours

    model = SetPhaseAutoEncoder(
        input_dim=preprocessing_info['n_components'],
        dropout=args.dropout
    )

    model, pred_order = train_model(
        model,
        train_dataset=train_dataset,
        preprocessing_info=preprocessing_info,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        save_dir=save_dir,
    )

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

    # Always perform acrophase-based alignment (do not use metadata time info - that's cheating)
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
                print(f"Applied acrophase-based alignment (shift={shift_rad:.4f} rad). Saved: {aligned_csv}")
            except Exception as e:
                print(f"[WARN] Failed to perform acrophase-based alignment: {e}")
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
                    print(f"Applied acrophase-based alignment (shift={shift_rad:.4f} rad). Saved: {aligned_csv}")
                except Exception as e:
                    print(f"[WARN] Failed to perform acrophase-based alignment: {e}")
    except Exception as e:
        print(f"[WARN] Unexpected error while attempting acrophase alignment: {e}")

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

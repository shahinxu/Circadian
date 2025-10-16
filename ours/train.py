def plot_decoder_fits(model, n_components, save_path):
    model.eval()
    phases = torch.linspace(0, 2 * np.pi, 200)
    phases = phases.to(next(model.parameters()).device) 
    with torch.no_grad():
        decoded = model.decode(phases)  # (200, n_components)
        decoded = decoded.cpu().numpy()
    plt.figure(figsize=(8, 5))
    for i in range(n_components):
        plt.plot(phases.cpu().numpy(), decoded[:, i], label=f'PC{i+1}')
    plt.xlabel('Phase (radian)')
    plt.ylabel('Decoder Output')
    plt.title('Decoder Output vs Phase')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Decoder拟合曲线已保存: {save_path}")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from AE import PhaseAutoEncoder
from utils import predict_and_save_phases
from dataset import load_and_preprocess_train_data, load_and_preprocess_test_data

def time_supervision_loss(phase_angles, true_times, lambda_time=1.0, period_hours=24.0):
    if true_times is None:
        return torch.tensor(0.0, device=phase_angles.device)
    
    true_phases = (2 * np.pi * true_times) / period_hours
    true_phases = torch.remainder(true_phases, 2 * np.pi)
    pred_phases = phase_angles
    
    phase_diff = torch.abs(pred_phases - true_phases)
    phase_diff = torch.min(phase_diff, 2*np.pi - phase_diff)
    
    return lambda_time * torch.mean(phase_diff)


def train_model(model, train_dataset, preprocessing_info, 
                num_epochs=100, lr=0.001, device='cuda',
                lambda_recon=1.0, lambda_time=0.5,
                period_hours=24.0, save_dir='./model_checkpoints', lambda_cyc=1.0):
    
    if 'train_has_time' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_time'] = 'time' in sample
    
    if 'train_has_celltype' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_celltype'] = 'celltype' in sample
    
    model = model.to(device)

    # Use AdamW with small weight decay and gradient clipping to stabilize training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Use ReduceLROnPlateau to reduce LR on plateau (better for noisy losses)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    
    recon_criterion = nn.MSELoss()
    
    train_losses = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    # prepare training tensors
    all_expressions = []
    all_times = []
    all_celltypes = []
    valid_mask = []
    
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_expressions.append(sample['expression'])
        
        if 'celltype' in sample and sample['celltype'] == 'PADDING':
            is_valid = False
        else:
            is_valid = True
        valid_mask.append(is_valid)
        
        if 'time' in sample:
            all_times.append(sample['time'])
        if 'celltype' in sample:
            all_celltypes.append(sample['celltype'])
    
    expressions_tensor = torch.stack(all_expressions).to(device)
    valid_mask_tensor = torch.tensor(valid_mask, device=device)
    
    times_tensor = None
    if all_times:
        times_tensor = torch.stack(all_times).to(device)
    
    celltypes_array = None
    if all_celltypes:
        celltypes_array = np.array(all_celltypes)
    
    celltype_indices_tensor = None
    if preprocessing_info['train_has_celltype'] and celltypes_array is not None:
        celltype_to_idx = preprocessing_info['celltype_to_idx']
        celltype_indices = []
        for ct in celltypes_array:
            if ct == 'PADDING':
                celltype_indices.append(0)
            else:
                celltype_indices.append(celltype_to_idx.get(ct, 0))
        celltype_indices_tensor = torch.tensor(celltype_indices, device=device)
    
    # Smooth loss hyperparameters
    lambda_smooth_max = 10
    smooth_warmup_frac = 0.25
    smooth_ramp_frac = 0.25
    smooth_total_frac = smooth_warmup_frac + smooth_ramp_frac

    with tqdm(total=num_epochs, desc="Training", disable=True) as pbar:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            _, phase_angles, reconstructed = model(expressions_tensor, celltype_indices_tensor)
            if valid_mask_tensor.sum() > 0:
                valid_expressions = expressions_tensor[valid_mask_tensor]
                valid_reconstructed = reconstructed[valid_mask_tensor]
                recon_loss = recon_criterion(valid_reconstructed, valid_expressions)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            time_loss = torch.tensor(0.0, device=device)
            if preprocessing_info['train_has_time'] and times_tensor is not None:
                valid_times = times_tensor[valid_mask_tensor]
                if len(valid_times) > 0:
                    valid_phase_angles = phase_angles[valid_mask_tensor]
                    time_loss = time_supervision_loss(valid_phase_angles, valid_times, 1.0, period_hours)

            smooth_loss = torch.tensor(0.0, device=device)
            # Enable smooth loss
            # If you want to re-enable, replace the following with the original computation:
            # valid_phase_angles = phase_angles[valid_mask_tensor]
            # smooth_jacobian_norms = model.decoder_jacobian_norm(valid_phase_angles, device=device)
            # smooth_loss = torch.mean(smooth_jacobian_norms)
            
            progress = epoch / num_epochs
            if progress < smooth_warmup_frac:
                lambda_smooth = 0.0
            elif progress < smooth_total_frac:
                lambda_smooth = lambda_smooth_max * (progress - smooth_warmup_frac) / smooth_ramp_frac
            else:
                lambda_smooth = lambda_smooth_max

            total_loss = lambda_recon * recon_loss + lambda_time * time_loss + lambda_smooth * smooth_loss
            total_loss.backward()
            optimizer.step()

            train_losses.append(total_loss.item())

            scheduler.step(total_loss.item())

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}: recon_loss={recon_loss.item():.4f}, "
                      f"time_loss={time_loss.item():.4f}, "
                      f"smooth_loss={smooth_loss.item():.4f}, "
                      f"lambda_smooth={lambda_smooth:.4f}, "
                      f"total_loss={total_loss.item():.4f}")

            pbar.update(1)
    
    leftover = os.path.join(save_dir, 'final_model.pth')
    if os.path.isfile(leftover):
        try:
            os.remove(leftover)
            pass
        except Exception as e:
            pass
    
    return train_losses


def main():
    # 训练曲线绘制后，画decoder拟合线
    parser = argparse.ArgumentParser(description="训练相位自编码器模型")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_time", type=float, default=0.5)
    parser.add_argument("--lambda_sine", type=float, default=0.5)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--save_dir", default='./phase_autoencoder_results')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_genes_plot", type=int, default=10)
    parser.add_argument("--custom_genes", nargs='*', default=None, required=True)
    parser.add_argument("--sine_predictor_hidden", type=int, default=64)
    parser.add_argument("--metadata", default=None)
    # cyc_mode removed: only boundary condition loss is used now

    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    # minimal logging

    df_tmp = pd.read_csv(args.train_file, low_memory=False)
    sample_columns_tmp = [col for col in df_tmp.columns if col != 'Gene_Symbol']
    max_samples = len(sample_columns_tmp)
    # device and genes are not printed
    
    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        args.train_file, args.n_components, max_samples, args.random_seed
    )
    
    preprocessing_info['period_hours'] = args.period_hours
    
    model = PhaseAutoEncoder(
        input_dim=args.n_components,
        n_celltypes=preprocessing_info.get('n_celltypes', 0),
        dropout=args.dropout
    )
    
    # start training
    train_losses = train_model(
        model=model,
        train_dataset=train_dataset,
        preprocessing_info=preprocessing_info,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        lambda_recon=args.lambda_recon,
        lambda_time=args.lambda_time,
        period_hours=args.period_hours,
    save_dir=args.save_dir
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_decoder_fits(model, args.n_components, os.path.join(args.save_dir, 'decoder_fit.png'))
    
    # training done
    
    if args.test_file:
    # testing
        
        test_dataset, test_preprocessing_info = load_and_preprocess_test_data(
            args.test_file, preprocessing_info
        )
        
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        results_df = predict_and_save_phases(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            device=args.device,
            save_dir=args.save_dir
        )

        if getattr(args, 'metadata', None):
            if os.path.isfile(args.metadata):
                from utils import generate_phase_metadata_comparison
                generate_phase_metadata_comparison(results_df, args.metadata, args.save_dir)
            else:
                pass

    # ...existing code...
        
    else:
        print("Training completed. No test file provided, so no predictions made.")

if __name__ == "__main__":
    main()
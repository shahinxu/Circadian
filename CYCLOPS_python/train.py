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
from utils import predict_and_save_phases, plot_comparsion
from dataset import load_and_preprocess_train_data, load_and_preprocess_test_data

def train_model(model, train_dataset, preprocessing_info, 
                num_epochs=100, lr=0.001, device='cuda',
                lambda_recon=1.0, lambda_time=0.5,
                period_hours=24.0, save_dir='./model_checkpoints'):
    
    if 'train_has_time' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_time'] = 'time' in sample
    
    if 'train_has_celltype' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_celltype'] = 'celltype' in sample
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
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
    with tqdm(total=num_epochs, desc="Training", disable=True) as pbar:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            # Forward
            phase_coords, _, reconstructed = model(expressions_tensor, celltype_indices_tensor)
            phase_coords_norm = torch.norm(phase_coords, dim=1, keepdim=True) + 1e-8
            phase_coords_normalized = phase_coords / phase_coords_norm
            circular_loss = nn.functional.mse_loss(phase_coords, phase_coords_normalized)
            recon_loss = nn.functional.mse_loss(reconstructed, expressions_tensor)
            total_loss = circular_loss + lambda_recon * recon_loss
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())
            scheduler.step(total_loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: circular_loss={circular_loss.item():.4f}, recon_loss={recon_loss.item():.4f}")
            pbar.update(1)
    return train_losses


def main():


    parser = argparse.ArgumentParser(description="训练相位自编码器模型")
    parser.add_argument("--dataset_path", required=True, help="如 GSE108539/Exon")
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_time", type=float, default=0.5)
    parser.add_argument("--lambda_sine", type=float, default=0.5)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_genes_plot", type=int, default=10)
    parser.add_argument("--sine_predictor_hidden", type=int, default=64)
    args = parser.parse_args()

    base_data = f"../data/{args.dataset_path}"
    train_file = os.path.join(base_data, "expression.csv")
    test_file = os.path.join(base_data, "expression.csv")
    metadata = os.path.join(base_data, "metadata.csv")
    save_dir = os.path.join("results", args.dataset_path)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    os.makedirs(save_dir, exist_ok=True)

    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        train_file, args.n_components
    )
    
    preprocessing_info['period_hours'] = args.period_hours
    
    model = PhaseAutoEncoder(
        input_dim=args.n_components,
        dropout=args.dropout
    )
    
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
        save_dir=save_dir
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
            
    if os.path.isfile(test_file):
        test_dataset, test_preprocessing_info = load_and_preprocess_test_data(
            test_file, preprocessing_info
        )
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        results_df = predict_and_save_phases(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            device=args.device,
            save_dir=save_dir
        )            
        plot_comparsion(results_df, metadata, save_dir)
    else:
        print("Training completed. No test file provided, so no predictions made.")

if __name__ == "__main__":
    main()
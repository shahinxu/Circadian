import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

from models import TemporalGraphPINN
from losses import total_unsupervised_loss
from utils import *
from data import load_expression_data

def train_unsupervised_model(model, eigengene_data, config, device):
    optimizer = optim.Adam(model.parameters(),
                          lr=config['training']['learning_rate'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config['training']['patience'] // 2,
        factor=0.5
    )

    best_loss = float('inf')
    train_losses = []
    loss_components_history = []

    exp_dir = create_experiment_dir(config['experiment']['save_dir'],
                                   config['experiment']['name'])

    pbar = tqdm(range(config['training']['n_epochs']), 
                desc="Training", 
                unit="epoch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for epoch in pbar:
        model.train()

        optimizer.zero_grad()

        total_loss, loss_components = total_unsupervised_loss(
            model, eigengene_data, device,
            lambda_recon=config['loss_weights']['reconstruction'],
            lambda_physics=config['loss_weights']['physics'],
            lambda_tree=config['loss_weights']['tree'],
            lambda_sign=config['loss_weights']['sign_consistency'],
            lambda_sparsity=config['loss_weights']['sparsity']
        )

        total_loss.backward()
        optimizer.step()

        scheduler.step(total_loss.detach())

        train_losses.append(total_loss.item())
        loss_components_history.append(loss_components)

        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'recon': f'{loss_components["reconstruction"]:.4f}',
            'physics': f'{loss_components["physics"]:.4f}',
            'sparsity': f'{loss_components["sparsity"]:.4f}'
        })

        if config['experiment']['save_checkpoints'] and total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(model, optimizer, epoch, total_loss, exp_dir / 'best_model.pth')

        if early_stopping(train_losses, config['training']['patience'], float(config['training']['min_delta'])):
            pbar.set_description(f"Early stopping at epoch {epoch}")
            break

    pbar.close()

    T, W, W_sparse = model.get_graph_matrices()
    inferred_times = model.infer_node_times(W_sparse)

    torch.save(model.state_dict(), exp_dir / 'final_model.pth')
    np.save(exp_dir / 'inferred_times.npy', inferred_times.detach().cpu().numpy())

    return train_losses, loss_components_history, exp_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    if args.data_path:
        eigengene_data, _ = load_expression_data(args.data_path)
        eigengene_data = torch.tensor(eigengene_data, dtype=torch.float32)
    else:
        from generate_data import generate_circadian_eigengenes
        _, eigengene_data = generate_circadian_eigengenes(
            n_samples=config['data']['n_samples'],
            n_eigengenes=config['model']['n_eigengenes']
        )
        eigengene_data = torch.tensor(eigengene_data, dtype=torch.float32)

    n_samples, n_eigengenes = eigengene_data.shape

    model = TemporalGraphPINN(
        n_nodes=n_samples,
        node_dim=config['model']['node_dim'],
        n_eigengenes=n_eigengenes
    ).to(device)

    set_random_seed(42)

    train_losses, loss_components, exp_dir = train_unsupervised_model(
        model, eigengene_data, config, device
    )

    T, W, W_sparse = model.get_graph_matrices()
    inferred_times = model.infer_node_times(W_sparse)

    model.eval()
    with torch.no_grad():
        final_predictions = model(inferred_times)
        np.save(exp_dir / 'final_predictions.npy', final_predictions.cpu().numpy())

if __name__ == "__main__":
    main()

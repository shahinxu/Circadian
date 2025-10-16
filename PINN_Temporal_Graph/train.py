"""
Training script for PINN Temporal Graph Learning (Unsupervised)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml

from models import TemporalGraphPINN
from losses import total_unsupervised_loss
from utils import *
from data import load_expression_data, prepare_training_data

def train_unsupervised_model(model, eigengene_data, config, device):
    """
    Train the PINN Temporal Graph model in unsupervised manner

    Args:
        model: TemporalGraphPINN model
        eigengene_data: eigengene expressions (n_samples, n_eigengenes)
        config: configuration dictionary
        device: torch device

    Returns: training losses and experiment directory
    """
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

    for epoch in range(config['training']['n_epochs']):
        model.train()

        optimizer.zero_grad()

        # Compute total unsupervised loss
        total_loss, loss_components = total_unsupervised_loss(
            model, eigengene_data, device,
            lambda_recon=config['loss_weights']['reconstruction'],
            lambda_physics=config['loss_weights']['physics'],
            lambda_tree=config['loss_weights']['tree'],
            lambda_sign=config['loss_weights']['sign_consistency']
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        scheduler.step(total_loss)

        train_losses.append(total_loss.item())
        loss_components_history.append(loss_components)

        if epoch % config['experiment']['log_interval'] == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}")
            print(f"  Reconstruction: {loss_components['reconstruction']:.4f}")
            print(f"  Physics: {loss_components['physics']:.4f}")
            print(f"  Tree: {loss_components['tree']:.4f}")
            print(f"  Sign Consistency: {loss_components['sign_consistency']:.4f}")

        # Save checkpoints
        if config['experiment']['save_checkpoints'] and total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(model, optimizer, epoch, total_loss, exp_dir / 'best_model.pth')

        # Early stopping
        if early_stopping(train_losses, config['training']['patience'], config['training']['min_delta']):
            print(f"Early stopping at epoch {epoch}")
            break

    # Save final results
    torch.save(model.state_dict(), exp_dir / 'final_model.pth')

    # Analyze and save graph properties
    T, W, W_sparse = model.get_graph_matrices()
    graph_properties = analyze_graph_properties(T, W_sparse)

    # Get inferred times
    inferred_times = model.infer_node_times(W_sparse)

    log_experiment_results(exp_dir, model, train_losses, loss_components_history, graph_properties)

    # Save inferred times
    np.save(exp_dir / 'inferred_times.npy', inferred_times.detach().cpu().numpy())

    # Plot results
    if config['experiment']['plot_results']:
        plot_training_history(train_losses,
                            [comp['reconstruction'] for comp in loss_components_history],
                            exp_dir / 'training_history.png')

        plot_graph_structure(W_sparse, save_path=exp_dir / 'learned_graph.png')

        # Plot inferred temporal ordering
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(inferred_times)), inferred_times.detach().cpu().numpy(),
                   c=range(len(inferred_times)), cmap='viridis')
        plt.xlabel('Sample Index')
        plt.ylabel('Inferred Time')
        plt.title('Inferred Temporal Ordering')
        plt.colorbar(label='Sample Index')
        plt.savefig(exp_dir / 'inferred_times.png')
        plt.close()

    return train_losses, loss_components_history, exp_dir

def main():
    parser = argparse.ArgumentParser(description='Train PINN Temporal Graph model (Unsupervised)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to expression data CSV file')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Configuration: {args.config}")

    # Load or generate data
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        eigengene_data, _ = load_expression_data(args.data_path)
        eigengene_data = torch.tensor(eigengene_data, dtype=torch.float32)
    else:
        print("No data path provided, generating synthetic data")
        from generate_data import generate_circadian_eigengenes
        _, eigengene_data = generate_circadian_eigengenes(
            n_samples=config['data']['n_samples'],
            n_eigengenes=config['model']['n_eigengenes']
        )
        eigengene_data = torch.tensor(eigengene_data, dtype=torch.float32)

    n_samples, n_eigengenes = eigengene_data.shape
    print(f"Data shape: {n_samples} samples × {n_eigengenes} eigengenes")

    # Create model
    model = TemporalGraphPINN(
        n_nodes=n_samples,  # Each sample is a node
        node_dim=config['model']['node_dim'],
        n_eigengenes=n_eigengenes
    ).to(device)

    print(f"Model created with {count_parameters(model):,} parameters")
    print_model_summary(model)

    # Set random seed
    set_random_seed(42)

    # Train model
    print("Starting unsupervised training...")
    train_losses, loss_components, exp_dir = train_unsupervised_model(
        model, eigengene_data, config, device
    )

    # Final analysis
    print(f"Training completed! Results saved to {exp_dir}")

    # Get final inferred times and graph
    T, W, W_sparse = model.get_graph_matrices()
    inferred_times = model.infer_node_times(W_sparse)

    print("\nFinal Results:")
    print(f"  Learned {T.sum().item():.1f} edges in topology matrix")
    print(f"  Inferred time range: {inferred_times.min().item():.3f} to {inferred_times.max().item():.3f}")

    # Save final eigengene predictions
    model.eval()
    with torch.no_grad():
        final_predictions = model(inferred_times)
        np.save(exp_dir / 'final_predictions.npy', final_predictions.cpu().numpy())

if __name__ == "__main__":
    main()

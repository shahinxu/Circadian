"""
Utility functions for PINN Temporal Graph Learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_graph_structure(W_sparse, node_labels=None, save_path=None):
    """Visualize learned graph structure"""
    W_np = W_sparse.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(W_np, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)

    if node_labels:
        plt.xticks(range(len(node_labels)), node_labels, rotation=45)
        plt.yticks(range(len(node_labels)), node_labels)

    plt.xlabel('Target Node')
    plt.ylabel('Source Node')
    plt.title('Learned Graph Structure (Edge Weights)')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_temporal_predictions(model, t_values, true_expressions, device, save_path=None):
    """Plot model predictions vs true values over time"""
    model.eval()
    with torch.no_grad():
        t_tensor = t_values.unsqueeze(-1).to(device)
        predictions = model(t_tensor).cpu()

    n_eigengenes = min(6, predictions.shape[1])  # Plot up to 6 eigengenes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_eigengenes):
        ax = axes[i]
        ax.plot(t_values.numpy(), true_expressions[:, i].numpy(),
               'b-', label='True', linewidth=2)
        ax.plot(t_values.numpy(), predictions[:, i].numpy(),
               'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(f'Eigengene {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_graph_properties(T, W_sparse):
    """Analyze properties of learned graph"""
    properties = {}

    # Sparsity
    n_edges = (T > 0.5).sum().item()
    n_possible_edges = T.numel() - T.shape[0]  # Exclude self-loops
    properties['sparsity'] = n_edges / n_possible_edges

    # Edge weight statistics
    edge_weights = W_sparse[W_sparse.abs() > 1e-6]
    if len(edge_weights) > 0:
        properties['mean_weight'] = edge_weights.mean().item()
        properties['std_weight'] = edge_weights.std().item()
        properties['max_weight'] = edge_weights.abs().max().item()

    # Graph connectivity (simplified)
    degree = (T > 0.5).sum(dim=1)
    properties['mean_degree'] = degree.float().mean().item()
    properties['max_degree'] = degree.max().item()

    return properties

def save_config(config, filepath):
    """Save configuration to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(filepath):
    """Load configuration from YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def create_experiment_dir(base_dir='experiments', exp_name=None):
    """Create directory for experiment results"""
    if exp_name is None:
        import datetime
        exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir

def log_experiment_results(exp_dir, model, train_losses, val_losses, graph_properties):
    """Log experiment results to files"""
    exp_dir = Path(exp_dir)

    # Save losses
    np.save(exp_dir / 'train_losses.npy', np.array(train_losses))
    np.save(exp_dir / 'val_losses.npy', np.array(val_losses))

    # Save graph properties
    with open(exp_dir / 'graph_properties.json', 'w') as f:
        json.dump(graph_properties, f, indent=2)

    # Save model info
    model_info = {
        'n_parameters': count_parameters(model),
        'n_nodes': model.n_nodes,
        'n_eigengenes': model.n_eigengenes
    }

    with open(exp_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    r2 = 1 - mse / torch.var(targets)

    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'r2': r2.item()
    }

def early_stopping(val_losses, patience=50, min_delta=1e-4):
    """Check if training should stop early"""
    if len(val_losses) < patience:
        return False

    recent_losses = val_losses[-patience:]
    min_recent = min(recent_losses)
    min_overall = min(val_losses)

    return min_recent > min_overall - min_delta

def print_model_summary(model):
    """Print model architecture summary"""
    print("Model Summary:")
    print("=" * 50)
    print(f"Total parameters: {count_parameters(model):,}")

    print("\nModel components:")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n_params:,} parameters")

    print("\nParameter details:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape} = {param.numel():,} parameters")
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def plot_training_history(train_losses, val_losses, save_path=None):
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
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed, falling back to matrix visualization")
        plot_graph_matrix(W_sparse, node_labels, save_path)
        return

    W_np = W_sparse.detach().cpu().numpy()
    n_nodes = W_np.shape[0]

    G = nx.DiGraph()

    for i in range(n_nodes):
        G.add_node(i, label=f'Node {i+1}' if node_labels is None else node_labels[i])

    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and abs(W_np[i, j]) > 1e-6:
                edges.append((i, j, W_np[i, j]))

    if not edges:
        print("No edges found in the graph")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No edges in graph', ha='center', va='center', fontsize=16)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    plt.figure(figsize=(12, 8))

    edge_weights = [abs(w) for _, _, w in edges]
    max_weight = max(edge_weights) if edge_weights else 1.0
    min_weight = min(edge_weights) if edge_weights else 0.0

    positive_edges = [(i, j, w) for i, j, w in edges if w > 0]
    negative_edges = [(i, j, w) for i, j, w in edges if w < 0]

    if positive_edges:
        pos_weights = [abs(w) for _, _, w in positive_edges]
        pos_widths = [2 + 8 * (w - min_weight) / (max_weight - min_weight + 1e-6) for w in pos_weights]
        pos_colors = ['red' if w > 0 else 'blue' for w in [w for _, _, w in positive_edges]]

        nx.draw_networkx_edges(G, pos,
                              edgelist=[(i, j) for i, j, _ in positive_edges],
                              width=pos_widths,
                              edge_color='red',
                              alpha=0.7,
                              arrows=True,
                              arrowsize=20,
                              connectionstyle='arc3,rad=0.1')

    if negative_edges:
        neg_weights = [abs(w) for _, _, w in negative_edges]
        neg_widths = [2 + 8 * (w - min_weight) / (max_weight - min_weight + 1e-6) for w in neg_weights]

        nx.draw_networkx_edges(G, pos,
                              edgelist=[(i, j) for i, j, _ in negative_edges],
                              width=neg_widths,
                              edge_color='blue',
                              alpha=0.7,
                              arrows=True,
                              arrowsize=20,
                              connectionstyle='arc3,rad=-0.1')

    node_sizes = [800 + 200 * (G.degree(i) / max(1, G.number_of_edges())) for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)

    labels = {i: f'Node {i+1}' if node_labels is None else node_labels[i] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    plt.title(f'Learned Graph Structure\n{len(edges)} edges, {n_nodes} nodes', fontsize=14, pad=20)
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_graph_matrix(W_sparse, node_labels=None, save_path=None):
    """Fallback matrix visualization"""
    W_np = W_sparse.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(W_np, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)

    if node_labels:
        plt.xticks(range(len(node_labels)), node_labels, rotation=45, ha='right')
        plt.yticks(range(len(node_labels)), node_labels)

    plt.xlabel('Target Node', fontsize=12)
    plt.ylabel('Source Node', fontsize=12)
    plt.title('Learned Graph Structure (Adjacency Matrix)', fontsize=14, pad=20)

    n_edges = np.sum(np.abs(W_np) > 1e-6)
    plt.text(0.02, 0.98, f'Edges: {n_edges}', transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_predictions(model, t_values, true_expressions, device, save_path=None):
    model.eval()
    with torch.no_grad():
        t_tensor = t_values.unsqueeze(-1).to(device)
        predictions = model(t_tensor).cpu()

    n_eigengenes = min(6, predictions.shape[1])
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

def plot_inferred_expressions(model, inferred_times, eigengene_data, device, save_path=None):
    """
    Plot eigengene expressions at inferred time points
    """
    model.eval()

    # Ensure all tensors are on the same device
    inferred_times = inferred_times.to(device)
    eigengene_data = eigengene_data.to(device)

    # Sort by inferred times for better visualization
    sorted_indices = torch.argsort(inferred_times)
    sorted_times = inferred_times[sorted_indices]
    sorted_expressions = eigengene_data[sorted_indices]

    # Generate predictions at inferred times
    with torch.no_grad():
        predicted_expressions = model(sorted_times.unsqueeze(-1).to(device)).cpu().detach()

    # Move back to CPU for plotting
    sorted_times = sorted_times.cpu().detach()
    sorted_expressions = sorted_expressions.cpu().detach()

    n_eigengenes = min(8, eigengene_data.shape[1])
    n_cols = 4
    n_rows = (n_eigengenes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(n_eigengenes):
        ax = axes[i]

        # Plot true expressions at inferred times
        ax.scatter(sorted_times.numpy(), sorted_expressions[:, i].numpy(),
                  c='blue', alpha=0.7, s=50, label='True expression')

        # Plot model predictions at inferred times
        ax.scatter(sorted_times.numpy(), predicted_expressions[:, i].numpy(),
                  c='red', alpha=0.7, s=30, marker='x', label='Model prediction')

        # Add trend lines
        ax.plot(sorted_times.numpy(), sorted_expressions[:, i].numpy(),
               'b-', alpha=0.3, linewidth=1)
        ax.plot(sorted_times.numpy(), predicted_expressions[:, i].numpy(),
               'r--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Inferred Time (hours)')
        ax.set_ylabel(f'Eigengene {i+1} Expression')
        ax.set_title(f'Eigengene {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_eigengenes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Eigengene Expressions at Inferred Time Points', fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_graph_properties(T, W_sparse):
    properties = {}

    n_edges = (T > 0.5).sum().item()
    n_possible_edges = T.numel() - T.shape[0]
    properties['sparsity'] = n_edges / n_possible_edges

    edge_weights = W_sparse[W_sparse.abs() > 1e-6]
    if len(edge_weights) > 0:
        properties['mean_weight'] = edge_weights.mean().item()
        properties['std_weight'] = edge_weights.std().item()
        properties['max_weight'] = edge_weights.abs().max().item()

    degree = (T > 0.5).sum(dim=1)
    properties['mean_degree'] = degree.float().mean().item()
    properties['max_degree'] = degree.max().item()

    return properties

def save_config(config, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def create_experiment_dir(base_dir='experiments', exp_name=None):
    if exp_name is None:
        import datetime
        exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir

def log_experiment_results(exp_dir, model, train_losses, val_losses, graph_properties):
    exp_dir = Path(exp_dir)

    np.save(exp_dir / 'train_losses.npy', np.array(train_losses))
    np.save(exp_dir / 'val_losses.npy', np.array(val_losses))

    with open(exp_dir / 'graph_properties.json', 'w') as f:
        json.dump(graph_properties, f, indent=2)

    model_info = {
        'n_parameters': count_parameters(model),
        'n_nodes': model.n_nodes,
        'n_eigengenes': model.n_eigengenes
    }

    with open(exp_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

def compute_metrics(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    r2 = 1 - mse / torch.var(targets)

    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'r2': r2.item()
    }

def early_stopping(val_losses, patience=50, min_delta=1e-4):
    if len(val_losses) < patience:
        return False

    recent_losses = val_losses[-patience:]
    min_recent = min(recent_losses)
    min_overall = min(val_losses)

    return min_recent > min_overall - min_delta

def print_model_summary(model):
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
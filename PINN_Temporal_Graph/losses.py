"""
Loss functions for PINN Temporal Graph Learning

Includes:
- Physics constraints (PINN losses)
- Graph structure constraints (tree, directed tree, sign consistency)
- Data reconstruction losses
"""

import torch
import torch.nn.functional as F
from models import TemporalGraphPINN

def reconstruction_loss(model: 'TemporalGraphPINN', eigengene_data, inferred_times, device):
    """
    Data reconstruction loss: omniscient network prediction vs true eigengene expressions

    Args:
        model: TemporalGraphPINN model
        eigengene_data: true eigengene expressions (n_samples, n_eigengenes)
        inferred_times: inferred time values for each sample (n_samples,)
        device: torch device

    Returns: MSE loss between predicted and true expressions
    """
    eigengene_data = eigengene_data.to(device)
    inferred_times = inferred_times.to(device)

    # Predict expressions using omniscient network
    predicted_expressions = model.omniscient_net(inferred_times.unsqueeze(-1))

    # MSE loss
    loss = F.mse_loss(predicted_expressions, eigengene_data)
    return loss

def physics_constraint_loss(
        model: TemporalGraphPINN, 
        eigengene_data, 
        inferred_times, 
        device
    ):
    """
    Physics constraint loss: dE/dt from omniscient network should match dE/dt from graph

    Args:
        model: TemporalGraphPINN model
        eigengene_data: true eigengene expressions (n_samples, n_eigengenes)
        inferred_times: inferred time values for each sample (n_samples,)
        device: torch device

    Returns: MSE loss between network derivatives and graph derivatives
    """
    eigengene_data = eigengene_data.to(device)
    inferred_times = inferred_times.to(device).detach().requires_grad_(True)

    # Get derivatives from omniscient network
    expressions_pred = model.omniscient_net(inferred_times.unsqueeze(-1))  # (n_samples, n_eigengenes)

    derivatives_network = []
    for i in range(expressions_pred.shape[1]):  # For each eigengene
        eigengene_i = expressions_pred[:, i]  # (n_samples,)
        # Ensure eigengene_i requires grad
        if not eigengene_i.requires_grad:
            eigengene_i = eigengene_i.detach().requires_grad_(True)
        dEdt_i = torch.autograd.grad(
            eigengene_i.sum(), inferred_times,
            create_graph=True, retain_graph=True
        )[0]  # (n_samples,)
        derivatives_network.append(dEdt_i)

    dEdt_network = torch.stack(derivatives_network, dim=1)  # (n_samples, n_eigengenes)

    # Get graph matrices and compute graph-based derivatives
    T, W, W_sparse = model.get_graph_matrices()

    dEdt_graph = torch.zeros_like(dEdt_network)  # (n_samples, n_eigengenes)

    for sample_idx in range(len(inferred_times)):
        # For each sample, compute graph-based derivative
        graph_derivatives = []

        for eigengene_idx in range(eigengene_data.shape[1]):
            neighbor_derivatives = []

            # Look at all neighbors (incoming and outgoing edges)
            for neighbor_idx in range(W_sparse.shape[0]):
                edge_weight = W_sparse[sample_idx, neighbor_idx]
                if edge_weight.abs() > 1e-6:  # Edge exists
                    eigengene_neighbor = eigengene_data[neighbor_idx, eigengene_idx]
                    eigengene_current = eigengene_data[sample_idx, eigengene_idx]
                    time_diff = inferred_times[neighbor_idx] - inferred_times[sample_idx]

                    if time_diff.abs() > 1e-6:  # Avoid division by zero
                        derivative = (eigengene_neighbor - eigengene_current) / time_diff
                        neighbor_derivatives.append(derivative)

            if neighbor_derivatives:
                # Use average of neighbor derivatives
                graph_deriv = torch.mean(torch.stack(neighbor_derivatives))
            else:
                # No neighbors: use the network derivative as default
                # This ensures physics loss is meaningful even for isolated nodes
                graph_deriv = dEdt_network[sample_idx, eigengene_idx].detach()

            graph_derivatives.append(graph_deriv)

        dEdt_graph[sample_idx] = torch.stack(graph_derivatives)

    # Physics loss: ||dE_network/dt - dE_graph/dt||^2
    physics_loss = F.mse_loss(dEdt_network, dEdt_graph)
    return physics_loss

def tree_loss(A, alpha=1.0, beta=0.1):
    """
    Tree structure and connectivity constraint loss

    Args:
        A: predicted adjacency matrix, shape [n, n], symmetric
        alpha: weight for edge count constraint
        beta: weight for connectivity constraint

    Returns: constraint loss
    """
    n = A.size(0)

    # Make sure A is symmetric (for undirected graph)
    A_sym = (A + A.t()) / 2

    # Laplacian matrix
    D = torch.diag(A_sym.sum(dim=1))
    L = D - A_sym

    # 1. Edge count constraint: tree should have n-1 edges
    n_edges = A_sym.sum() / 2  # Divide by 2 because symmetric
    E_loss = (n_edges - (n - 1))**2

    # 2. Connectivity constraint: Laplacian second smallest eigenvalue λ₂
    eigvals = torch.linalg.eigvalsh(L)
    lambda2 = eigvals[1]  # Second smallest eigenvalue
    Connect_loss = torch.relu(-lambda2)**2  # Penalize if λ₂ < 0 (disconnected)

    return alpha * E_loss + beta * Connect_loss

def sign_consistency_loss(W_sparse):
    """
    Simplified sign consistency loss - penalize sign flips between adjacent edges
    """
    # For now, just penalize large negative weights (encourage positive temporal flow)
    negative_penalty = torch.clamp(-W_sparse, min=0).sum()
    return negative_penalty

def sparsity_loss(T, target_sparsity=0.05):
    """
    Sparsity loss: encourage sparse graph structure
    
    Args:
        T: topology matrix (n_nodes, n_nodes), values in [0,1]
        target_sparsity: desired fraction of edges (default 5%)
    
    Returns: loss that penalizes deviation from target sparsity
    """
    n_nodes = T.shape[0]
    n_possible_edges = n_nodes * (n_nodes - 1)  # Exclude self-loops
    
    # Current number of edges
    n_edges = T.sum()
    current_sparsity = n_edges / n_possible_edges
    
    # Loss: squared deviation from target sparsity
    sparsity_loss = (current_sparsity - target_sparsity)**2
    
    return sparsity_loss

def total_unsupervised_loss(
        model: TemporalGraphPINN, 
        eigengene_data, 
        device,
        lambda_recon=1.0, 
        lambda_physics=1.0,
        lambda_tree=1.0, 
        lambda_sign=1.0,
        lambda_sparsity=0.1
    ):
    """
    Combined total loss for unsupervised learning

    Args:
        model: TemporalGraphPINN model
        eigengene_data: eigengene expressions (n_samples, n_eigengenes)
        device: torch device
        lambda_*: loss weights

    Returns: total loss and loss components
    """
    # Get graph matrices
    T, W, W_sparse = model.get_graph_matrices()

    # Infer times from graph structure
    inferred_times = model.infer_node_times(W_sparse)

    # Reconstruction loss: omniscient network prediction vs true data
    recon_loss = reconstruction_loss(model, eigengene_data, inferred_times, device)

    # Physics constraint loss: network derivatives vs graph derivatives
    physics_loss = physics_constraint_loss(model, eigengene_data, inferred_times, device)

    # Tree structure constraint
    tree_loss_val = tree_loss(T)

    # Sign consistency constraint
    sign_loss = sign_consistency_loss(W_sparse)

    # Sparsity constraint
    sparse_loss = sparsity_loss(T, target_sparsity=0.01)

    # Total loss
    total = (lambda_recon * recon_loss +
             lambda_physics * physics_loss +
             lambda_tree * tree_loss_val +
             lambda_sign * sign_loss +
             lambda_sparsity * sparse_loss)

    return total, {
        'reconstruction': recon_loss.item(),
        'physics': physics_loss.item(),
        'tree': tree_loss_val.item(),
        'sign_consistency': sign_loss.item(),
        'sparsity': sparse_loss.item()
    }
import torch
import torch.nn.functional as F
from models import TemporalGraphPINN

def reconstruction_loss(model, eigengene_data, inferred_times, device):
    eigengene_data = eigengene_data.to(device)
    inferred_times = inferred_times.to(device)
    predicted_expressions = model.omniscient_net(inferred_times.unsqueeze(-1))
    loss = F.mse_loss(predicted_expressions, eigengene_data)
    return loss

def physics_constraint_loss(model, eigengene_data, inferred_times, device):
    eigengene_data = eigengene_data.to(device)
    inferred_times = inferred_times.to(device).detach().requires_grad_(True)

    expressions_pred = model.omniscient_net(inferred_times.unsqueeze(-1))

    derivatives_network = []
    for i in range(expressions_pred.shape[1]):
        eigengene_i = expressions_pred[:, i]
        if not eigengene_i.requires_grad:
            eigengene_i = eigengene_i.detach().requires_grad_(True)
        dEdt_i = torch.autograd.grad(
            eigengene_i.sum(), inferred_times,
            create_graph=True, retain_graph=True
        )[0]
        derivatives_network.append(dEdt_i)

    dEdt_network = torch.stack(derivatives_network, dim=1)

    T, W, W_sparse = model.get_graph_matrices()

    dEdt_graph = torch.zeros_like(dEdt_network)

    for sample_idx in range(len(inferred_times)):
        graph_derivatives = []

        for eigengene_idx in range(eigengene_data.shape[1]):
            neighbor_derivatives = []

            for neighbor_idx in range(W_sparse.shape[0]):
                edge_weight = W_sparse[sample_idx, neighbor_idx]
                if edge_weight.abs() > 1e-6:
                    eigengene_neighbor = eigengene_data[neighbor_idx, eigengene_idx]
                    eigengene_current = eigengene_data[sample_idx, eigengene_idx]
                    time_diff = inferred_times[neighbor_idx] - inferred_times[sample_idx]

                    if time_diff.abs() > 1e-6:
                        derivative = (eigengene_neighbor - eigengene_current) / time_diff
                        neighbor_derivatives.append(derivative)

            if neighbor_derivatives:
                graph_deriv = torch.mean(torch.stack(neighbor_derivatives))
            else:
                graph_deriv = dEdt_network[sample_idx, eigengene_idx].detach()

            graph_derivatives.append(graph_deriv)

        dEdt_graph[sample_idx] = torch.stack(graph_derivatives)

    physics_loss = F.mse_loss(dEdt_network, dEdt_graph)
    return physics_loss

def tree_loss(A, alpha=1.0, beta=0.1):
    n = A.size(0)
    A_sym = (A + A.t()) / 2
    D = torch.diag(A_sym.sum(dim=1))
    L = D - A_sym

    n_edges = A_sym.sum() / 2
    E_loss = (n_edges - (n - 1))**2

    eigvals = torch.linalg.eigvalsh(L)
    lambda2 = eigvals[1]
    Connect_loss = torch.relu(-lambda2)**2

    return alpha * E_loss + beta * Connect_loss

def sign_consistency_loss(W_sparse):
    negative_penalty = torch.clamp(-W_sparse, min=0).sum()
    return negative_penalty

def sparsity_loss(T, target_sparsity=0.05):
    n_nodes = T.shape[0]
    n_possible_edges = n_nodes * (n_nodes - 1)
    
    n_edges = T.sum()
    current_sparsity = n_edges / n_possible_edges
    
    sparsity_loss = (current_sparsity - target_sparsity)**2
    
    return sparsity_loss

def total_unsupervised_loss(
        model, 
        eigengene_data, 
        device,
        lambda_recon=1.0, 
        lambda_physics=1.0,
        lambda_tree=1.0, 
        lambda_sign=1.0,
        lambda_sparsity=0.01
    ):
    T, W, W_sparse = model.get_graph_matrices()
    inferred_times = model.infer_node_times(W_sparse)

    recon_loss = reconstruction_loss(model, eigengene_data, inferred_times, device)
    physics_loss = physics_constraint_loss(model, eigengene_data, inferred_times, device)
    tree_loss_val = tree_loss(T)
    sign_loss = sign_consistency_loss(W_sparse)
    sparse_loss = sparsity_loss(T)

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
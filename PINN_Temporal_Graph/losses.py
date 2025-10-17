import torch
import torch.nn.functional as F
from models import TemporalGraphPINN

def reconstruction_loss(
        model: TemporalGraphPINN, 
        eigengene_data, 
        inferred_times, 
        device
    ):
    eigengene_data = eigengene_data.to(device)
    inferred_times = inferred_times.to(device)
    predicted_expressions = model.omniscient_net(inferred_times.unsqueeze(-1))
    loss = F.mse_loss(predicted_expressions, eigengene_data)
    return loss

def physics_loss(
        model: TemporalGraphPINN,
        eigengene_data, 
        inferred_times, 
        probs,
        device
    ):
    eigengene_data = eigengene_data.to(device)
    inferred_times = inferred_times.to(device).detach().requires_grad_(True)

    expressions_pred = model.omniscient_net(inferred_times.unsqueeze(-1))
    expressions_pred.requires_grad_(True)

    dEdt_list = []
    for i in range(expressions_pred.shape[1]):
        grad_i = torch.autograd.grad(
            outputs=expressions_pred[:, i],
            inputs=inferred_times,
            grad_outputs=torch.ones_like(inferred_times),
            create_graph=True,
            retain_graph=True
        )[0]
        dEdt_list.append(grad_i)
    dEdt_network = torch.stack(dEdt_list, dim=1)

    N, D = eigengene_data.shape
    E_i = eigengene_data.unsqueeze(1).expand(-1, N, -1)
    E_j = eigengene_data.unsqueeze(0).expand(N, -1, -1)
    t_i = inferred_times.unsqueeze(1).expand(-1, N)
    t_j = inferred_times.unsqueeze(0).expand(N, -1)

    time_diff = t_j - t_i
    feature_diff = E_j - E_i

    T, _, _ = model.get_graph_matrices()
    edge_mask = T > 1e-6
    probs_safe = probs * edge_mask.float()
    probs_safe = probs_safe.unsqueeze(-1)
    safe_time_diff = torch.where(
        (time_diff.abs() < 1e-6) | (~edge_mask), 
        torch.ones_like(time_diff), 
        time_diff
    ).unsqueeze(-1)
    
    weighted_derivatives = (feature_diff / safe_time_diff) * probs_safe
    neighbor_weight_sum = probs_safe.sum(dim=1).clamp_min(1e-6)
    dEdt_graph = weighted_derivatives.sum(dim=1) / neighbor_weight_sum

    return F.mse_loss(dEdt_network, dEdt_graph)

def tree_loss(A, alpha=1.0, beta=0.1, eps=1e-6):
    n = A.size(0)
    device = A.device

    n_edges = A.sum()
    E_loss = (n_edges - (n - 1))**2

    P = A / (A.sum(dim=0, keepdim=True) + eps)
    I = torch.eye(n, device=device)
    P_hat = P + I

    reach = torch.matrix_power(P_hat, n)
    Connect_loss = torch.relu(1.0 - reach[0,1:]).sum()

    return alpha * E_loss + beta * Connect_loss

def negative_loss(W_sparse):
    negative_penalty = torch.clamp(-W_sparse, min=0).sum()
    return negative_penalty

def compute_loss(
    model: TemporalGraphPINN, 
    eigengene_data, 
    device,
    lambda_recon=1.0, 
    lambda_physics=1.0,
    lambda_tree=1.0, 
    lambda_sign=1.0
):
    T, _, W_sparse = model.get_graph_matrices()
    inferred_times, probs = model.infer_node_times(W_sparse)

    loss_recon = reconstruction_loss(model, eigengene_data, inferred_times, device)
    loss_physics = physics_loss(model, eigengene_data, inferred_times, probs, device)
    loss_tree = tree_loss(T)
    loss_sign = negative_loss(W_sparse)

    total_loss = (
        lambda_recon * loss_recon +
        lambda_physics * loss_physics +
        lambda_tree * loss_tree +
        lambda_sign * loss_sign
    )

    return total_loss, {
        'reconstruction': loss_recon.item(),
        'physics': loss_physics.item(),
        'tree': loss_tree.item(),
        'negative': loss_sign.item()
    }
import torch
import torch.nn as nn
import math

class OmniscientNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_eigengenes=5):
        super().__init__()
        self.n_eigengenes = n_eigengenes
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_eigengenes)
        )

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(t)

class TopologyNetwork(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        T_init = torch.zeros(n_nodes, n_nodes)

        node_order = torch.randperm(n_nodes)
        for i in range(n_nodes - 1):
            src = node_order[i]
            dst = node_order[i + 1]
            T_init[src, dst] = 1.0

        self.T = nn.Parameter(T_init + torch.randn(n_nodes, n_nodes) * 0.1)

    def forward(self, threshold=0.5):
        T_logits = self.T
        probs = torch.sigmoid(T_logits)
        T_hard = (probs > threshold).float()
        T = T_hard + (probs - probs.detach())
        T = T * (1 - torch.eye(T.shape[0], device=T.device))
        return T

class WeightNetwork(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)

    def forward(self):
        W = self.W
        W = W * (1 - torch.eye(W.shape[0], device=W.device))
        return W

class TemporalGraphPINN(nn.Module):
    def __init__(self, n_nodes, n_eigengenes=5):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_eigengenes = n_eigengenes

        self.omniscient_net = OmniscientNetwork(1, n_eigengenes=n_eigengenes)
        self.topology_net = TopologyNetwork(n_nodes)
        self.weight_net = WeightNetwork(n_nodes)

    def forward(self, t_values):
        t_input = t_values.unsqueeze(-1)
        return self.omniscient_net(t_input)

    def get_graph_matrices(self):
        T = self.topology_net()
        W = self.weight_net()
        W_sparse = T * W
        return T, W, W_sparse

    def infer_node_times(self, W_sparse, reference_node=0, max_iterations=10):
        n = W_sparse.shape[0]
        device = W_sparse.device

        times = torch.full((n,), float('inf'), device=device)
        times[reference_node] = 0.0

        edge_mask = (W_sparse.abs() > 1e-6)
        rows, cols = edge_mask.nonzero(as_tuple=True)
        weights = W_sparse[rows, cols]

        for _ in range(max_iterations):
            updated_times = times.clone()

            candidate_times = times[rows] + weights
            updated_times = torch.scatter_reduce(
                updated_times, 0, cols, candidate_times, reduce='amin'
            )

            if torch.allclose(times, updated_times, atol=1e-6):
                break

            times = updated_times

        inf_mask = torch.isinf(times)
        if inf_mask.any():
            degrees = edge_mask.sum(dim=0).float()
            if degrees.std() > 1e-6:
                default_times = (degrees - degrees.mean()) / degrees.std()
            else:
                default_times = torch.zeros_like(degrees)
            times = torch.where(inf_mask, default_times, times)

        return times
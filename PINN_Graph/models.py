import torch
import torch.nn as nn


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
    def __init__(self, n_nodes, eigengene_data=None):
        super().__init__()
        T_init = torch.zeros(n_nodes, n_nodes)

        node_order = torch.randperm(n_nodes)
        for i in range(n_nodes - 1):
            src = node_order[i]
            dst = node_order[i + 1]
            T_init[src, dst] = 1.0

        self.T = nn.Parameter(T_init + torch.randn(n_nodes, n_nodes) * 0.001)

    def forward(self, threshold=0.5):
        T_logits = self.T
        probs = torch.sigmoid(T_logits)
        T_hard = (probs > threshold).float()
        T = T_hard + (probs - probs.detach())
        T = T * (1 - torch.eye(T.shape[0], device=T.device))
        return T, probs

class WeightNetwork(nn.Module):
    def __init__(self, n_nodes, eigengene_data=None):
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
        if t_values.dim() == 1:
            t_values = t_values.unsqueeze(-1)
        return self.omniscient_net(t_values)
    
    def get_graph_matrices(self):
        T, _ = self.topology_net()
        W = self.weight_net()
        W_sparse = T * W
        return T, W, W_sparse

    def infer_node_times(
            self, 
            W_sparse, 
            reference_node=0, 
            max_iterations=50, 
            eps=1e-6
        ):
        _, probs = self.topology_net()
        n = W_sparse.shape[0]
        device = W_sparse.device
        start = reference_node
        times = torch.zeros(n, device=device)
        times.requires_grad_(True)

        probs = probs * (1 - torch.eye(n, device=device))
        for _ in range(max_iterations):
            t_i = times.unsqueeze(1).expand(-1, n)
            t_j_candidates = t_i + W_sparse
            weight_sum = probs.sum(dim=0).clamp_min(1e-6)
            times_new = (probs * t_j_candidates).sum(dim=0) / weight_sum

            times_new[start] = 0.0
            if torch.allclose(times, times_new, atol=eps):
                break
            times = times_new
            # print(f"Inferred times: {times}")
        return times, probs
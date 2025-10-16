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
    def __init__(self, node_dim, hidden_dim=64, temperature=2.0):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        with torch.no_grad():
            self.edge_predictor[-1].bias.fill_(-2.0)
        
        self.temperature = temperature

    def forward(self, node_features):
        n = node_features.shape[0]
        node_emb = self.node_encoder(node_features)

        node_emb_i = node_emb.unsqueeze(1).expand(-1, n, -1)
        node_emb_j = node_emb.unsqueeze(0).expand(n, -1, -1)

        edge_inputs = torch.cat([node_emb_i, node_emb_j], dim=-1)

        T_logits = self.edge_predictor(edge_inputs.view(-1, 2 * node_emb.shape[-1]))
        T_logits = T_logits.view(n, n).squeeze(-1)
        
        T = torch.sigmoid(T_logits / self.temperature)
        T = T * (1 - torch.eye(n, device=node_features.device))

        return T

class WeightNetwork(nn.Module):
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features):
        n = node_features.shape[0]
        node_emb = self.node_encoder(node_features)

        node_emb_i = node_emb.unsqueeze(1).expand(-1, n, -1)
        node_emb_j = node_emb.unsqueeze(0).expand(n, -1, -1)

        edge_inputs = torch.cat([node_emb_i, node_emb_j], dim=-1)

        W = self.edge_predictor(edge_inputs.view(-1, 2 * node_emb.shape[-1]))
        W = W.view(n, n).squeeze(-1)

        W = W * (1 - torch.eye(n, device=node_features.device))

        return W

class TemporalGraphPINN(nn.Module):
    def __init__(self, n_nodes, node_dim, n_eigengenes=5):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_eigengenes = n_eigengenes

        self.omniscient_net = OmniscientNetwork(1, n_eigengenes=n_eigengenes)
        self.topology_net = TopologyNetwork(node_dim)
        self.weight_net = WeightNetwork(node_dim)

        self.node_features = nn.Parameter(self._initialize_sparse_node_features(n_nodes, node_dim))

    def _initialize_sparse_node_features(self, n_nodes, node_dim):
        node_order = torch.randperm(n_nodes)
        node_features = torch.randn(n_nodes, node_dim)
        
        linked_list_strength = 2.0
        
        for i in range(n_nodes - 1):
            node_a = node_order[i]
            node_b = node_order[i + 1]
            
            avg_features = (node_features[node_a] + node_features[node_b]) / 2
            node_features[node_a] = (node_features[node_a] + avg_features * linked_list_strength) / (1 + linked_list_strength)
            node_features[node_b] = (node_features[node_b] + avg_features * linked_list_strength) / (1 + linked_list_strength)
        
        return node_features

    def forward(self, t_values):
        t_input = t_values.unsqueeze(-1)
        return self.omniscient_net(t_input)

    def get_graph_matrices(self, sparsity_threshold=0.3):
        T = self.topology_net(self.node_features)
        W = self.weight_net(self.node_features)

        T_sparse = (T > sparsity_threshold).float()
        W_sparse = T_sparse * W

        return T_sparse, W, W_sparse

    def infer_node_times(self, W_sparse, reference_node=0, max_iterations=5):
        n = W_sparse.shape[0]
        device = W_sparse.device

        times = torch.full((n,), float('inf'), device=device)
        times[reference_node] = 0.0

        for iteration in range(max_iterations):

            edge_mask = W_sparse.abs() > 1e-6
            potential_times = times.unsqueeze(1) + W_sparse
            potential_times = torch.where(edge_mask, potential_times, float('inf'))

            min_times = torch.min(potential_times, dim=0)[0]
            times = torch.min(times, min_times)

        inf_mask = times == float('inf')
        if inf_mask.any():
            connectivity_scores = (W_sparse.abs() > 1e-6).sum(dim=0).float()

            if connectivity_scores.std() > 1e-6:
                default_times = (connectivity_scores - connectivity_scores.mean()) / connectivity_scores.std()
            else:
                default_times = torch.zeros_like(connectivity_scores)

            times = torch.where(inf_mask, default_times, times)

        return times
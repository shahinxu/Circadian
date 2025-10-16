"""
PINN Temporal Graph Learning Framework

A physics-informed neural network framework for learning temporal relationships
through graph-structured constraints, specifically designed for circadian rhythm analysis.

Core Components:
- PINN: Physics-informed neural network for temporal modeling
- Graph Constraints: Directed tree structure with sign consistency
- Dual Matrix Framework: Topology matrix + Dense weight matrix
"""

import torch
import torch.nn as nn
import math

class OmniscientNetwork(nn.Module):
    """
    全知全能网络 (Omniscient Network)
    输入时间t，输出对应时间的eigengene expression
    """
    def __init__(self, input_dim, hidden_dim=128, n_eigengenes=5):
        super().__init__()
        self.n_eigengenes = n_eigengenes

        # Simple MLP network - input is time t only
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_eigengenes)
        )

    def forward(self, t):
        """
        t: time values, shape (batch_size, 1) or (batch_size,)
        Returns: eigengene expressions, shape (batch_size, n_eigengenes)
        """
        # Ensure t has shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        return self.net(t)

class TopologyNetwork(nn.Module):
    """
    Learns the topology matrix T (edge existence probabilities)
    """
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features):
        """
        node_features: (n_nodes, node_dim)
        Returns: topology matrix T (n_nodes, n_nodes) with values in [0,1]
        """
        n = node_features.shape[0]
        node_emb = self.node_encoder(node_features)  # (n, hidden_dim)

        # Predict edge probabilities for each pair
        T = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:  # No self-loops
                    edge_input = torch.cat([node_emb[i], node_emb[j]], dim=-1)
                    T[i, j] = self.edge_predictor(edge_input).squeeze()

        return T

class WeightNetwork(nn.Module):
    """
    Learns the dense weight matrix W (edge strengths)
    """
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features):
        """
        node_features: (n_nodes, node_dim)
        Returns: weight matrix W (n_nodes, n_nodes)
        """
        n = node_features.shape[0]
        node_emb = self.node_encoder(node_features)

        W = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_input = torch.cat([node_emb[i], node_emb[j]], dim=-1)
                    W[i, j] = self.edge_predictor(edge_input).squeeze()

        return W

class TemporalGraphPINN(nn.Module):
    """
    Complete PINN + Graph framework for unsupervised temporal learning

    Learns both:
    1. Omniscient network: time → eigengene expression
    2. Graph structure: infers time relationships from eigengene patterns
    """
    def __init__(self, n_nodes, node_dim, n_eigengenes=5):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_eigengenes = n_eigengenes

        # Core networks
        self.omniscient_net = OmniscientNetwork(1, n_eigengenes=n_eigengenes)
        self.topology_net = TopologyNetwork(node_dim)
        self.weight_net = WeightNetwork(node_dim)

        # Node features (learned or fixed) - represent eigengene patterns
        self.node_features = nn.Parameter(torch.randn(n_nodes, node_dim))

    def forward(self, t_values):
        """
        t_values: time points to evaluate, shape (batch_size,)
        Returns: expressions at those times, shape (batch_size, n_eigengenes)
        """
        t_input = t_values.unsqueeze(-1)  # (batch, 1)
        return self.omniscient_net(t_input)

    def get_graph_matrices(self):
        """
        Returns: T (topology), W (weights), W_sparse (T ⊙ W)
        """
        T = self.topology_net(self.node_features)
        W = self.weight_net(self.node_features)
        W_sparse = T * W  # Element-wise multiplication

        return T, W, W_sparse

    def infer_node_times(self, W_sparse, reference_node=0):
        """
        Infer time values for each node based on graph structure
        Starting from reference node (time = 0), propagate times via edge weights

        Args:
            W_sparse: sparse weight matrix (n_nodes, n_nodes)
            reference_node: node with known time (t=0)

        Returns:
            times: inferred time for each node (n_nodes,)
        """
        n = W_sparse.shape[0]
        times = torch.full((n,), float('inf'))  # Initialize with infinity
        times[reference_node] = 0.0  # Reference time

        # Use Bellman-Ford like algorithm to compute shortest paths
        # Edge weights represent time differences
        for _ in range(n - 1):  # Relax all edges n-1 times
            for i in range(n):
                for j in range(n):
                    if W_sparse[i, j].abs() > 1e-6:  # Edge exists
                        # Time propagation: t_j = t_i + w_ij
                        new_time = times[i] + W_sparse[i, j]
                        if new_time < times[j]:  # Update if shorter path found
                            times[j] = new_time

        # Handle disconnected components (set to reasonable defaults)
        inf_mask = times == float('inf')
        if inf_mask.any():
            # For disconnected nodes, set time based on eigengene similarity
            # This is a simplified approach
            times[inf_mask] = torch.randn(inf_mask.sum()) * 0.1

        return times
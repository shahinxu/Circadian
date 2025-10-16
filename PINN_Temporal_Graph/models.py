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
    def __init__(self, node_dim, hidden_dim=64, temperature=2.0):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize the final layer with negative bias to encourage sparsity
        with torch.no_grad():
            self.edge_predictor[-1].bias.fill_(-2.0)  # Strong negative bias
        
        self.temperature = temperature

    def forward(self, node_features):
        """
        node_features: (n_nodes, node_dim)
        Returns: topology matrix T (n_nodes, n_nodes) with values in [0,1]
        """
        n = node_features.shape[0]
        node_emb = self.node_encoder(node_features)  # (n, hidden_dim)

        # Vectorized edge prediction: create all pairs at once
        # node_emb[i] and node_emb[j] for all i,j pairs
        node_emb_i = node_emb.unsqueeze(1).expand(-1, n, -1)  # (n, n, hidden_dim)
        node_emb_j = node_emb.unsqueeze(0).expand(n, -1, -1)  # (n, n, hidden_dim)

        # Concatenate along feature dimension
        edge_inputs = torch.cat([node_emb_i, node_emb_j], dim=-1)  # (n, n, 2*hidden_dim)

        # Predict all edges at once
        T_logits = self.edge_predictor(edge_inputs.view(-1, 2 * node_emb.shape[-1]))  # (n*n, 1)
        T_logits = T_logits.view(n, n).squeeze(-1)  # (n, n)
        
        # Temperature-scaled sigmoid for sharper decisions
        T = torch.sigmoid(T_logits / self.temperature)

        # Remove self-loops
        T = T * (1 - torch.eye(n, device=node_features.device))

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
        node_emb = self.node_encoder(node_features)  # (n, hidden_dim)

        # Vectorized edge prediction: create all pairs at once
        node_emb_i = node_emb.unsqueeze(1).expand(-1, n, -1)  # (n, n, hidden_dim)
        node_emb_j = node_emb.unsqueeze(0).expand(n, -1, -1)  # (n, n, hidden_dim)

        # Concatenate along feature dimension
        edge_inputs = torch.cat([node_emb_i, node_emb_j], dim=-1)  # (n, n, 2*hidden_dim)

        # Predict all edges at once
        W = self.edge_predictor(edge_inputs.view(-1, 2 * node_emb.shape[-1]))  # (n*n, 1)
        W = W.view(n, n).squeeze(-1)  # (n, n)

        # Remove self-loops
        W = W * (1 - torch.eye(n, device=node_features.device))

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
        # Initialize with sparse linked-list structure
        self.node_features = nn.Parameter(self._initialize_sparse_node_features(n_nodes, node_dim))

    def _initialize_sparse_node_features(self, n_nodes, node_dim):
        """
        Initialize node features to encourage sparse linked-list graph structure
        
        Creates a random permutation of nodes and initializes features so that
        adjacent nodes in the permutation have strong connections.
        """
        # Create random node ordering (linked list)
        node_order = torch.randperm(n_nodes)
        
        # Initialize node features randomly
        node_features = torch.randn(n_nodes, node_dim)
        
        # Add structure: make adjacent nodes in the linked list more similar
        # This encourages the topology network to learn edges between them
        linked_list_strength = 2.0
        
        for i in range(n_nodes - 1):
            node_a = node_order[i]
            node_b = node_order[i + 1]
            
            # Make these nodes more similar by averaging their features slightly
            avg_features = (node_features[node_a] + node_features[node_b]) / 2
            node_features[node_a] = (node_features[node_a] + avg_features * linked_list_strength) / (1 + linked_list_strength)
            node_features[node_b] = (node_features[node_b] + avg_features * linked_list_strength) / (1 + linked_list_strength)
        
        return node_features

    def forward(self, t_values):
        """
        t_values: time points to evaluate, shape (batch_size,)
        Returns: expressions at those times, shape (batch_size, n_eigengenes)
        """
        t_input = t_values.unsqueeze(-1)  # (batch, 1)
        return self.omniscient_net(t_input)

    def get_graph_matrices(self, sparsity_threshold=0.3):
        """
        Returns: T (topology), W (weights), W_sparse (T ⊙ W)

        Args:
            sparsity_threshold: threshold for topology matrix (higher = sparser graph)
        """
        T = self.topology_net(self.node_features)
        W = self.weight_net(self.node_features)

        # Apply threshold to make topology sparser
        T_sparse = (T > sparsity_threshold).float()

        # Element-wise multiplication with sparse topology
        W_sparse = T_sparse * W

        return T_sparse, W, W_sparse

    def infer_node_times(self, W_sparse, reference_node=0, max_iterations=5):
        """
        Fast topological time inference using limited Bellman-Ford iterations
        Considers actual graph topology and paths, but with bounded computation

        Args:
            W_sparse: sparse weight matrix (n_nodes, n_nodes) - only existing edges
            reference_node: node with known time (t=0)
            max_iterations: maximum relaxation iterations (trade-off between accuracy and speed)

        Returns:
            times: inferred time for each node (n_nodes,)
        """
        n = W_sparse.shape[0]
        device = W_sparse.device

        # Initialize times: reference node at 0, others at infinity
        times = torch.full((n,), float('inf'), device=device)
        times[reference_node] = 0.0

        # Limited Bellman-Ford: relax edges for max_iterations
        for iteration in range(max_iterations):
            # Create a copy for this iteration
            new_times = times.clone()

            # For each possible edge (i->j), relax if edge exists
            # Only consider edges that actually exist (non-zero in W_sparse)
            edge_mask = W_sparse.abs() > 1e-6

            # Compute potential times for all edges simultaneously
            # potential_times[i,j] = times[i] + W_sparse[i,j] if edge exists
            potential_times = times.unsqueeze(1) + W_sparse  # (n, n)

            # Only keep valid edges, set invalid ones to infinity
            potential_times = torch.where(edge_mask, potential_times, float('inf'))

            # For each target node j, take the minimum over all source nodes i
            # This gives the best time estimate considering all incoming paths
            min_times = torch.min(potential_times, dim=0)[0]  # (n,)

            # Update times: keep the better (smaller) time estimates
            times = torch.min(times, min_times)

        # Handle disconnected components: assign reasonable default times
        inf_mask = times == float('inf')
        if inf_mask.any():
            # For disconnected nodes, use a simple heuristic based on graph structure
            # Count incoming edges as a proxy for "lateness"
            connectivity_scores = (W_sparse.abs() > 1e-6).sum(dim=0).float()  # Incoming edges

            if connectivity_scores.std() > 1e-6:
                default_times = (connectivity_scores - connectivity_scores.mean()) / connectivity_scores.std()
            else:
                default_times = torch.zeros_like(connectivity_scores)

            times = torch.where(inf_mask, default_times, times)

        return times
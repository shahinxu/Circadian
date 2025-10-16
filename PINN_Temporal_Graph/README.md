# PINN Temporal Graph Learning Framework

A physics-informed neural network (PINN) framework for learning temporal relationships in circadian rhythm data through differentiable graph structures.

## Overview

This framework addresses the limitations of existing approaches that either:
- Focus on single cycles (autoencoders) or multi-cycle sorting (greedy ranking)
- Lack sophisticated fusion between discrete ranking and continuous learning
- Don't enforce temporal constraints during training

## Key Features

### 🔬 Physics-Informed Neural Networks (PINN)
- Enforces circadian periodicity (24-hour cycles)
- Includes derivative constraints for smoothness
- Learns continuous temporal patterns

### 📊 Dual Matrix Framework
- **Topology Matrix (T)**: Learns which edges exist (sparsity)
- **Weight Matrix (W)**: Learns edge strengths and signs
- **Sparse Weight Matrix**: T ⊙ W for efficient computation

### 🌳 Directed Tree Constraints
- Flexible root node selection
- Single parent per node (tree structure)
- Acyclicity guarantees
- Sign consistency along paths

### 🎯 Sign Consistency Constraints
- Minimizes sign flips along temporal paths
- Ensures coherent phase relationships
- Differentiable optimization

## Installation

```bash
# Clone repository
cd /path/to/Circadian
mkdir PINN_Temporal_Graph
cd PINN_Temporal_Graph

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib scipy pyyaml
pip install torch-geometric  # For graph operations
```

## Usage

### Training

```bash
# Train with synthetic data
python train.py --n_nodes 10 --n_eigengenes 5 --n_epochs 1000

# Train with real data
python train.py --data_path ../data/expression.csv --n_epochs 2000
```

### Key Parameters

- `--n_nodes`: Number of nodes in temporal graph
- `--n_eigengenes`: Number of eigengenes to learn
- `--n_epochs`: Training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate

## Architecture

### Core Components

1. **TemporalPINN**: Neural network for time → expression mapping
   - Fourier features for periodic functions
   - Physics constraints (periodicity, smoothness)

2. **TopologyNetwork**: Learns graph structure existence
   - Predicts edge probabilities between nodes
   - Sigmoid activation for [0,1] probabilities

3. **WeightNetwork**: Learns edge strengths
   - Predicts signed edge weights
   - No activation (unconstrained weights)

4. **TemporalGraphPINN**: Complete framework
   - Combines PINN with graph learning
   - Dual matrix optimization

### Loss Functions

- **Physics Loss**: Periodicity + derivative constraints
- **Graph Structure Loss**: Tree constraints + acyclicity
- **Sign Consistency Loss**: Path-based sign coherence
- **Reconstruction Loss**: Data fitting

## Mathematical Formulation

### PINN Constraints
```
∂E/∂t(t) ≈ ∂E/∂t(t + 24)    # Periodicity
‖∂²E/∂t²‖² → 0               # Smoothness
```

### Graph Constraints
```
T ∈ [0,1]^{n×n}                # Topology matrix
W ∈ ℝ^{n×n}                   # Weight matrix
W_sparse = T ⊙ W              # Sparse weights

Single parent: ∀i≠root: ∑T[:,i] ≤ 1
Acyclicity: No cycles in directed graph
Sign consistency: Minimize sign flips on paths
```

## Data Format

Expected input format:
```
time,gene1,gene2,gene3,...
0.0,1.2,0.8,1.5,...
1.0,1.1,0.9,1.4,...
2.0,1.0,1.0,1.3,...
...
```

## Results Visualization

The framework provides several visualization tools:

- **Temporal Predictions**: Model fit vs true data
- **Graph Structure**: Learned adjacency matrix
- **Training History**: Loss curves
- **Graph Properties**: Sparsity, connectivity metrics

## Advanced Usage

### Custom Constraints

```python
from losses import flexible_directed_tree_constraint_loss, path_sign_consistency_loss

# Custom graph constraints
T, W, W_sparse = model.get_graph_matrices()
tree_loss = flexible_directed_tree_constraint_loss(T)
sign_loss = path_sign_consistency_loss(W_sparse)
```

### Integration with Existing Methods

```python
# Load greedy ranking results
greedy_order = load_greedy_ranking('greedy_results.csv')

# Initialize graph with ranking priors
initial_adjacency = create_graph_from_ranking(greedy_order)
model.initialize_graph(initial_adjacency)
```

## Citation

If you use this framework in your research, please cite:

```
PINN Temporal Graph Learning for Circadian Rhythm Analysis
- Physics-informed constraints for temporal modeling
- Differentiable graph structure learning
- Dual matrix framework for topology and weights
```

## License

This project is part of the broader Circadian rhythm analysis toolkit.
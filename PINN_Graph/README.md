# PINN Temporal Graph Learning Framework

A physics-informed neural network (PINN) framework for learning temporal relationships in circadian rhythm data through differentiable graph structures.

## Quick Start

### 1. One-Command Setup and Run
```bash
chmod +x run.sh
./run.sh
```

### 2. Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (optional)
python generate_data.py

# Run training
python train.py --config config.yaml
```

## Usage

### Training with Synthetic Data
```bash
# Use default config
python train.py --config config.yaml

# Use custom config
python train.py --config your_config.yaml
```

### Training with Real Data
```bash
# Your data should be CSV format with columns: Time, Gene1, Gene2, ...
python train.py --config config.yaml --data_path path/to/your/expression_data.csv
```

### Generate Custom Synthetic Data
```bash
# Generate 200 samples with 8 eigengenes
python generate_data.py  # Uses defaults: 100 samples, 5 eigengenes

# Modify generate_data.py parameters for custom data
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  n_nodes: 10          # Number of samples/nodes in graph
  node_dim: 16         # Node feature dimension
  n_eigengenes: 5      # Number of eigengenes to learn

training:
  n_epochs: 1000       # Training epochs
  batch_size: 32
  learning_rate: 0.001

loss_weights:
  reconstruction: 1.0  # Data reconstruction weight
  physics: 1.0         # PINN physics constraints weight
  graph: 1.0           # Graph structure constraints weight
```

## Output

Training results are saved in `experiments/YYYYMMDD_HHMMSS/` directory:
- `training_history.png` - Loss curves
- `graph_structure.png` - Learned graph adjacency matrix
- `temporal_predictions.png` - Model predictions vs true data
- `inferred_times.npy` - Inferred time values for each sample
- `final_model.pth` - Trained model weights

## Architecture

### Core Components

1. **Omniscient Network**: Time → Eigengene expression prediction
2. **Topology Network**: Learns which edges exist (sparsity)
3. **Weight Network**: Learns edge strengths and signs
4. **Time Inference**: Graph-based temporal ordering from eigengene patterns

### Loss Functions

- **Reconstruction Loss**: Network predictions vs true eigengene data
- **Physics Loss**: Network derivatives vs graph-based derivatives
- **Tree Loss**: Graph structure constraints (connectivity, acyclicity)
- **Sign Consistency Loss**: Temporal coherence along graph paths

## Data Format

Expected CSV format:
```
Time,Eigengene_1,Eigengene_2,Eigengene_3,Eigengene_4,Eigengene_5
0.0,-0.05,1.72,1.33,-0.41,-1.05
1.0,0.01,1.88,0.97,-0.86,-1.15
...
```

## Advanced Usage

### Custom Loss Weights
```python
from losses import total_unsupervised_loss

# Custom weights
loss, components = total_unsupervised_loss(
    model, data, device,
    lambda_recon=2.0,    # Stronger reconstruction
    lambda_physics=0.5,  # Weaker physics constraints
    lambda_tree=1.0,     # Standard graph constraints
    lambda_sign=1.0      # Standard sign consistency
)
```

### Model Inspection
```python
# Get learned graph structure
T, W, W_sparse = model.get_graph_matrices()

# Get inferred times
inferred_times = model.infer_node_times(W_sparse)

# Visualize
plot_graph_structure(W_sparse)
plot_temporal_predictions(model, time_points, true_data, device)
```

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

## Citation

If you use this framework in your research, please cite:

```
PINN Temporal Graph Learning for Circadian Rhythm Analysis
- Physics-informed constraints for temporal modeling
- Differentiable graph structure learning
- Dual matrix framework for topology and weights
```
"""
Test script to verify PINN Temporal Graph framework with synthetic eigengene data
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

from models import TemporalGraphPINN
from losses import total_loss
from data import CircadianDataset
from utils import plot_temporal_predictions, analyze_graph_properties

def test_with_synthetic_data():
    """
    Test the PINN framework with synthetic eigengene data
    """
    print("Testing PINN Temporal Graph Framework")
    print("=" * 50)

    # Load synthetic data
    data_path = Path('data/synthetic_eigengenes_100x5.csv')
    if not data_path.exists():
        print("Error: Synthetic data not found. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    time_points = df['Time'].values
    eigengenes = df.drop('Time', axis=1).values

    print(f"Loaded data: {eigengenes.shape[0]} samples, {eigengenes.shape[1]} eigengenes")
    print(f"Time range: {time_points[0]:.1f} - {time_points[-1]:.1f} hours")

    # Create dataset
    dataset = CircadianDataset(eigengenes, time_points, normalize=True)
    print(f"Dataset created with {len(dataset)} samples")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGraphPINN(
        n_nodes=5,  # Same as number of eigengenes
        node_dim=8,
        n_eigengenes=5
    ).to(device)

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {device}")

    # Quick training test (just a few steps)
    print("\nRunning quick training test...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 16
    n_test_steps = 50

    model.train()
    for step in range(n_test_steps):
        # Sample random batch
        indices = torch.randperm(len(dataset))[:batch_size]
        t_batch = torch.stack([dataset[i][0] for i in indices])
        expr_batch = torch.stack([dataset[i][1] for i in indices])

        t_batch = t_batch.to(device)
        expr_batch = expr_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(t_batch)

        # Reconstruction loss
        recon_loss = torch.nn.MSELoss()(predictions, expr_batch)

        # Compute losses
        total_loss_val, loss_components = total_loss(
            model, None, device,  # Pass None for data_loader since we're computing physics/graph losses only
            lambda_physics=0.1, lambda_graph=0.1, lambda_recon=0.0
        )

        # Add reconstruction loss
        total_loss_val += recon_loss

        # Backward pass
        total_loss_val.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Total Loss = {total_loss_val:.4f}")
            print(f"  Reconstruction: {recon_loss:.4f}")
            print(f"  Physics: {loss_components['physics']:.4f}")
            print(f"  Graph: {loss_components['graph']:.4f}")

    # Analyze learned graph
    print("\nAnalyzing learned graph structure...")
    model.eval()
    T, W, W_sparse = model.get_graph_matrices()

    graph_props = analyze_graph_properties(T, W_sparse)
    print("Graph Properties:")
    for key, value in graph_props.items():
        print(f"  {key}: {value}")

    # Generate predictions for visualization
    print("\nGenerating predictions for visualization...")
    t_test = torch.linspace(0, 24, 100).unsqueeze(-1).to(device)
    with torch.no_grad():
        predictions = model(t_test).cpu().numpy()

    # Plot results
    plot_temporal_predictions(
        model, torch.linspace(0, 24, 100),
        eigengenes[:100], device,
        save_path='data/test_results.png'
    )

    print("\nTest completed successfully!")
    print("Results saved to data/test_results.png")

if __name__ == "__main__":
    test_with_synthetic_data()
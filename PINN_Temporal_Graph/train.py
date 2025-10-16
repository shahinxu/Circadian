import torch
import torch.optim as optim
import argparse
from tqdm import tqdm

from models import TemporalGraphPINN
from losses import compute_loss
from utils import (
    load_config, set_random_seed, create_experiment_dir,
    save_checkpoint, early_stopping, plot_graph_structure
)
from data import load_expression_data

def Trainer(model: TemporalGraphPINN, eigengene_data, config, device):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config['training']['patience'] // 2,
        factor=0.5
    )

    best_loss = float('inf')
    train_losses = []
    loss_components_history = []

    exp_dir = create_experiment_dir(
        config['experiment']['save_dir'],
        config['experiment']['name']
    )

    n_epochs = config['training']['n_epochs']
    patience = config['training']['patience']
    min_delta = float(config['training']['min_delta'])

    for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        total_loss, loss_components = compute_loss(
            model, eigengene_data, device,
            lambda_recon=config['loss_weights']['reconstruction'],
            lambda_physics=config['loss_weights']['physics'],
            lambda_tree=config['loss_weights']['tree'],
            lambda_sign=config['loss_weights']['sign_consistency'],
            lambda_sparsity=config['loss_weights']['sparsity']
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.detach())

        train_losses.append(total_loss.item())
        loss_components_history.append(loss_components)

        if (epoch + 1) % 10 == 0:
            tqdm.write(
                f"Epoch {epoch+1:04d} | "
                f"Loss={total_loss.item():.4f} | "
                f"Recon={loss_components['reconstruction']:.4f} | "
                f"Phys={loss_components['physics']:.4f} | "
                f"Sparse={loss_components['sparsity']:.4f}"
            )

        if config['experiment']['save_checkpoints'] and total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(model, optimizer, epoch, total_loss, exp_dir / 'best_model.pth')

        # if early_stopping(train_losses, patience, min_delta):
        #     tqdm.write(f"Early stopping at epoch {epoch+1}")
        #     break

    return train_losses, loss_components_history, exp_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    device_str = config['system']['device']
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    data_path = config['system']['data_path']
    if data_path:
        eigengene_data, _ = load_expression_data(data_path)
        eigengene_data = torch.tensor(eigengene_data, dtype=torch.float32)
    else:
        from generate_data import generate_circadian_eigengenes
        _, eigengene_data = generate_circadian_eigengenes(
            n_samples=config['data']['n_samples'],
            n_eigengenes=config['model']['n_eigengenes']
        )
        eigengene_data = torch.tensor(eigengene_data, dtype=torch.float32)

    n_samples, n_eigengenes = eigengene_data.shape

    model = TemporalGraphPINN(
        n_nodes=n_samples,
        n_eigengenes=n_eigengenes
    ).to(device)

    set_random_seed(42)

    _, _, exp_dir = Trainer(
        model, eigengene_data, config, device
    )

    _, _, W_sparse = model.get_graph_matrices()
    inferred_times = model.infer_node_times(W_sparse)

    model.eval()

    plot_graph_structure(W_sparse, save_path=exp_dir / 'final_graph.png')

if __name__ == "__main__":
    main()

from utils import plot_comparsion
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
from utils import rank_loss
from AE import PhaseAutoEncoder
from data_load import load_and_preprocess_train_data, load_and_preprocess_test_data
from torch.utils.data import DataLoader
from utils import predict_and_save_phases

def greedy_ordering(components: np.ndarray):
    n = components.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = []
    start = np.argmin(np.sum(np.abs(components), axis=1))
    cur = start
    order.append(cur)
    visited[cur] = True
    for _ in range(n - 1):
        diffs = np.abs(components - components[cur:cur+1, :])
        dists = np.sum(diffs, axis=1)
        dists[visited] = np.inf
        nxt = np.argmin(dists)
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return np.array(order, dtype=int)

def plot_components_by_phase(components, phases, save_path, n_plot=None):
    order = np.argsort(phases)
    phases_sorted = phases[order]
    comp_sorted = components[order]
    n_components = comp_sorted.shape[1]
    if n_plot is None:
        n_plot = n_components
    plt.figure(figsize=(8, 6))
    for i in range(n_plot):
        plt.plot(phases_sorted, comp_sorted[:, i], label=f'PC{i+1}')
    plt.xlabel('Predicted phase')
    plt.title('Expression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def train_model(
    model: PhaseAutoEncoder, train_dataset, preprocessing_info,
    num_epochs=100, lr=1e-3, device='cuda',
    lambda_recon=0.2, lambda_time=0.0, lambda_align=1.0,
    save_dir='./model_checkpoints', stage1_frac=0.8):

    if 'train_has_time' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_time'] = 'time' in sample
    
    if 'train_has_celltype' not in preprocessing_info:
        sample = train_dataset[0]
        preprocessing_info['train_has_celltype'] = 'celltype' in sample
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=100
    )
    recon_criterion = nn.MSELoss()
    train_losses = []
    os.makedirs(save_dir, exist_ok=True)

    all_expressions = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_expressions.append(sample['expression'])
    expressions_tensor = torch.stack(all_expressions).to(device)
    components = expressions_tensor.cpu().numpy().astype(np.float32)
    n_samples = components.shape[0]
    order = greedy_ordering(components)
    ranks = np.empty(n_samples, dtype=np.int64)
    ranks[order] = np.arange(n_samples)
    X = expressions_tensor

    stage1_epochs = int(num_epochs * stage1_frac)
    stage2_epochs = num_epochs - stage1_epochs

    for epoch in range(stage1_epochs):
        model.train()
        optimizer.zero_grad()
        _, pred_phases, recon = model(X)
        recon_loss = recon_criterion(recon, X)
        a_loss = rank_loss(pred_phases, ranks, window=5)
        total = lambda_recon * recon_loss + lambda_align * a_loss
        total.backward()
        optimizer.step()
        train_losses.append(total.item())
        scheduler.step(total.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{stage1_epochs}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"Align Loss: {a_loss.item():.4f}, "
                    f"Total Loss: {total.item():.4f}")

    for epoch in range(stage2_epochs):
        model.train()
        optimizer.zero_grad()
        _, pred_phases, recon = model(X)
        recon_loss = recon_criterion(recon, X)
        total = lambda_recon * recon_loss
        total.backward()
        optimizer.step()
        train_losses.append(total.item())
        scheduler.step(total.item())

    model.eval()
    with torch.no_grad():
        _, pred_phases, recon = model(X)
    return model, order


def evaluate_order_plot(
        pred_order, 
        preprocessing_info, 
        metadata_path, 
        save_dir, 
        period_hours=24.0
    ):
    if not os.path.isfile(metadata_path):
        return None

    n_samples = len(pred_order)
    pred_hours = (np.arange(n_samples) / n_samples) * period_hours
    sample_names = preprocessing_info.get('sample_columns', [])
    order_df = pd.DataFrame({
        'Sample_ID': sample_names,
        'Predicted_Order': pred_order,
        'Predicted_Phase_Hours': pred_hours
    })
    save_greedy = os.path.join(save_dir + '_greedy') if save_dir else save_dir + '_greedy'
    return plot_comparsion(order_df, metadata_path, save_greedy)


def evaluate_test_set(
        model, 
        test_file, 
        preprocessing_info, 
        save_dir, 
        device='cuda', 
        metadata_path=None
    ):
    if not os.path.isfile(test_file):
        print("No test file provided; skipping test set evaluation.")
        return None

    test_dataset, test_preprocessing_info = load_and_preprocess_test_data(
        test_file, preprocessing_info
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    results_df = predict_and_save_phases(
        model=model,
        test_loader=test_loader,
        preprocessing_info=test_preprocessing_info,
        device=device,
        save_dir=save_dir
    )

    if metadata_path and os.path.isfile(metadata_path) and results_df is not None:
        plot_comparsion(results_df, metadata_path, save_dir)

    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--stage1_frac", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=0.01)
    parser.add_argument("--lambda_align", type=float, default=1)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    base_data = f"../data/{args.dataset_path}"
    train_file = os.path.join(base_data, "expression.csv")
    test_file = os.path.join(base_data, "expression.csv")
    metadata = os.path.join(base_data, "metadata.csv")
    save_dir = os.path.join("results", args.dataset_path)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    os.makedirs(save_dir, exist_ok=True)

    train_dataset, preprocessing_info = load_and_preprocess_train_data(
        train_file, n_components=args.n_components
    )

    preprocessing_info['period_hours'] = args.period_hours

    model = PhaseAutoEncoder(
        input_dim=args.n_components,
        dropout=args.dropout
    )

    model, pred_order = train_model(
        model,
        train_dataset=train_dataset,
        preprocessing_info=preprocessing_info,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        lambda_recon=args.lambda_recon,
        lambda_align=args.lambda_align,
        save_dir=save_dir,
        stage1_frac=args.stage1_frac
    )

    evaluate_order_plot(
        pred_order, 
        preprocessing_info, 
        metadata, 
        save_dir, 
        period_hours=args.period_hours
    )

    results_df = evaluate_test_set(
        model, 
        test_file, 
        preprocessing_info, 
        save_dir, 
        device=args.device, 
        metadata_path=metadata
    )
    if results_df is None and not os.path.isfile(test_file):
        print("Training completed. No test file provided, so no predictions made.")


if __name__ == '__main__':
    main()

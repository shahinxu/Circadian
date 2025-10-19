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
from tqdm import tqdm

from AE import PhaseAutoEncoder
from dataset import load_and_preprocess_train_data, load_and_preprocess_test_data
from torch.utils.data import DataLoader
from utils import predict_and_save_phases

def greedy_ordering(components: np.ndarray):
    n, d = components.shape
    visited = np.zeros(n, dtype=bool)
    order = []
    start = np.argmin(np.sum(np.abs(components), axis=1))
    cur = start
    order.append(cur)
    visited[cur] = True
    for _ in range(n - 1):
        # compute L1 distances to unvisited
        diffs = np.abs(components - components[cur:cur+1, :])
        dists = np.sum(diffs, axis=1)
        dists[visited] = np.inf
        nxt = np.argmin(dists)
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return np.array(order, dtype=int)

def optimal_ordering_dp(components: np.ndarray):
    n, d = components.shape

    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sum(np.abs(components[i] - components[j]))
            dists[i, j] = dists[j, i] = dist

    start_node = np.argmin(np.sum(np.abs(components), axis=1))
    num_masks = 1 << n
    dp = np.full((num_masks, n), np.inf)
    parent = np.full((num_masks, n), -1, dtype=int)

    dp[1 << start_node, start_node] = 0

    # 4. 填充 DP 表
    for mask in range(1, num_masks):
        for j in range(n):
            if (mask >> j) & 1:
                prev_mask = mask ^ (1 << j)
                if prev_mask == 0:
                    continue

                for i in range(n):
                    if (prev_mask >> i) & 1:
                        new_dist = dp[prev_mask, i] + dists[i, j]
                        if new_dist < dp[mask, j]:
                            dp[mask, j] = new_dist
                            parent[mask, j] = i

    final_mask = num_masks - 1
    
    last_node = np.argmin(dp[final_mask])

    order = []
    cur_node = last_node
    cur_mask = final_mask

    for _ in range(n):
        order.append(cur_node)
        prev_node = parent[cur_mask, cur_node]
        cur_mask ^= (1 << cur_node)
        cur_node = prev_node
    return np.array(order[::-1], dtype=int)


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
    model, train_dataset, preprocessing_info,
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    recon_criterion = nn.MSELoss()
    train_losses = []
    os.makedirs(save_dir, exist_ok=True)

    # === 预训练阶段（greedy排序+对齐loss） ===
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
    target_phases = (ranks.astype(np.float32) / n_samples) * (2 * np.pi)
    X = expressions_tensor
    target_phases_t = torch.from_numpy(target_phases).to(device)

    def align_loss(pred_phases, target_phases):
        diff = pred_phases - target_phases
        return torch.mean(1.0 - torch.cos(diff))

    stage1_epochs = int(num_epochs * stage1_frac)
    stage2_epochs = num_epochs - stage1_epochs

    print(f"[Pretrain] Stage1: {stage1_epochs} epochs, Stage2: {stage2_epochs} epochs")
    with tqdm(total=stage1_epochs, desc="Pretrain", disable=False) as pbar:
        for epoch in range(stage1_epochs):
            model.train()
            optimizer.zero_grad()
            _, pred_phases, recon = model(X, None)
            recon_loss = recon_criterion(recon, X)
            a_loss = align_loss(pred_phases, target_phases_t)
            total = lambda_recon * recon_loss + lambda_align * a_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(total.item())
            scheduler.step(total.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{stage1_epochs}, "
                      f"Recon Loss: {recon_loss.item():.4f}, "
                      f"Align Loss: {a_loss.item():.4f}, "
                      f"Total Loss: {total.item():.4f}")
            # pbar.update(1)

    # === 微调阶段（只重建loss） ===
    with tqdm(total=stage2_epochs, desc="Finetune", disable=False) as pbar:
        for epoch in range(stage2_epochs):
            model.train()
            optimizer.zero_grad()
            _, pred_phases, recon = model(X, None)
            recon_loss = recon_criterion(recon, X)
            total = lambda_recon * recon_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(total.item())
            scheduler.step(total.item())
            pbar.update(1)

    torch.save(model.state_dict(), os.path.join(save_dir, 'two_stage_model.pth'))
    model.eval()
    with torch.no_grad():
        _, pred_phases, recon = model(X, None)
        pred_phases_np = pred_phases.cpu().numpy()
    plot_components_by_phase(components, pred_phases_np, os.path.join(save_dir, 'components_by_predicted_phase.png'))
    return model, order

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--stage1_frac", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=0.001)
    parser.add_argument("--lambda_align", type=float, default=2.0)
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
    save_greedy = save_dir + '_greedy'

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

    if os.path.isfile(metadata):
        n_samples = len(pred_order)
        pred_hours = (np.arange(n_samples) / n_samples) * args.period_hours
        sample_names = preprocessing_info.get('sample_columns', [])
        print(len(sample_names))
        print(len(pred_order))
        print(len(pred_hours))
        order_df = pd.DataFrame({
            'Sample_ID': sample_names,
            'Predicted_Order': pred_order,
            'Predicted_Phase_Hours': pred_hours
        })
        
        order_df.to_csv(os.path.join(save_dir, 'greedy_order.csv'), index=False)
        plot_comparsion(order_df, metadata, save_greedy)

    # 测试集推理
    if os.path.isfile(test_file):
        test_dataset, test_preprocessing_info = load_and_preprocess_test_data(
            test_file, preprocessing_info
        )
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        results_df = predict_and_save_phases(
            model=model,
            test_loader=test_loader,
            preprocessing_info=test_preprocessing_info,
            device=args.device,
            save_dir=save_dir
        )

        if os.path.isfile(metadata):
            plot_comparsion(results_df, metadata, save_dir)
    else:
        print("Training completed. No test file provided, so no predictions made.")


if __name__ == '__main__':
    main()

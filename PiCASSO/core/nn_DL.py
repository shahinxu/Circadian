import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SimpleScoreNet(nn.Module):

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        return out


def simple_neural_optimize(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
    value: Optional[np.ndarray] = None,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str = 'cpu',
    tau: float = 1.0,
    perm_reg: float = 1.0,
    hidden: int = 64) -> np.ndarray:

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    n_samples, n_features = x.shape

    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        if w.numel() != n_features:
            w = w.repeat(n_features)[:n_features]
    else:
        w = torch.ones(n_features, dtype=torch.float32, device=device)

    with torch.no_grad():
        if value is not None:
            v = torch.tensor(value, dtype=torch.float32, device=device).reshape(-1)
            if v.numel() != n_samples:
                raise ValueError('value must have length n_samples')
            vi = v.unsqueeze(1)  # (n,1)
            vj = v.unsqueeze(0)  # (1,n)
            dist_mat = torch.abs(vi - vj)
            dist_mat_per_dim = None
        else:
            x_expanded_i = x_tensor.unsqueeze(1)
            x_expanded_j = x_tensor.unsqueeze(0)
            abs_diff = torch.abs(x_expanded_i - x_expanded_j)
            dist_mat_per_dim = abs_diff * w.view(1, 1, -1)
            dist_mat = torch.sum(dist_mat_per_dim, dim=-1)

    model = SimpleScoreNet(n_features, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        s = model(x_tensor)  # (n,)

        diff = s.unsqueeze(1) - s.unsqueeze(0)  # (n,n)
        logits = - (diff ** 2) / (tau + 1e-12)

        mask = torch.ones_like(logits, device=device)
        mask.fill_diagonal_(0.0)
        neg_inf = torch.tensor(-1e9, device=device)
        masked_logits = torch.where(mask > 0, logits, neg_inf)

        P = F.softmax(masked_logits, dim=1)

        if 'dist_mat_per_dim' in locals() and dist_mat_per_dim is not None:
            per_dim_lengths = torch.sum(P.unsqueeze(-1) * dist_mat_per_dim, dim=(0, 1))
            expected_length = torch.sum(per_dim_lengths)
        else:
            expected_length = torch.sum(P * dist_mat)

        col_sums = torch.sum(P, dim=0)
        perm_penalty = torch.mean((col_sums - 1.0) ** 2)

        loss = expected_length + perm_reg * perm_penalty

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (epoch + 1) % max(1, n_epochs // 5) == 0 or epoch == 0:
            try:
                print(f"[simple_nn] Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, expected_len={expected_length.item():.4f}, perm_pen={perm_penalty.item():.4f}")
            except Exception:
                pass

    model.eval()
    with torch.no_grad():
        s_final = model(x_tensor).detach().cpu().numpy()

    order = np.argsort(s_final)
    ranks = np.zeros(n_samples, dtype=int)
    ranks[order] = np.arange(n_samples)

    return ranks.reshape(-1, 1)


def neural_multi_scale_optimize(
    x: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_epochs: int = 100,
    hidden_dim: int = 128,
    learning_rate: float = 1e-4,
    device: str = 'cpu') -> np.ndarray:
    return simple_neural_optimize(
        x,
        weights=weights,
        n_epochs=n_epochs,
        lr=learning_rate,
        hidden=hidden_dim,
        device=device
    )
